import asyncio
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import path
from subprocess import Popen
from typing import Any, Callable, Dict, List, Optional, Tuple

import google.protobuf
from multiaddr import Multiaddr

import hivemind.hivemind_cli as cli
import hivemind.p2p.p2p_daemon_bindings.p2pclient as p2pclient
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID, StreamInfo
from hivemind.proto import p2pd_pb2
from hivemind.utils import MSGPackSerializer
from hivemind.utils.logging import get_logger
from hivemind.utils.networking import find_open_port

logger = get_logger(__name__)


P2PD_FILENAME = 'p2pd'
NUM_RETRIES = 3
RETRY_DELAY = 0.4


@dataclass(frozen=True)
class P2PContext(object):
    handle_name: str
    local_id: PeerID
    remote_id: PeerID = None
    remote_maddr: Multiaddr = None


class P2P:
    """
    This class is responsible for establishing peer-to-peer connections through NAT and/or firewalls.
    It creates and manages a libp2p daemon in a background process, then terminates it when P2P is shut down.
    In order to communicate, a P2P instance should either use one or more bootstrap_peers that will connect it
    to the rest of the swarm or use the public IPFS network (https://ipfs.io).

    For incoming connections, P2P instances add RPC handlers that may be accessed by other peers:
      - `P2P.add_unary_handler` accepts a protobuf message and returns another protobuf
      - `P2P.add_stream_handler` transfers raw data using bi-directional streaming interface

    To access these handlers, a P2P instance can `P2P.call_unary_handler`/`P2P.call_stream_handler`,
    using the recipient's unique `P2P.id` and the name of the corresponding handler.
    """

    HEADER_LEN = 8
    BYTEORDER = 'big'
    PB_HEADER_LEN = 1
    RESULT_MESSAGE = b'\x00'
    ERROR_MESSAGE = b'\x01'
    DHT_MODE_MAPPING = {
        'dht': {'dht': 1},
        'dht_server': {'dhtServer': 1},
        'dht_client': {'dhtClient': 1},
    }
    FORCE_REACHABILITY_MAPPING = {
        'public': {'forceReachabilityPublic': 1},
        'private': {'forceReachabilityPrivate': 1},
    }

    def __init__(self):
        self._child = None
        self._alive = False
        self._listen_task = None
        self._server_stopped = asyncio.Event()

    @classmethod
    async def create(cls, *args, quic: bool = True, tls: bool = True, conn_manager: bool = True,
                     dht_mode: str = 'dht_server', force_reachability: Optional[str] = None,
                     nat_port_map: bool = True, auto_nat: bool = True,
                     bootstrap_peers: Optional[List[Multiaddr]] = None,
                     use_ipfs: bool = False, external_port: int = None,
                     daemon_listen_port: int = None, use_relay: bool = True, use_relay_hop: bool = False,
                     use_relay_discovery: bool = False, use_auto_relay: bool = False, relay_hop_limit: int = 0,
                     **kwargs) -> 'P2P':
        """
        Start a new p2pd process and connect to it.
        :param quic: Enables the QUIC transport
        :param tls: Enables TLS1.3 channel security protocol
        :param conn_manager: Enables the Connection Manager
        :param dht_mode: DHT mode (dht_client/dht_server/dht)
        :param force_reachability: Force reachability mode (public/private)
        :param nat_port_map: Enables NAT port mapping
        :param auto_nat: Enables the AutoNAT service
        :param bootstrap: Connects to bootstrap peers and bootstraps the dht if enabled
        :param bootstrap_peers: List of bootstrap peers
        :param use_ipfs: Bootstrap to IPFS (works only if bootstrap=True and bootstrap_peers=None)
        :param external_port: port for external connections from other p2p instances
        :param daemon_listen_port: port for connection daemon and client binding
        :param use_relay: enables circuit relay
        :param use_relay_hop: enables hop for relay
        :param use_relay_discovery: enables passive discovery for relay
        :param use_auto_relay: enables autorelay
        :param relay_hop_limit: sets the hop limit for hop relays
        :param args: positional CLI arguments for the p2p daemon
        :param kwargs: keyword CLI arguments for the p2p daemon
        :return: a wrapper for the p2p daemon
        """

        assert not (bootstrap_peers and use_ipfs), \
            'User-defined bootstrap_peers and use_ipfs=True are incompatible, please choose one option'

        self = cls()
        with path(cli, P2PD_FILENAME) as p:
            p2pd_path = p

        need_bootstrap = bool(bootstrap_peers) or use_ipfs
        bootstrap_peers = cls._make_bootstrap_peers(bootstrap_peers)
        dht = cls.DHT_MODE_MAPPING.get(dht_mode, {'dht': 0})
        force_reachability = cls.FORCE_REACHABILITY_MAPPING.get(force_reachability, {})
        proc_args = self._make_process_args(
            str(p2pd_path), *args,
            quic=quic, tls=tls, connManager=conn_manager,
            natPortMap=nat_port_map, autonat=auto_nat,
            relay=use_relay, relayHop=use_relay_hop, relayDiscovery=use_relay_discovery,
            autoRelay=use_auto_relay, relayHopLimit=relay_hop_limit,
            b=need_bootstrap, **{**bootstrap_peers, **dht, **force_reachability, **kwargs})
        self._assign_daemon_ports(external_port, daemon_listen_port)

        for try_count in range(NUM_RETRIES):
            try:
                self._initialize(proc_args)
                await self._wait_for_client(RETRY_DELAY * (2 ** try_count))
                break
            except Exception as e:
                logger.debug(f"Failed to initialize p2p daemon: {e}")
                self._terminate()
                if try_count == NUM_RETRIES - 1:
                    raise
                self._assign_daemon_ports()

        return self

    @classmethod
    async def replicate(cls, daemon_listen_port: int, external_port: int) -> 'P2P':
        """
        Connect to existing p2p daemon
        :param daemon_listen_port: port for connection daemon and client binding
        :param external_port: port for external connections from other p2p instances
        :return: new wrapper for existing p2p daemon
        """

        self = cls()
        # There is no child under control
        # Use external already running p2pd
        self._child = None
        self._alive = True
        self._assign_daemon_ports(external_port, daemon_listen_port)
        self._client_listen_port = find_open_port()
        self._client = p2pclient.Client(
            Multiaddr(f'/ip4/127.0.0.1/tcp/{self._daemon_listen_port}'),
            Multiaddr(f'/ip4/127.0.0.1/tcp/{self._client_listen_port}'))
        await self._wait_for_client()
        return self

    async def identify_maddrs(self) -> List[Multiaddr]:
        _, maddrs = await self._client.identify()
        if not maddrs:
            raise ValueError(f"No multiaddrs found for peer {self.id}")

        p2p_maddr = Multiaddr(f'/p2p/{self.id.to_base58()}')
        return [addr.encapsulate(p2p_maddr) for addr in maddrs]

    async def wait_for_at_least_n_peers(self, n_peers: int, attempts: int = 3, delay: float = 1) -> None:
        for _ in range(attempts):
            peers = await self._client.list_peers()
            if len(peers) >= n_peers:
                return
            await asyncio.sleep(delay)

        raise RuntimeError('Not enough peers')

    def _initialize(self, proc_args: List[str]) -> None:
        proc_args = deepcopy(proc_args)
        proc_args.extend(self._make_process_args(
            hostAddrs=f'/ip4/0.0.0.0/tcp/{self._external_port},/ip4/0.0.0.0/udp/{self._external_port}/quic',
            listen=f'/ip4/127.0.0.1/tcp/{self._daemon_listen_port}'
        ))
        self._child = Popen(args=proc_args, encoding="utf8")
        self._alive = True
        self._client_listen_port = find_open_port()
        self._client = p2pclient.Client(
            Multiaddr(f'/ip4/127.0.0.1/tcp/{self._daemon_listen_port}'),
            Multiaddr(f'/ip4/127.0.0.1/tcp/{self._client_listen_port}'))

    async def _wait_for_client(self, delay: float = 0) -> None:
        await asyncio.sleep(delay)
        self.id, _ = await self._client.identify()

    def _assign_daemon_ports(self, external_port: int = None, daemon_listen_port: int = None) -> None:
        if external_port is None:
            external_port = find_open_port()
        if daemon_listen_port is None:
            daemon_listen_port = find_open_port()
            while daemon_listen_port == external_port:
                daemon_listen_port = find_open_port()

        self._external_port, self._daemon_listen_port = external_port, daemon_listen_port

    @property
    def external_port(self) -> int:
        return self._external_port

    @staticmethod
    async def send_raw_data(data: bytes, writer: asyncio.StreamWriter) -> None:
        request = len(data).to_bytes(P2P.HEADER_LEN, P2P.BYTEORDER) + data
        writer.write(request)

    @staticmethod
    async def send_msgpack(data: Any, writer: asyncio.StreamWriter) -> None:
        raw_data = MSGPackSerializer.dumps(data)
        await P2P.send_raw_data(raw_data, writer)

    @staticmethod
    async def send_protobuf(protobuf, out_proto_type: type, writer: asyncio.StreamWriter) -> None:
        if type(protobuf) != out_proto_type:
            raise TypeError('Unary handler returned protobuf of wrong type.')
        if out_proto_type == p2pd_pb2.RPCError:
            await P2P.send_raw_data(P2P.ERROR_MESSAGE, writer)
        else:
            await P2P.send_raw_data(P2P.RESULT_MESSAGE, writer)

        await P2P.send_raw_data(protobuf.SerializeToString(), writer)

    @staticmethod
    async def receive_raw_data(reader: asyncio.StreamReader) -> bytes:
        header = await reader.readexactly(P2P.HEADER_LEN)
        content_length = int.from_bytes(header, P2P.BYTEORDER)
        data = await reader.readexactly(content_length)
        return data

    @staticmethod
    async def receive_msgpack(reader: asyncio.StreamReader) -> Any:
        return MSGPackSerializer.loads(await P2P.receive_raw_data(reader))

    @staticmethod
    async def receive_protobuf(in_proto_type: type, reader: asyncio.StreamReader) -> \
            Tuple[Any, Optional[p2pd_pb2.RPCError]]:
        msg_type = await P2P.receive_raw_data(reader)
        if msg_type == P2P.RESULT_MESSAGE:
            protobuf = in_proto_type()
            protobuf.ParseFromString(await P2P.receive_raw_data(reader))
            return protobuf, None
        elif msg_type == P2P.ERROR_MESSAGE:
            protobuf = p2pd_pb2.RPCError()
            protobuf.ParseFromString(await P2P.receive_raw_data(reader))
            return None, protobuf
        else:
            raise TypeError('Invalid Protobuf message type')

    @staticmethod
    def _handle_stream(handle: Callable[[bytes], bytes]):
        async def do_handle_stream(
                stream_info: StreamInfo, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            try:
                request = await P2P.receive_raw_data(reader)
            except asyncio.IncompleteReadError:
                logger.debug("Incomplete read while receiving request from peer")
                writer.close()
                return
            try:
                result = handle(request)
                await P2P.send_raw_data(result, writer)
            finally:
                writer.close()

        return do_handle_stream

    def _handle_unary_stream(self, handle: Callable[[Any, P2PContext], Any], handle_name: str,
                             in_proto_type: type, out_proto_type: type):
        async def watchdog(reader: asyncio.StreamReader) -> None:
            await reader.read(n=1)
            raise P2PInterruptedError()

        async def do_handle_unary_stream(stream_info: StreamInfo,
                                         reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                try:
                    request, err = await P2P.receive_protobuf(in_proto_type, reader)
                except asyncio.IncompleteReadError:
                    logger.debug('Incomplete read while receiving request from peer')
                    return
                except google.protobuf.message.DecodeError as error:
                    logger.debug(f'Failed to decode request protobuf: {error}')
                    return
                if err is not None:
                    logger.debug(f'Got an error instead of a request: {err}')

                context = P2PContext(handle_name=handle_name, local_id=self.id,
                                     remote_id=stream_info.peer_id, remote_maddr=stream_info.addr)
                done, pending = await asyncio.wait([watchdog(reader), handle(request, context)],
                                                   return_when=asyncio.FIRST_COMPLETED)
                try:
                    result = done.pop().result()
                    await P2P.send_protobuf(result, out_proto_type, writer)
                except P2PInterruptedError:
                    pass
                except Exception as exc:
                    error = p2pd_pb2.RPCError(message=str(exc))
                    await P2P.send_protobuf(error, p2pd_pb2.RPCError, writer)
                finally:
                    pending_task = pending.pop()
                    pending_task.cancel()
                    try:
                        await pending_task
                    except asyncio.CancelledError:
                        pass
            finally:
                writer.close()

        return do_handle_unary_stream

    def start_listening(self) -> None:
        async def listen() -> None:
            async with self._client.listen():
                await self._server_stopped.wait()

        self._listen_task = asyncio.create_task(listen())

    async def stop_listening(self) -> None:
        if self._listen_task is not None:
            self._server_stopped.set()
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                self._listen_task = None
                self._server_stopped.clear()

    async def add_stream_handler(self, name: str, handle: Callable[[bytes], bytes]) -> None:
        if self._listen_task is None:
            self.start_listening()
        await self._client.stream_handler(name, self._handle_stream(handle))

    async def add_unary_handler(self, name: str, handle: Callable[[Any, P2PContext], Any],
                                in_proto_type: type, out_proto_type: type) -> None:
        if self._listen_task is None:
            self.start_listening()
        await self._client.stream_handler(
            name, self._handle_unary_stream(handle, name, in_proto_type, out_proto_type))

    async def call_peer_handler(self, peer_id: PeerID, handler_name: str, input_data: bytes) -> bytes:
        stream_info, reader, writer = await self._client.stream_open(peer_id, (handler_name,))
        try:
            await P2P.send_raw_data(input_data, writer)
            return await P2P.receive_raw_data(reader)
        finally:
            writer.close()

    async def call_unary_handler(self, peer_id: PeerID, handler_name: str,
                                 request_protobuf: Any, response_proto_type: type) -> Any:
        stream_info, reader, writer = await self._client.stream_open(peer_id, (handler_name,))
        try:
            await P2P.send_protobuf(request_protobuf, type(request_protobuf), writer)
            result, err = await P2P.receive_protobuf(response_proto_type, reader)
            if err is not None:
                raise P2PHandlerError(f'Failed to call unary handler {handler_name} at {peer_id}: {err.message}')

            return result
        finally:
            writer.close()

    def __del__(self):
        self._terminate()

    @property
    def is_alive(self) -> bool:
        return self._alive

    async def shutdown(self) -> None:
        await asyncio.get_event_loop().run_in_executor(None, self._terminate)

    def _terminate(self) -> None:
        self._alive = False
        if self._child is not None and self._child.poll() is None:
            self._child.terminate()
            self._child.wait()

    @staticmethod
    def _make_process_args(*args, **kwargs) -> List[str]:
        proc_args = []
        proc_args.extend(
            str(entry) for entry in args
        )
        proc_args.extend(
            f'-{key}={P2P._convert_process_arg_type(value)}' if value is not None else f'-{key}'
            for key, value in kwargs.items()
        )
        return proc_args

    @staticmethod
    def _convert_process_arg_type(val: Any) -> Any:
        if isinstance(val, bool):
            return int(val)
        return val

    @staticmethod
    def _make_bootstrap_peers(maddrs: Optional[List[Multiaddr]]) -> Dict[str, str]:
        if maddrs is None:
            return {}

        return {'bootstrapPeers': ','.join(str(addr) for addr in maddrs)}


class P2PInterruptedError(Exception):
    pass


class P2PHandlerError(Exception):
    pass
