import asyncio
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import path
from subprocess import Popen
from typing import List, Optional

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


class P2PInterruptedError(Exception):
    pass


@dataclass(frozen=False)
class P2PContext(object):
    id: str
    port: int
    handle_name: str
    peer_id: PeerID = None
    peer_addr: Multiaddr = None


class P2P:
    """
    Forks a child process and executes p2pd command with given arguments.
    Can be used for peer to peer communication and procedure calls.
    Sends SIGKILL to the child in destructor.
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
                     nat_port_map: bool = True, auto_nat: bool = True, bootstrap: bool = False,
                     bootstrap_peers: Optional[List[str]] = None, use_global_ipfs: bool = False, host_port: int = None,
                     daemon_listen_port: int = None, use_relay: bool = True, use_relay_hop: bool = False,
                     use_relay_discovery: bool = False, use_auto_relay: bool = False, relay_hop_limit: int = 0, **kwargs):
        """
        Start a new p2pd process and connect to it.
        :param args:
        :param quic: Enables the QUIC transport
        :param tls: Enables TLS1.3 channel security protocol
        :param conn_manager: Enables the Connection Manager
        :param dht_mode: DHT mode (dht_client/dht_server/dht)
        :param force_reachability: Force reachability mode (public/private)
        :param nat_port_map: Enables NAT port mapping
        :param auto_nat: Enables the AutoNAT service
        :param bootstrap: Connects to bootstrap peers and bootstraps the dht if enabled
        :param bootstrap_peers: List of bootstrap peers; defaults to the IPFS DHT peers
        :param use_global_ipfs: Bootstrap to global ipfs (works only if bootstrap=True and bootstrap_peers=None)
        :param host_port: port for p2p network
        :param daemon_listen_port: port for connection daemon and client binding
        :param use_relay: enables circuit relay
        :param use_relay_hop: enables hop for relay
        :param use_relay_discovery: enables passive discovery for relay
        :param use_auto_relay: enables autorelay
        :param relay_hop_limit: sets the hop limit for hop relays
        :param kwargs:
        :return: new wrapper for p2p daemon
        """

        assert not (bootstrap and bootstrap_peers is None and not use_global_ipfs), \
            'Trying to create with bootstrap node without bootstrap nodes list. ' \
            'It is very dangerous, because p2pd connects to global ipfs and it is very unstable. ' \
            'If you really want this, pass use_global_ipfs=True'
        assert not (bootstrap_peers is not None and use_global_ipfs), \
            'Non empty bootstrap_nodes and use_global_ipfs=True are incompatible.' \
            'Choose one option: your nodes list (preferable) or global ipfs (very unstable)'

        self = cls()
        with path(cli, P2PD_FILENAME) as p:
            p2pd_path = p
        bootstrap_peers = cls._make_bootstrap_peers(bootstrap_peers)
        dht = cls.DHT_MODE_MAPPING.get(dht_mode, {'dht': 0})
        force_reachability = cls.FORCE_REACHABILITY_MAPPING.get(force_reachability, {})
        proc_args = self._make_process_args(
            str(p2pd_path), *args,
            quic=quic, tls=tls, connManager=conn_manager,
            natPortMap=nat_port_map, autonat=auto_nat,
            relay=use_relay, relayHop=use_relay_hop, relayDiscovery=use_relay_discovery,
            autoRelay=use_auto_relay, relayHopLimit=relay_hop_limit,
            b=bootstrap, **{**bootstrap_peers, **dht, **force_reachability, **kwargs})
        self._assign_daemon_ports(host_port, daemon_listen_port)

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
    async def replicate(cls, daemon_listen_port: int, host_port: int):
        """
        Connect to existing p2p daemon
        :param daemon_listen_port: port for connection daemon and client binding
        :param host_port: port for p2p network
        :return: new wrapper for existing p2p daemon
        """

        self = cls()
        # There is no child under control
        # Use external already running p2pd
        self._child = None
        self._alive = True
        self._assign_daemon_ports(host_port, daemon_listen_port)
        self._client_listen_port = find_open_port()
        self._client = p2pclient.Client(
            Multiaddr(f'/ip4/127.0.0.1/tcp/{self._daemon_listen_port}'),
            Multiaddr(f'/ip4/127.0.0.1/tcp/{self._client_listen_port}'))
        await self._wait_for_client()
        return self

    async def wait_for_at_least_n_peers(self, n_peers, attempts=3, delay=1):
        for _ in range(attempts):
            peers = await self._client.list_peers()
            if len(peers) >= n_peers:
                return
            await asyncio.sleep(delay)

        raise RuntimeError('Not enough peers')

    def _initialize(self, proc_args: List[str]) -> None:
        proc_args = deepcopy(proc_args)
        proc_args.extend(self._make_process_args(
            hostAddrs=f'/ip4/0.0.0.0/tcp/{self._host_port},/ip4/0.0.0.0/udp/{self._host_port}/quic',
            listen=f'/ip4/127.0.0.1/tcp/{self._daemon_listen_port}'
        ))
        self._child = Popen(args=proc_args, encoding="utf8")
        self._alive = True
        self._client_listen_port = find_open_port()
        self._client = p2pclient.Client(
            Multiaddr(f'/ip4/127.0.0.1/tcp/{self._daemon_listen_port}'),
            Multiaddr(f'/ip4/127.0.0.1/tcp/{self._client_listen_port}'))

    async def _wait_for_client(self, delay=0):
        await asyncio.sleep(delay)
        encoded = await self._client.identify()
        self.id = encoded[0].to_base58()

    def _assign_daemon_ports(self, host_port=None, daemon_listen_port=None):
        if host_port is None:
            host_port = find_open_port()
        if daemon_listen_port is None:
            daemon_listen_port = find_open_port()
            while daemon_listen_port == host_port:
                daemon_listen_port = find_open_port()

        self._host_port, self._daemon_listen_port = host_port, daemon_listen_port

    @staticmethod
    async def send_raw_data(byte_str, writer):
        request = len(byte_str).to_bytes(P2P.HEADER_LEN, P2P.BYTEORDER) + byte_str
        writer.write(request)

    @staticmethod
    async def send_msgpack(data, writer):
        raw_data = MSGPackSerializer.dumps(data)
        await P2P.send_raw_data(raw_data, writer)

    @staticmethod
    async def send_protobuf(protobuf, out_proto_type, writer):
        if type(protobuf) != out_proto_type:
            raise TypeError('Unary handler returned protobuf of wrong type.')
        if out_proto_type == p2pd_pb2.RPCError:
            await P2P.send_raw_data(P2P.ERROR_MESSAGE, writer)
        else:
            await P2P.send_raw_data(P2P.RESULT_MESSAGE, writer)

        await P2P.send_raw_data(protobuf.SerializeToString(), writer)

    @staticmethod
    async def receive_raw_data(reader: asyncio.StreamReader, header_len=HEADER_LEN):
        header = await reader.readexactly(header_len)
        content_length = int.from_bytes(header, P2P.BYTEORDER)
        data = await reader.readexactly(content_length)
        return data

    @staticmethod
    async def receive_msgpack(reader):
        return MSGPackSerializer.loads(await P2P.receive_raw_data(reader))

    @staticmethod
    async def receive_protobuf(in_proto_type, reader):
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
    def _handle_stream(handle):
        async def do_handle_stream(stream_info, reader, writer):
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

    @staticmethod
    def _handle_unary_stream(handle, context, in_proto_type, out_proto_type):
        async def watchdog(reader: asyncio.StreamReader):
            await reader.read(n=1)
            raise P2PInterruptedError()

        async def do_handle_unary_stream(
                stream_info: StreamInfo,
                reader: asyncio.StreamReader,
                writer: asyncio.StreamWriter) -> None:
            try:
                try:
                    request = await P2P.receive_protobuf(in_proto_type, reader)
                except asyncio.IncompleteReadError:
                    logger.debug("Incomplete read while receiving request from peer")
                    return
                except google.protobuf.message.DecodeError as error:
                    logger.exception(error)
                    return

                context.peer_id, context.peer_addr = stream_info.peer_id, stream_info.addr
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

    def start_listening(self):
        async def listen():
            async with self._client.listen():
                await self._server_stopped.wait()

        self._listen_task = asyncio.create_task(listen())

    async def stop_listening(self):
        if self._listen_task is not None:
            self._server_stopped.set()
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                self._listen_task = None
                self._server_stopped.clear()

    async def add_stream_handler(self, name, handle):
        if self._listen_task is None:
            self.start_listening()
        await self._client.stream_handler(name, self._handle_stream(handle))

    async def add_unary_handler(self, name, handle, in_proto_type, out_proto_type):
        if self._listen_task is None:
            self.start_listening()
        context = P2PContext(id=self.id, port=self._host_port, handle_name=name)
        await self._client.stream_handler(
            name, P2P._handle_unary_stream(handle, context, in_proto_type, out_proto_type))

    async def call_peer_handler(self, peer_id, handler_name, input_data):
        libp2p_peer_id = PeerID.from_base58(peer_id)
        stream_info, reader, writer = await self._client.stream_open(libp2p_peer_id, (handler_name,))
        try:
            await P2P.send_raw_data(input_data, writer)
            return await P2P.receive_raw_data(reader)
        finally:
            writer.close()

    def __del__(self):
        self._terminate()

    @property
    def is_alive(self):
        return self._alive

    async def shutdown(self):
        await asyncio.get_event_loop().run_in_executor(None, self._terminate)

    def _terminate(self):
        self._alive = False
        if self._child is not None and self._child.poll() is None:
            self._child.kill()
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
    def _convert_process_arg_type(val):
        if isinstance(val, bool):
            return 1 if val else 0
        return val

    @staticmethod
    def _make_bootstrap_peers(nodes):
        if nodes is None:
            return {}
        return {'bootstrapPeers': ','.join(nodes)}
