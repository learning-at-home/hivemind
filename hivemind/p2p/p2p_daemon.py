import asyncio
import copy
import dataclasses
import subprocess
import typing as tp
from pathlib import Path

import google.protobuf
from multiaddr import Multiaddr

import hivemind.p2p.p2p_daemon_bindings.p2pclient as p2pclient
from hivemind.p2p.p2p_daemon_bindings.datastructures import ID, StreamInfo
from hivemind.proto import p2pd_pb2
from hivemind.utils import MSGPackSerializer
from hivemind.utils.logging import get_logger
from hivemind.utils.networking import find_open_port

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=False)
class P2PContext(object):
    ours_id: str
    ours_port: int
    handle_name: str
    peer_id: ID = None
    peer_addr: Multiaddr = None

from hivemind.utils.networking import find_open_port


class P2PContext(object):
    def __init__(self, ours_id, ours_port, handle_name):
        self.peer_id = None
        self.peer_addr = None
        self.ours_id = ours_id
        self.ours_port = ours_port
        self.handle_name = handle_name


class P2P(object):
    """
    Forks a child process and executes p2pd command with given arguments.
    Can be used for peer to peer communication and procedure calls.
    Sends SIGKILL to the child in destructor.
    """

    P2PD_RELATIVE_PATH = 'hivemind_cli/p2pd'
    NUM_RETRIES = 3
    RETRY_DELAY = 0.4
    HEADER_LEN = 8
    BYTEORDER = 'big'
    PB_HEADER_LEN = 1
    RESULT_MESSAGE = int(0).to_bytes(PB_HEADER_LEN, BYTEORDER)
    ERROR_MESSAGE = int(1).to_bytes(PB_HEADER_LEN, BYTEORDER)

    class InterruptedError(Exception):
        pass

    def __init__(self):
        self._child = None
        self._alive = False
        self._listen_task = None
        self._server_stopped = asyncio.Event()

    @classmethod
    async def create(cls, *args, quic=1, tls=1, conn_manager=1, dht_mode='dht_server', force_reachability=None,
                     nat_port_map=True, auto_nat=True, bootstrap=False, boostrap_peers=None, use_global_ipfs=False,
                     host_port: int = None, daemon_listen_port: int = None, **kwargs):
        if bootstrap and boostrap_peers is None and not use_global_ipfs:
            raise AttributeError('Trying to create with bootstrap node without bootstrap nodes list. '
                                 'It is very dangerous, because p2pd connects to global ipfs and it is very unstable. '
                                 'If you really want this, pass use_global_ipfs=True')
        if boostrap_peers is not None and use_global_ipfs:
            raise AttributeError('Non empty boostrap_nodes and use_global_ipfs=True are incompatible.'
                                 'Choose one option: your nodes list (preferable) or global ipfs (very unstable)')

        self = cls()
        p2pd_path = Path(__file__).resolve().parents[1] / P2P.P2PD_RELATIVE_PATH
        bpeers = cls._make_bootstrap_peers(boostrap_peers)
        dht = cls._make_dht_mode(dht_mode)
        freachability = cls._make_force_reachability(force_reachability)
        proc_args = self._make_process_args(
            str(p2pd_path), *args,
            quic=quic, tls=tls, connManager=conn_manager,
            natPortMap=nat_port_map, autonat=auto_nat,
            b=bootstrap, **{**bpeers, **dht, **freachability, **kwargs})
        self._assign_daemon_ports(host_port, daemon_listen_port)
        for try_count in range(self.NUM_RETRIES):
            try:
                self._initialize(proc_args)
                await self._identify_client(P2P.RETRY_DELAY * (2 ** try_count))
            except Exception as e:
                logger.debug(f"Failed to initialize p2p daemon: {e}")
                self._kill_child()
                if try_count == P2P.NUM_RETRIES - 1:
                    raise
                self._assign_daemon_ports()
                continue
            break
        return self

    @classmethod
    async def replicate(cls, daemon_listen_port: int, host_port: int):
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
        await self._identify_client(0)
        return self

    async def wait_for_at_least_n_peers(self, n_peers, attempts=3):
        for _ in range(attempts):
            peers = await self._client.list_peers()
            if len(peers) >= n_peers:
                return
            await asyncio.sleep(1)

        raise RuntimeError('Not enough peers')

    def _initialize(self, proc_args: tp.List[str]) -> None:
        proc_args = copy.deepcopy(proc_args)
        proc_args.extend(self._make_process_args(
            hostAddrs=f'/ip4/0.0.0.0/tcp/{self._host_port},/ip4/0.0.0.0/udp/{self._host_port}/quic',
            listen=f'/ip4/127.0.0.1/tcp/{self._daemon_listen_port}'
        ))
        self._child = subprocess.Popen(
            args=proc_args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, encoding="utf8"
        )
        self._alive = True
        self._client_listen_port = find_open_port()
        self._client = p2pclient.Client(
            Multiaddr(f'/ip4/127.0.0.1/tcp/{self._daemon_listen_port}'),
            Multiaddr(f'/ip4/127.0.0.1/tcp/{self._client_listen_port}'))

    async def _identify_client(self, delay):
        await asyncio.sleep(delay)
        encoded = await self._client.identify()
        self.id = encoded[0].to_base58()

    def _assign_daemon_ports(self, host_port=None, daemon_listen_port=None):
        self._host_port, self._daemon_listen_port = host_port, daemon_listen_port
        if host_port is None:
            self._host_port = find_open_port()
        if daemon_listen_port is None:
            self._daemon_listen_port = find_open_port()
            while self._daemon_listen_port == self._host_port:
                self._daemon_listen_port = find_open_port()

    @staticmethod
    async def send_raw_data(byte_str, writer):
        request = len(byte_str).to_bytes(P2P.HEADER_LEN, P2P.BYTEORDER) + byte_str
        writer.write(request)

    @staticmethod
    async def send_message_pack(data, writer):
        raw_data = MSGPackSerializer.dumps(data)
        await P2P.send_raw_data(raw_data, writer)

    @staticmethod
    async def send_protobuf(protobuf, out_proto_type, writer):
        if type(protobuf) != out_proto_type:
            raise TypeError('Unary handler returned protobuf of wrong type.')
        await P2P.send_raw_data(protobuf.SerializeToString(), writer)

    @staticmethod
    async def send_protobuf_with_error(protobuf, out_proto_type, writer):
        if type(protobuf) != out_proto_type:
            raise TypeError('Unary handler returned protobuf of wrong type.')
        if out_proto_type == p2pd_pb2.P2PRPCError:
            await P2P.send_raw_data(P2P.ERROR_MESSAGE, writer)
        else:
            await P2P.send_raw_data(P2P.RESULT_MESSAGE, writer)

        await P2P.send_raw_data(protobuf.SerializeToString(), writer)

    @staticmethod
    async def send_error_protobuf(protobuf, out_proto_type, writer):
        await P2P.send_raw_data(P2P.RESULT_MESSAGE, writer)
        await P2P.send_raw_data(protobuf.SerializeToString(), writer)

    @staticmethod
    async def receive_raw_data(reader: asyncio.StreamReader, header_len=HEADER_LEN):
        header = await reader.readexactly(header_len)
        content_length = int.from_bytes(header, P2P.BYTEORDER)
        data = await reader.readexactly(content_length)
        return data

    @staticmethod
    async def receive_message_pack(reader):
        return MSGPackSerializer.loads(await P2P.receive_raw_data(reader))

    @staticmethod
    async def receive_protobuf(in_proto_type, reader):
        protobuf = in_proto_type()
        protobuf.ParseFromString(await P2P.receive_raw_data(reader))
        return protobuf

    @staticmethod
    async def receive_protobuf_with_error(in_proto_type, reader):
        msg_type = await P2P.receive_raw_data(reader)
        if msg_type == P2P.RESULT_MESSAGE:
            protobuf = in_proto_type()
            protobuf.ParseFromString(await P2P.receive_raw_data(reader))
            return protobuf, None
        elif msg_type == P2P.ERROR_MESSAGE:
            protobuf = p2pd_pb2.P2PRPCError()
            protobuf.ParseFromString(await P2P.receive_raw_data(reader))
            return None, protobuf
        else:
            raise TypeError('invalid protobuf message type')

    @staticmethod
    def _handle_stream(handle):
        async def do_handle_stream(stream_info, reader, writer):
            try:
                request = await P2P.receive_raw_data(reader) # receive raw data
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
            raise P2P.InterruptedError()

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
                    logger.warning(repr(error))
                    return

                context.peer_id, context.peer_addr = stream_info.peer_id, stream_info.addr
                done, pending = await asyncio.wait([watchdog(reader), handle(request, context)],
                                                   return_when=asyncio.FIRST_COMPLETED)
                try:
                    result = done.pop().result()
                    await P2P.send_protobuf_with_error(result, out_proto_type, writer)
                except P2P.InterruptedError:
                    pass
                except Exception as exc:
                    error = p2pd_pb2.P2PRPCError(message=str(exc))
                    await P2P.send_protobuf_with_error(error, p2pd_pb2.P2PRPCError, writer)
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
        await self._client.stream_handler(name, P2P._handle_stream(handle))

    async def add_unary_handler(self, name, handle, in_proto_type, out_proto_type):
        if self._listen_task is None:
            self.start_listening()
        context = P2PContext(ours_id=self.id, ours_port=self._host_port, handle_name=name)
        await self._client.stream_handler(
            name, P2P._handle_unary_stream(handle, context, in_proto_type, out_proto_type))

    async def call_peer_handler(self, peer_id, handler_name, input_data):
        libp2p_peer_id = ID.from_base58(peer_id)
        stream_info, reader, writer = await self._client.stream_open(libp2p_peer_id, (handler_name,))
        try:
            await P2P.send_raw_data(input_data, writer)
            return await P2P.receive_raw_data(reader)
        finally:
            writer.close()

    def __del__(self):
        self._kill_child()

    @property
    def is_alive(self):
        return self._alive

    async def shutdown(self, timeout=None):
        await asyncio.get_event_loop().run_in_executor(None, self._kill_child)

    def _kill_child(self):
        self._alive = False
        if self._child is not None and self._child.poll() is None:
            self._child.kill()
            self._child.wait()

    def _make_process_args(self, *args, **kwargs) -> tp.List[str]:
        proc_args = []
        proc_args.extend(
            str(entry) for entry in args
        )
        proc_args.extend(
            f'-{key}={value}' if value is not None else f'-{key}'
            for key, value in kwargs.items()
        )
        return proc_args

    @staticmethod
    def _make_bootstrap_peers(nodes):
        if nodes is None:
            return {}
        return {'bootstrapPeers': ','.join(nodes)}

    @staticmethod
    def _make_dht_mode(dht_mode):
        if dht_mode == 'dht':
            return {'dht': 1}
        if dht_mode == 'dht_server':
            return {'dhtServer': 1}
        if dht_mode == 'dht_client':
            return {'dhtClient': 1}
        return {'dht': 0}

    @staticmethod
    def _make_force_reachability(force_reachability):
        if force_reachability == 'public':
            return {'forceReachabilityPublic': 1}
        if force_reachability == 'private':
            return {'forceReachabilityPrivate': 1}
        return {}
