import asyncio
import copy
from pathlib import Path
import pickle
import subprocess
import typing as tp

import google.protobuf
from multiaddr import Multiaddr
import hivemind.p2p.p2p_daemon_bindings.p2pclient as p2pclient
from hivemind.p2p.p2p_daemon_bindings.datastructures import ID, StreamInfo

from hivemind.utils.networking import find_open_port
from hivemind.utils.logging import get_logger


logger = get_logger(__name__)


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

    class IncompleteRead(Exception):
        pass

    class InterruptedError(Exception):
        pass

    def __init__(self):
        self._child = None
        self._listen_task = None
        self._server_stopped = asyncio.Event()

    @classmethod
    async def create(cls, *args, quic=1, tls=1, conn_manager=1, dht_client=1,
                     nat_port_map=True, auto_nat=True, bootstrap=True,
                     host_port: int = None, daemon_listen_port: int = None, **kwargs):
        self = cls()
        p2pd_path = Path(__file__).resolve().parents[1] / P2P.P2PD_RELATIVE_PATH
        proc_args = self._make_process_args(
            str(p2pd_path), *args,
            quic=quic, tls=tls, connManager=conn_manager,
            dhtClient=dht_client, natPortMap=nat_port_map,
            autonat=auto_nat, b=bootstrap, **kwargs)
        self._assign_daemon_ports(host_port, daemon_listen_port)
        for try_count in range(self.NUM_RETRIES):
            try:
                self._initialize(proc_args)
                await self._identify_client(P2P.RETRY_DELAY * (2 ** try_count))
            except Exception as e:
                logger.debug(f"Failed to initialize p2p daemon: {e}", RuntimeWarning)
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
        self._assign_daemon_ports(host_port, daemon_listen_port)
        self._client_listen_port = find_open_port()
        self._client = p2pclient.Client(
            Multiaddr(f'/ip4/127.0.0.1/tcp/{self._daemon_listen_port}'),
            Multiaddr(f'/ip4/127.0.0.1/tcp/{self._client_listen_port}'))
        await self._identify_client(0)
        return self

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
    async def send_data(data, writer):
        await P2P.send_raw_data(pickle.dumps(data), writer)

    @staticmethod
    async def send_protobuf(protobuf, out_proto_type, writer):
        if type(protobuf) != out_proto_type:
            error = TypeError('Unary handler returned protobuf of wrong type.')
            await P2P.send_raw_data(pickle.dumps(error), writer)
            raise error
        await P2P.send_raw_data(protobuf.SerializeToString(), writer)

    @staticmethod
    async def receive_exactly(reader, n_bytes, max_bytes=1 << 16):
        buffer = bytearray()
        while len(buffer) < n_bytes:
            data = await reader.read(min(max_bytes, n_bytes - len(buffer)))
            if len(data) == 0:
                raise P2P.IncompleteRead()
            buffer.extend(data)
        return bytes(buffer)

    @staticmethod
    async def receive_raw_data(reader):
        header = await P2P.receive_exactly(reader, P2P.HEADER_LEN)
        content_length = int.from_bytes(header, P2P.BYTEORDER)
        data = await P2P.receive_exactly(reader, content_length)
        return data

    @staticmethod
    async def receive_data(reader):
        return pickle.loads(await P2P.receive_raw_data(reader))

    @staticmethod
    async def receive_protobuf(in_proto_type, reader):
        protobuf = in_proto_type()
        protobuf.ParseFromString(await P2P.receive_raw_data(reader))
        return protobuf

    @staticmethod
    def _handle_stream(handle):
        async def do_handle_stream(stream_info, reader, writer):
            try:
                request = await P2P.receive_data(reader)
            except P2P.IncompleteRead:
                logger.debug("Incomplete read while receiving request from peer")
                writer.close()
                return
            try:
                result = handle(request)
                await P2P.send_data(result, writer)
            except Exception as exc:
                await P2P.send_data(exc, writer)
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
                except P2P.IncompleteRead:
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
                    await P2P.send_protobuf(result, out_proto_type, writer)
                except P2P.InterruptedError:
                    pass
                except Exception as exc:
                    await P2P.send_data(exc, writer)
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
            await P2P.send_data(input_data, writer)
            return await P2P.receive_data(reader)
        finally:
            writer.close()

    def __del__(self):
        self._kill_child()

    @property
    def is_alive(self):
        return self._child.is_alive

    async def shutdown(self, timeout=None):
        await asyncio.get_event_loop().run_in_executor(None, self._kill_child)

    def _kill_child(self):
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
