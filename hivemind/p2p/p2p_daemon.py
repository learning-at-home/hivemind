import asyncio
import contextlib
import copy
from pathlib import Path
import pickle
import socket
import subprocess
import typing as tp
import warnings

from multiaddr import Multiaddr
import p2pclient
from libp2p.peer.id import ID


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

    def __init__(self):
        self._child = None
        self._listen_task = None
        self._server_stopped = asyncio.Event()
        self._buffer = bytearray()

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
            except Exception as exc:
                warnings.warn("Failed to initialize p2p daemon: " + str(exc), RuntimeWarning)
                self._kill_child()
                if try_count == P2P.NUM_RETRIES - 1:
                    raise
                self._assign_daemon_ports()
                continue
            break
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
    async def send_data(data, stream):
        byte_str = pickle.dumps(data)
        request = len(byte_str).to_bytes(P2P.HEADER_LEN, P2P.BYTEORDER) + byte_str
        await stream.send_all(request)

    class IncompleteRead(Exception):
        pass

    async def _receive_exactly(self, stream, n_bytes, max_bytes=1 << 16):
        while len(self._buffer) < n_bytes:
            data = await stream.receive_some(max_bytes)
            if len(data) == 0:
                raise P2P.IncompleteRead()
            self._buffer.extend(data)

        result = self._buffer[:n_bytes]
        self._buffer = self._buffer[n_bytes:]
        return bytes(result)

    async def receive_data(self, stream, max_bytes=(1 < 16)):
        header = await self._receive_exactly(stream, P2P.HEADER_LEN)
        content_length = int.from_bytes(header, P2P.BYTEORDER)
        data = await self._receive_exactly(stream, content_length)
        return pickle.loads(data)

    def _handle_stream(self, handle):
        async def do_handle_stream(stream_info, stream):
            try:
                request = await self.receive_data(stream)
            except P2P.IncompleteRead:
                warnings.warn("Incomplete read while receiving request from peer", RuntimeWarning)
                return
            finally:
                stream.close()
            try:
                result = handle(request)
                await self.send_data(result, stream)
            except Exception as exc:
                await self.send_data(exc, stream)
            finally:
                await stream.close()

        return do_handle_stream

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

    async def call_peer_handler(self, peer_id, handler_name, input_data):
        libp2p_peer_id = ID.from_base58(peer_id)
        stream_info, stream = await self._client.stream_open(libp2p_peer_id, (handler_name,))
        try:
            await self.send_data(input_data, stream)
            return await self.receive_data(stream)
        finally:
            await stream.close()

    def __del__(self):
        self._kill_child()

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


def find_open_port(params=(socket.AF_INET, socket.SOCK_STREAM),
                   opt=(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)):
    """ Finds a tcp port that can be occupied with a socket with *params and use *opt options """
    try:
        with contextlib.closing(socket.socket(*params)) as sock:
            sock.bind(('', 0))
            sock.setsockopt(*opt)
            return sock.getsockname()[1]
    except Exception:
        raise
