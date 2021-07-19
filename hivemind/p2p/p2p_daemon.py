import asyncio
import os
import secrets
from collections.abc import AsyncIterable as AsyncIterableABC
from contextlib import closing, suppress
from dataclasses import dataclass
from importlib.resources import path
from subprocess import Popen
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from multiaddr import Multiaddr

import hivemind.hivemind_cli as cli
import hivemind.p2p.p2p_daemon_bindings.p2pclient as p2pclient
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID, PeerInfo, StreamInfo
from hivemind.proto.p2pd_pb2 import RPCError
from hivemind.utils.asyncio import aiter
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


P2PD_FILENAME = "p2pd"


@dataclass(frozen=True)
class P2PContext(object):
    handle_name: str
    local_id: PeerID
    remote_id: PeerID = None


class P2P:
    """
    This class is responsible for establishing peer-to-peer connections through NAT and/or firewalls.
    It creates and manages a libp2p daemon (https://libp2p.io) in a background process,
    then terminates it when P2P is shut down. In order to communicate, a P2P instance should
    either use one or more initial_peers that will connect it to the rest of the swarm or
    use the public IPFS network (https://ipfs.io).

    For incoming connections, P2P instances add RPC handlers that may be accessed by other peers:
      - `P2P.add_protobuf_handler` accepts a protobuf message and returns another protobuf
      - `P2P.add_binary_stream_handler` transfers raw data using bi-directional streaming interface

    To access these handlers, a P2P instance can `P2P.call_protobuf_handler`/`P2P.call_binary_stream_handler`,
    using the recipient's unique `P2P.id` and the name of the corresponding handler.
    """

    HEADER_LEN = 8
    BYTEORDER = "big"
    MESSAGE_MARKER = b"\x00"
    ERROR_MARKER = b"\x01"
    END_OF_STREAM = RPCError()

    DHT_MODE_MAPPING = {
        "dht": {"dht": 1},
        "dht_server": {"dhtServer": 1},
        "dht_client": {"dhtClient": 1},
    }
    FORCE_REACHABILITY_MAPPING = {
        "public": {"forceReachabilityPublic": 1},
        "private": {"forceReachabilityPrivate": 1},
    }
    _UNIX_SOCKET_PREFIX = "/unix/tmp/hivemind-"

    def __init__(self):
        self.id = None
        self._child = None
        self._alive = False
        self._listen_task = None
        self._server_stopped = asyncio.Event()

    @classmethod
    async def create(
        cls,
        initial_peers: Optional[Sequence[Union[Multiaddr, str]]] = None,
        use_ipfs: bool = False,
        host_maddrs: Optional[Sequence[Union[Multiaddr, str]]] = ("/ip4/127.0.0.1/tcp/0",),
        announce_maddrs: Optional[Sequence[Union[Multiaddr, str]]] = None,
        quic: bool = True,
        tls: bool = True,
        conn_manager: bool = True,
        dht_mode: str = "dht_server",
        force_reachability: Optional[str] = None,
        nat_port_map: bool = True,
        auto_nat: bool = True,
        use_relay: bool = True,
        use_relay_hop: bool = False,
        use_relay_discovery: bool = False,
        use_auto_relay: bool = False,
        relay_hop_limit: int = 0,
        quiet: bool = True,
        ping_n_attempts: int = 5,
        ping_delay: float = 0.4,
    ) -> "P2P":
        """
        Start a new p2pd process and connect to it.
        :param initial_peers: List of bootstrap peers
        :param use_ipfs: Bootstrap to IPFS (incompatible with initial_peers)
        :param host_maddrs: Multiaddrs to listen for external connections from other p2p instances
        :param announce_maddrs: Visible multiaddrs that the peer will announce
          for external connections from other p2p instances
        :param quic: Enables the QUIC transport
        :param tls: Enables TLS1.3 channel security protocol
        :param conn_manager: Enables the Connection Manager
        :param dht_mode: DHT mode (dht_client/dht_server/dht)
        :param force_reachability: Force reachability mode (public/private)
        :param nat_port_map: Enables NAT port mapping
        :param auto_nat: Enables the AutoNAT service
        :param use_relay: enables circuit relay
        :param use_relay_hop: enables hop for relay
        :param use_relay_discovery: enables passive discovery for relay
        :param use_auto_relay: enables autorelay
        :param relay_hop_limit: sets the hop limit for hop relays
        :param quiet: make the daemon process quiet
        :param ping_n_attempts: try to ping the daemon with this number of attempts after starting it
        :param ping_delay: wait for ``ping_delay * (2 ** (k - 1))`` seconds before the k-th attempt to ping the daemon
          (in particular, wait for ``ping_delay`` seconds before the first attempt)
        :return: a wrapper for the p2p daemon
        """

        assert not (
            initial_peers and use_ipfs
        ), "User-defined initial_peers and use_ipfs=True are incompatible, please choose one option"

        self = cls()
        with path(cli, P2PD_FILENAME) as p:
            p2pd_path = p

        socket_uid = secrets.token_urlsafe(8)
        self._daemon_listen_maddr = Multiaddr(cls._UNIX_SOCKET_PREFIX + f"p2pd-{socket_uid}.sock")
        self._client_listen_maddr = Multiaddr(cls._UNIX_SOCKET_PREFIX + f"p2pclient-{socket_uid}.sock")

        need_bootstrap = bool(initial_peers) or use_ipfs
        process_kwargs = cls.DHT_MODE_MAPPING.get(dht_mode, {"dht": 0})
        process_kwargs.update(cls.FORCE_REACHABILITY_MAPPING.get(force_reachability, {}))
        for param, value in [
            ("bootstrapPeers", initial_peers),
            ("hostAddrs", host_maddrs),
            ("announceAddrs", announce_maddrs),
        ]:
            if value:
                process_kwargs[param] = self._maddrs_to_str(value)

        proc_args = self._make_process_args(
            str(p2pd_path),
            listen=self._daemon_listen_maddr,
            quic=quic,
            tls=tls,
            connManager=conn_manager,
            natPortMap=nat_port_map,
            autonat=auto_nat,
            relay=use_relay,
            relayHop=use_relay_hop,
            relayDiscovery=use_relay_discovery,
            autoRelay=use_auto_relay,
            relayHopLimit=relay_hop_limit,
            b=need_bootstrap,
            q=quiet,
            **process_kwargs,
        )

        self._child = Popen(args=proc_args, encoding="utf8")
        self._alive = True
        self._client = p2pclient.Client(self._daemon_listen_maddr, self._client_listen_maddr)

        await self._ping_daemon_with_retries(ping_n_attempts, ping_delay)

        return self

    async def add_unary_handler(self, handle_name: str, handler: p2pclient.TUnaryHandler):
        return await self._client.add_unary_handler(handle_name, handler)

    async def unary_call(self, peer_id: PeerID, handle_name: str, data: bytes) -> bytes:
        return await self._client.unary_call(peer_id, handle_name, data)

    async def _ping_daemon_with_retries(self, ping_n_attempts: int, ping_delay: float) -> None:
        for try_number in range(ping_n_attempts):
            await asyncio.sleep(ping_delay * (2 ** try_number))

            if self._child.poll() is not None:  # Process died
                break

            try:
                await self._ping_daemon()
                break
            except Exception as e:
                if try_number == ping_n_attempts - 1:
                    logger.exception("Failed to ping p2pd that has just started")
                    await self.shutdown()
                    raise

        if self._child.returncode is not None:
            raise RuntimeError(f"The p2p daemon has died with return code {self._child.returncode}")

    @classmethod
    async def replicate(cls, daemon_listen_maddr: Multiaddr) -> "P2P":
        """
        Connect to existing p2p daemon
        :param daemon_listen_maddr: multiaddr of the existing p2p daemon
        :return: new wrapper for the existing p2p daemon
        """

        self = cls()
        # There is no child under control
        # Use external already running p2pd
        self._child = None
        self._alive = True

        socket_uid = secrets.token_urlsafe(8)
        self._daemon_listen_maddr = daemon_listen_maddr
        self._client_listen_maddr = Multiaddr(cls._UNIX_SOCKET_PREFIX + f"p2pclient-{socket_uid}.sock")

        self._client = p2pclient.Client(self._daemon_listen_maddr, self._client_listen_maddr)

        await self._ping_daemon()
        return self

    async def _ping_daemon(self) -> None:
        self.id, self._visible_maddrs = await self._client.identify()
        logger.debug(f"Launched p2pd with id = {self.id}, host multiaddrs = {self._visible_maddrs}")

    async def get_visible_maddrs(self, latest: bool = False) -> List[Multiaddr]:
        """
        Get multiaddrs of the current peer that should be accessible by other peers.

        :param latest: ask the P2P daemon to refresh the visible multiaddrs
        """

        if latest:
            _, self._visible_maddrs = await self._client.identify()

        if not self._visible_maddrs:
            raise ValueError(f"No multiaddrs found for peer {self.id}")

        p2p_maddr = Multiaddr(f"/p2p/{self.id.to_base58()}")
        return [addr.encapsulate(p2p_maddr) for addr in self._visible_maddrs]

    async def list_peers(self) -> List[PeerInfo]:
        return list(await self._client.list_peers())

    async def wait_for_at_least_n_peers(self, n_peers: int, attempts: int = 3, delay: float = 1) -> None:
        for _ in range(attempts):
            peers = await self._client.list_peers()
            if len(peers) >= n_peers:
                return
            await asyncio.sleep(delay)

        raise RuntimeError("Not enough peers")

    @property
    def daemon_listen_maddr(self) -> Multiaddr:
        return self._daemon_listen_maddr

    @staticmethod
    async def send_raw_data(data: bytes, writer: asyncio.StreamWriter, *, chunk_size: int = 2 ** 16) -> None:
        writer.write(len(data).to_bytes(P2P.HEADER_LEN, P2P.BYTEORDER))
        data = memoryview(data)
        for offset in range(0, len(data), chunk_size):
            writer.write(data[offset : offset + chunk_size])
        await writer.drain()

    @staticmethod
    async def receive_raw_data(reader: asyncio.StreamReader) -> bytes:
        header = await reader.readexactly(P2P.HEADER_LEN)
        content_length = int.from_bytes(header, P2P.BYTEORDER)
        data = await reader.readexactly(content_length)
        return data

    TInputProtobuf = TypeVar("TInputProtobuf")
    TOutputProtobuf = TypeVar("TOutputProtobuf")

    @staticmethod
    async def send_protobuf(protobuf: Union[TOutputProtobuf, RPCError], writer: asyncio.StreamWriter) -> None:
        if isinstance(protobuf, RPCError):
            writer.write(P2P.ERROR_MARKER)
        else:
            writer.write(P2P.MESSAGE_MARKER)
        await P2P.send_raw_data(protobuf.SerializeToString(), writer)

    @staticmethod
    async def receive_protobuf(
        input_protobuf_type: type, reader: asyncio.StreamReader
    ) -> Tuple[Optional[TInputProtobuf], Optional[RPCError]]:
        msg_type = await reader.readexactly(1)
        if msg_type == P2P.MESSAGE_MARKER:
            protobuf = input_protobuf_type()
            protobuf.ParseFromString(await P2P.receive_raw_data(reader))
            return protobuf, None
        elif msg_type == P2P.ERROR_MARKER:
            protobuf = RPCError()
            protobuf.ParseFromString(await P2P.receive_raw_data(reader))
            return None, protobuf
        else:
            raise TypeError("Invalid Protobuf message type")

    TInputStream = AsyncIterator[TInputProtobuf]
    TOutputStream = AsyncIterator[TOutputProtobuf]

    async def _add_protobuf_stream_handler(
        self,
        name: str,
        handler: Callable[[TInputStream, P2PContext], TOutputStream],
        input_protobuf_type: type,
        max_prefetch: int = 5,
    ) -> None:
        """
        :param max_prefetch: Maximum number of items to prefetch from the request stream.
          ``max_prefetch <= 0`` means unlimited.

        :note:  Since the cancel messages are sent via the input stream,
          they will not be received while the prefetch buffer is full.
        """

        if self._listen_task is None:
            self._start_listening()

        async def _handle_stream(
            stream_info: StreamInfo, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            context = P2PContext(
                handle_name=name,
                local_id=self.id,
                remote_id=stream_info.peer_id,
            )
            requests = asyncio.Queue(max_prefetch)

            async def _read_stream() -> P2P.TInputStream:
                while True:
                    request = await requests.get()
                    if request is None:
                        break
                    yield request

            async def _process_stream() -> None:
                try:
                    async for response in handler(_read_stream(), context):
                        await P2P.send_protobuf(response, writer)
                except Exception as e:
                    logger.warning("Exception while processing stream and sending responses:", exc_info=True)
                    await P2P.send_protobuf(RPCError(message=str(e)), writer)

            with closing(writer):
                processing_task = asyncio.create_task(_process_stream())
                try:
                    while True:
                        receive_task = asyncio.create_task(P2P.receive_protobuf(input_protobuf_type, reader))
                        await asyncio.wait({processing_task, receive_task}, return_when=asyncio.FIRST_COMPLETED)

                        if processing_task.done():
                            receive_task.cancel()
                            return

                        if receive_task.done():
                            try:
                                request, _ = await receive_task
                            except asyncio.IncompleteReadError:  # Connection is closed (the client cancelled or died)
                                return
                            await requests.put(request)  # `request` is None for the end-of-stream message
                except Exception:
                    logger.warning("Exception while receiving requests:", exc_info=True)
                finally:
                    processing_task.cancel()

        await self._client.stream_handler(name, _handle_stream)

    async def _iterate_protobuf_stream_handler(
        self, peer_id: PeerID, name: str, requests: TInputStream, output_protobuf_type: type
    ) -> TOutputStream:
        _, reader, writer = await self._client.stream_open(peer_id, (name,))

        async def _write_to_stream() -> None:
            async for request in requests:
                await P2P.send_protobuf(request, writer)
            await P2P.send_protobuf(P2P.END_OF_STREAM, writer)

        with closing(writer):
            writing_task = asyncio.create_task(_write_to_stream())
            try:
                while True:
                    try:
                        response, err = await P2P.receive_protobuf(output_protobuf_type, reader)
                    except asyncio.IncompleteReadError:  # Connection is closed
                        break

                    if err is not None:
                        raise P2PHandlerError(f"Failed to call handler `{name}` at {peer_id}: {err.message}")
                    yield response

                await writing_task
            finally:
                writing_task.cancel()

    async def add_protobuf_handler(
        self,
        name: str,
        handler: Callable[
            [Union[TInputProtobuf, TInputStream], P2PContext], Union[Awaitable[TOutputProtobuf], TOutputStream]
        ],
        input_protobuf_type: type,
        *,
        stream_input: bool = False,
    ) -> None:
        """
        :param stream_input: If True, assume ``handler`` to take ``TInputStream``
                             (not just ``TInputProtobuf``) as input.
        """

        if not stream_input:
            await self._add_protobuf_unary_handler(name, handler, input_protobuf_type)
            return

        async def _stream_handler(requests: P2P.TInputStream, context: P2PContext) -> P2P.TOutputStream:
            input = requests
            output = handler(input, context)

            if isinstance(output, AsyncIterableABC):
                async for item in output:
                    yield item
            else:
                yield await output

        await self._add_protobuf_stream_handler(name, _stream_handler, input_protobuf_type)

    # only registers request-response handlers
    async def _add_protobuf_unary_handler(
        self,
        handle_name: str,
        handler: Callable[[TInputProtobuf, P2PContext], Awaitable[TOutputProtobuf]],
        input_protobuf_type: type,
    ) -> None:
        async def _unary_handler(request: bytes) -> bytes:
            input_serialized = input_protobuf_type().FromString(request)
            context = P2PContext(
                handle_name=handle_name,
                local_id=self.id,
                # TODO: add remote id
            )

            response = await handler(input_serialized, context)
            return response.SerializeToString()

        await self.add_unary_handler(handle_name, _unary_handler)

    async def call_protobuf_handler(
        self,
        peer_id: PeerID,
        name: str,
        input: Union[TInputProtobuf, TInputStream],
        output_protobuf_type: type,
    ) -> Awaitable[TOutputProtobuf]:

        if not isinstance(input, AsyncIterableABC):
            return self._call_unary_protobuf_handler(peer_id, name, input, output_protobuf_type)

        responses = self._iterate_protobuf_stream_handler(peer_id, name, input, output_protobuf_type)

        count = 0
        async for response in responses:
            count += 1
        if count != 1:
            raise ValueError(f"Got {count} responses from handler {name} instead of one")
        return response

    async def _call_unary_protobuf_handler(
        self,
        peer_id: PeerID,
        handle_name: str,
        input: TInputProtobuf,
        output_protobuf_type: type,
    ) -> Awaitable[TOutputProtobuf]:
        serialized_input = input.SerializeToString()
        response = await self.unary_call(peer_id, handle_name, serialized_input)
        return output_protobuf_type().FromString(response)

    def iterate_protobuf_handler(
        self,
        peer_id: PeerID,
        name: str,
        input: Union[TInputProtobuf, TInputStream],
        output_protobuf_type: type,
    ) -> TOutputStream:
        requests = input if isinstance(input, AsyncIterableABC) else aiter(input)
        return self._iterate_protobuf_stream_handler(peer_id, name, requests, output_protobuf_type)

    def _start_listening(self) -> None:
        async def listen() -> None:
            async with self._client.listen():
                await self._server_stopped.wait()

        self._listen_task = asyncio.create_task(listen())

    async def _stop_listening(self) -> None:
        if self._listen_task is not None:
            self._server_stopped.set()
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                self._listen_task = None
                self._server_stopped.clear()

    async def add_binary_stream_handler(self, name: str, handler: p2pclient.StreamHandler) -> None:
        if self._listen_task is None:
            self._start_listening()
        await self._client.stream_handler(name, handler)

    async def call_binary_stream_handler(
        self, peer_id: PeerID, handler_name: str
    ) -> Tuple[StreamInfo, asyncio.StreamReader, asyncio.StreamWriter]:
        return await self._client.stream_open(peer_id, (handler_name,))

    def __del__(self):
        self._terminate()

    @property
    def is_alive(self) -> bool:
        return self._alive

    async def shutdown(self) -> None:
        await self._stop_listening()
        await asyncio.get_event_loop().run_in_executor(None, self._terminate)

    def _terminate(self) -> None:
        self._alive = False
        if self._child is not None and self._child.poll() is None:
            self._child.terminate()
            self._child.wait()
            logger.debug(f"Terminated p2pd with id = {self.id}")

            with suppress(FileNotFoundError):
                os.remove(self._daemon_listen_maddr["unix"])
        with suppress(FileNotFoundError):
            os.remove(self._client_listen_maddr["unix"])

    @staticmethod
    def _make_process_args(*args, **kwargs) -> List[str]:
        proc_args = []
        proc_args.extend(str(entry) for entry in args)
        proc_args.extend(
            f"-{key}={P2P._convert_process_arg_type(value)}" if value is not None else f"-{key}"
            for key, value in kwargs.items()
        )
        return proc_args

    @staticmethod
    def _convert_process_arg_type(val: Any) -> Any:
        if isinstance(val, bool):
            return int(val)
        return val

    @staticmethod
    def _maddrs_to_str(maddrs: List[Multiaddr]) -> str:
        return ",".join(str(addr) for addr in maddrs)


class P2PInterruptedError(Exception):
    pass


class P2PHandlerError(Exception):
    pass
