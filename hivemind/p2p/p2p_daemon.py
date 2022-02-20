import asyncio
import json
import logging
import os
import secrets
from collections.abc import AsyncIterable as AsyncIterableABC
from contextlib import closing, suppress
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from google.protobuf.message import Message
from multiaddr import Multiaddr

import hivemind.hivemind_cli as cli
import hivemind.p2p.p2p_daemon_bindings.p2pclient as p2pclient
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, P2PDaemonError, P2PHandlerError
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID, PeerInfo, StreamInfo
from hivemind.proto.p2pd_pb2 import RPCError
from hivemind.utils.asyncio import as_aiter, asingle
from hivemind.utils.logging import get_logger, golog_level_to_python, loglevel, python_level_to_golog

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
    using the recipient's unique `P2P.peer_id` and the name of the corresponding handler.
    """

    HEADER_LEN = 8
    BYTEORDER = "big"
    MESSAGE_MARKER = b"\x00"
    ERROR_MARKER = b"\x01"
    END_OF_STREAM = RPCError()

    DHT_MODE_MAPPING = {
        "auto": {"dht": 1},
        "server": {"dhtServer": 1},
        "client": {"dhtClient": 1},
    }
    FORCE_REACHABILITY_MAPPING = {
        "public": {"forceReachabilityPublic": 1},
        "private": {"forceReachabilityPrivate": 1},
    }
    _UNIX_SOCKET_PREFIX = "/unix/tmp/hivemind-"

    def __init__(self):
        self.peer_id = None
        self._client = None
        self._child = None
        self._alive = False
        self._reader_task = None
        self._listen_task = None

    @classmethod
    async def create(
        cls,
        initial_peers: Optional[Sequence[Union[Multiaddr, str]]] = None,
        *,
        announce_maddrs: Optional[Sequence[Union[Multiaddr, str]]] = None,
        auto_nat: bool = True,
        conn_manager: bool = True,
        dht_mode: str = "server",
        force_reachability: Optional[str] = None,
        host_maddrs: Optional[Sequence[Union[Multiaddr, str]]] = ("/ip4/127.0.0.1/tcp/0",),
        identity_path: Optional[str] = None,
        idle_timeout: float = 30,
        nat_port_map: bool = True,
        quic: bool = False,
        relay_hop_limit: int = 0,
        startup_timeout: float = 15,
        tls: bool = True,
        use_auto_relay: bool = False,
        use_ipfs: bool = False,
        use_relay: bool = True,
        use_relay_hop: bool = False,
        use_relay_discovery: bool = False,
        persistent_conn_max_msg_size: int = DEFAULT_MAX_MSG_SIZE,
    ) -> "P2P":
        """
        Start a new p2pd process and connect to it.
        :param initial_peers: List of bootstrap peers
        :param auto_nat: Enables the AutoNAT service
        :param announce_maddrs: Visible multiaddrs that the peer will announce
                                for external connections from other p2p instances
        :param conn_manager: Enables the Connection Manager
        :param dht_mode: libp2p DHT mode (auto/client/server).
                         Defaults to "server" to make collaborations work in local networks.
                         Details: https://pkg.go.dev/github.com/libp2p/go-libp2p-kad-dht#ModeOpt
        :param force_reachability: Force reachability mode (public/private)
        :param host_maddrs: Multiaddrs to listen for external connections from other p2p instances
        :param identity_path: Path to a pre-generated private key file. If defined, makes the peer ID deterministic.
                              May be generated using ``./p2p-keygen`` from ``go-libp2p-daemon``.
        :param idle_timeout: kill daemon if client has been idle for a given number of
                             seconds before opening persistent streams
        :param nat_port_map: Enables NAT port mapping
        :param quic: Enables the QUIC transport
        :param relay_hop_limit: sets the hop limit for hop relays
        :param startup_timeout: raise a P2PDaemonError if the daemon does not start in ``startup_timeout`` seconds
        :param tls: Enables TLS1.3 channel security protocol
        :param use_auto_relay: enables autorelay
        :param use_ipfs: Bootstrap to IPFS (incompatible with initial_peers)
        :param use_relay: enables circuit relay
        :param use_relay_hop: enables hop for relay
        :param use_relay_discovery: enables passive discovery for relay
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
        if announce_maddrs is not None:
            for addr in announce_maddrs:
                addr = Multiaddr(addr)
                if ("tcp" in addr and addr["tcp"] == "0") or ("udp" in addr and addr["udp"] == "0"):
                    raise ValueError("Please specify an explicit port in announce_maddrs: port 0 is not supported")

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
        if identity_path is not None:
            process_kwargs["id"] = identity_path

        proc_args = self._make_process_args(
            str(p2pd_path),
            autoRelay=use_auto_relay,
            autonat=auto_nat,
            b=need_bootstrap,
            connManager=conn_manager,
            idleTimeout=f"{idle_timeout}s",
            listen=self._daemon_listen_maddr,
            natPortMap=nat_port_map,
            quic=quic,
            relay=use_relay,
            relayDiscovery=use_relay_discovery,
            relayHop=use_relay_hop,
            relayHopLimit=relay_hop_limit,
            tls=tls,
            persistentConnMaxMsgSize=persistent_conn_max_msg_size,
            **process_kwargs,
        )

        env = os.environ.copy()
        env.setdefault("GOLOG_LOG_LEVEL", python_level_to_golog(loglevel))
        env["GOLOG_LOG_FMT"] = "json"

        logger.debug(f"Launching {proc_args}")
        self._child = await asyncio.subprocess.create_subprocess_exec(
            *proc_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
        )
        self._alive = True

        ready = asyncio.Future()
        self._reader_task = asyncio.create_task(self._read_outputs(ready))
        try:
            await asyncio.wait_for(ready, startup_timeout)
        except asyncio.TimeoutError:
            await self.shutdown()
            raise P2PDaemonError(f"Daemon failed to start in {startup_timeout:.1f} seconds")

        self._client = await p2pclient.Client.create(
            control_maddr=self._daemon_listen_maddr,
            listen_maddr=self._client_listen_maddr,
            persistent_conn_max_msg_size=persistent_conn_max_msg_size,
        )

        await self._ping_daemon()
        return self

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

        self._client = await p2pclient.Client.create(self._daemon_listen_maddr, self._client_listen_maddr)

        await self._ping_daemon()
        return self

    async def _ping_daemon(self) -> None:
        self.peer_id, self._visible_maddrs = await self._client.identify()
        logger.debug(f"Launched p2pd with peer id = {self.peer_id}, host multiaddrs = {self._visible_maddrs}")

    async def get_visible_maddrs(self, latest: bool = False) -> List[Multiaddr]:
        """
        Get multiaddrs of the current peer that should be accessible by other peers.

        :param latest: ask the P2P daemon to refresh the visible multiaddrs
        """

        if latest:
            _, self._visible_maddrs = await self._client.identify()

        if not self._visible_maddrs:
            raise ValueError(f"No multiaddrs found for peer {self.peer_id}")

        p2p_maddr = Multiaddr(f"/p2p/{self.peer_id.to_base58()}")
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
    async def send_raw_data(data: bytes, writer: asyncio.StreamWriter, *, chunk_size: int = 2**16) -> None:
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
        input_protobuf_type: Type[Message], reader: asyncio.StreamReader
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
        input_protobuf_type: Type[Message],
        max_prefetch: int = 5,
    ) -> None:
        """
        :param max_prefetch: Maximum number of items to prefetch from the request stream.
          ``max_prefetch <= 0`` means unlimited.

        :note:  Since the cancel messages are sent via the input stream,
          they will not be received while the prefetch buffer is full.
        """

        async def _handle_stream(
            stream_info: StreamInfo, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            context = P2PContext(
                handle_name=name,
                local_id=self.peer_id,
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
                        try:
                            await P2P.send_protobuf(response, writer)
                        except Exception:
                            # The connection is unexpectedly closed by the caller or broken.
                            # The loglevel is DEBUG since the actual error will be reported on the caller
                            logger.debug("Exception while sending response:", exc_info=True)
                            break
                except Exception as e:
                    logger.warning("Handler failed with the exception:", exc_info=True)
                    with suppress(Exception):
                        # Sometimes `e` is a connection error, so it is okay if we fail to report `e` to the caller
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

        await self.add_binary_stream_handler(name, _handle_stream)

    async def _iterate_protobuf_stream_handler(
        self, peer_id: PeerID, name: str, requests: TInputStream, output_protobuf_type: Type[Message]
    ) -> TOutputStream:
        _, reader, writer = await self.call_binary_stream_handler(peer_id, name)

        async def _write_to_stream() -> None:
            async for request in requests:
                await P2P.send_protobuf(request, writer)
            await P2P.send_protobuf(P2P.END_OF_STREAM, writer)

        async def _read_from_stream() -> AsyncIterator[Message]:
            with closing(writer):
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

        writing_task = asyncio.create_task(_write_to_stream())
        return _read_from_stream()

    async def add_protobuf_handler(
        self,
        name: str,
        handler: Callable[
            [Union[TInputProtobuf, TInputStream], P2PContext], Union[Awaitable[TOutputProtobuf], TOutputStream]
        ],
        input_protobuf_type: Type[Message],
        *,
        stream_input: bool = False,
        stream_output: bool = False,
    ) -> None:
        """
        :param stream_input: If True, assume ``handler`` to take ``TInputStream``
                             (not just ``TInputProtobuf``) as input.
        :param stream_output: If True, assume ``handler`` to return ``TOutputStream``
                              (not ``Awaitable[TOutputProtobuf]``).
        """

        if not stream_input and not stream_output:
            await self._add_protobuf_unary_handler(name, handler, input_protobuf_type)
            return

        async def _stream_handler(requests: P2P.TInputStream, context: P2PContext) -> P2P.TOutputStream:
            input = requests if stream_input else await asingle(requests)
            output = handler(input, context)

            if isinstance(output, AsyncIterableABC):
                async for item in output:
                    yield item
            else:
                yield await output

        await self._add_protobuf_stream_handler(name, _stream_handler, input_protobuf_type)

    async def _add_protobuf_unary_handler(
        self,
        handle_name: str,
        handler: Callable[[TInputProtobuf, P2PContext], Awaitable[TOutputProtobuf]],
        input_protobuf_type: Type[Message],
    ) -> None:
        """
        Register a request-response (unary) handler. Unary requests and responses
        are sent through persistent multiplexed connections to the daemon for the
        sake of reducing the number of open files.
        :param handle_name: name of the handler (protocol id)
        :param handler: function handling the unary requests
        :param input_protobuf_type: protobuf type of the request
        """

        async def _unary_handler(request: bytes, remote_id: PeerID) -> bytes:
            input_serialized = input_protobuf_type.FromString(request)
            context = P2PContext(
                handle_name=handle_name,
                local_id=self.peer_id,
                remote_id=remote_id,
            )

            response = await handler(input_serialized, context)
            return response.SerializeToString()

        await self._client.add_unary_handler(handle_name, _unary_handler)

    async def call_protobuf_handler(
        self,
        peer_id: PeerID,
        name: str,
        input: Union[TInputProtobuf, TInputStream],
        output_protobuf_type: Type[Message],
    ) -> Awaitable[TOutputProtobuf]:

        if not isinstance(input, AsyncIterableABC):
            return await self._call_unary_protobuf_handler(peer_id, name, input, output_protobuf_type)

        responses = await self._iterate_protobuf_stream_handler(peer_id, name, input, output_protobuf_type)
        return await asingle(responses)

    async def _call_unary_protobuf_handler(
        self,
        peer_id: PeerID,
        handle_name: str,
        input: TInputProtobuf,
        output_protobuf_type: Type[Message],
    ) -> Awaitable[TOutputProtobuf]:
        serialized_input = input.SerializeToString()
        response = await self._client.call_unary_handler(peer_id, handle_name, serialized_input)
        return output_protobuf_type.FromString(response)

    async def iterate_protobuf_handler(
        self,
        peer_id: PeerID,
        name: str,
        input: Union[TInputProtobuf, TInputStream],
        output_protobuf_type: Type[Message],
    ) -> TOutputStream:
        requests = input if isinstance(input, AsyncIterableABC) else as_aiter(input)
        return await self._iterate_protobuf_stream_handler(peer_id, name, requests, output_protobuf_type)

    def _start_listening(self) -> None:
        async def listen() -> None:
            async with self._client.listen():
                await asyncio.Future()  # Wait until this task will be cancelled in _terminate()

        self._listen_task = asyncio.create_task(listen())

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
        self._terminate()
        if self._child is not None:
            await self._child.wait()

    def _terminate(self) -> None:
        if self._client is not None:
            self._client.close()
        if self._listen_task is not None:
            self._listen_task.cancel()
        if self._reader_task is not None:
            self._reader_task.cancel()

        self._alive = False
        if self._child is not None and self._child.returncode is None:
            self._child.terminate()
            logger.debug(f"Terminated p2pd with id = {self.peer_id}")

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

    async def _read_outputs(self, ready: asyncio.Future) -> None:
        last_line = None
        while True:
            line = await self._child.stdout.readline()
            if not line:  # Stream closed
                break
            last_line = line.rstrip().decode(errors="ignore")

            self._log_p2pd_message(last_line)
            if last_line.startswith("Peer ID:"):
                ready.set_result(None)

        if not ready.done():
            ready.set_exception(P2PDaemonError(f"Daemon failed to start: {last_line}"))

    @staticmethod
    def _log_p2pd_message(line: str) -> None:
        if '"logger"' not in line:  # User-friendly info from p2pd stdout
            logger.debug(line, extra={"caller": "p2pd"})
            return

        try:
            record = json.loads(line)
            caller = record["caller"]

            level = golog_level_to_python(record["level"])
            if level <= logging.WARNING:
                # Many Go loggers are excessively verbose (e.g. show warnings for unreachable peers),
                # so we downgrade INFO and WARNING messages to DEBUG.
                # The Go verbosity can still be controlled via the GOLOG_LOG_LEVEL env variable.
                # Details: https://github.com/ipfs/go-log#golog_log_level
                level = logging.DEBUG

            message = record["msg"]
            if "error" in record:
                message += f": {record['error']}"

            logger.log(
                level,
                message,
                extra={
                    "origin_created": datetime.strptime(record["ts"], "%Y-%m-%dT%H:%M:%S.%f%z").timestamp(),
                    "caller": caller,
                },
            )
        except Exception:
            # Parsing errors are unlikely, but we don't want to lose these messages anyway
            logger.warning(line, extra={"caller": "p2pd"})
            logger.exception("Failed to parse go-log message:")
