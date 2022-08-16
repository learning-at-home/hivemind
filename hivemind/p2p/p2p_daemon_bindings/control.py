"""
Originally taken from: https://github.com/mhchia/py-libp2p-daemon-bindings
Licence: MIT
Author: Kevin Mai-Husan Chia
"""

import asyncio
from contextlib import asynccontextmanager, closing
from typing import AsyncIterator, Awaitable, Callable, Dict, Iterable, Optional, Sequence, Tuple
from uuid import UUID, uuid4

from multiaddr import Multiaddr, protocols

from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID, PeerInfo, StreamInfo
from hivemind.p2p.p2p_daemon_bindings.utils import (
    DispatchFailure,
    P2PDaemonError,
    P2PHandlerError,
    raise_if_failed,
    read_pbmsg_safe,
    write_pbmsg,
)
from hivemind.proto import p2pd_pb2 as p2pd_pb
from hivemind.utils.logging import get_logger

StreamHandler = Callable[[StreamInfo, asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]

SUPPORT_CONN_PROTOCOLS = (
    protocols.P_IP4,
    # protocols.P_IP6,
    protocols.P_UNIX,
)
SUPPORTED_PROTOS = (protocols.protocol_with_code(proto) for proto in SUPPORT_CONN_PROTOCOLS)
logger = get_logger(__name__)

DEFAULT_MAX_MSG_SIZE = 4 * 1024**2
MAX_UNARY_PAYLOAD_SIZE = DEFAULT_MAX_MSG_SIZE // 2
# note: we check vs. 2x max message size to account for serialization overhead. The actual overhead is
# typically smaller. We err on the side of streaming, because even 2MB messages can be streamed efficiently.


def parse_conn_protocol(maddr: Multiaddr) -> int:
    proto_codes = set(proto.code for proto in maddr.protocols())
    proto_cand = proto_codes.intersection(SUPPORT_CONN_PROTOCOLS)
    if len(proto_cand) != 1:
        raise ValueError(
            f"connection protocol should be only one protocol out of {SUPPORTED_PROTOS}" f", maddr={maddr}"
        )
    return tuple(proto_cand)[0]


class DaemonConnector:
    DEFAULT_CONTROL_MADDR = "/unix/tmp/p2pd.sock"

    def __init__(self, control_maddr: Multiaddr = Multiaddr(DEFAULT_CONTROL_MADDR)) -> None:
        self.control_maddr = control_maddr
        self.proto_code = parse_conn_protocol(self.control_maddr)

    async def open_connection(self) -> (asyncio.StreamReader, asyncio.StreamWriter):
        if self.proto_code == protocols.P_UNIX:
            control_path = self.control_maddr.value_for_protocol(protocols.P_UNIX)
            return await asyncio.open_unix_connection(control_path)
        elif self.proto_code == protocols.P_IP4:
            host = self.control_maddr.value_for_protocol(protocols.P_IP4)
            port = int(self.control_maddr.value_for_protocol(protocols.P_TCP))
            return await asyncio.open_connection(host, port)
        else:
            raise ValueError(f"Protocol not supported: {protocols.protocol_with_code(self.proto_code)}")

    async def open_persistent_connection(self) -> (asyncio.StreamReader, asyncio.StreamWriter):
        """
        Open connection to daemon and upgrade it to a persistent one
        """
        reader, writer = await self.open_connection()
        req = p2pd_pb.Request(type=p2pd_pb.Request.PERSISTENT_CONN_UPGRADE)
        await write_pbmsg(writer, req)

        response = p2pd_pb.Response()
        await read_pbmsg_safe(reader, response)

        if response.type == "ERROR":
            raise P2PDaemonError(response.error.msg)

        return reader, writer


TUnaryHandler = Callable[[bytes, PeerID], Awaitable[bytes]]
CallID = UUID


class ControlClient:
    DEFAULT_LISTEN_MADDR = "/unix/tmp/p2pclient.sock"

    def __init__(
        self,
        daemon_connector: DaemonConnector,
        listen_maddr: Multiaddr = Multiaddr(DEFAULT_LISTEN_MADDR),
        *,
        _initialized_with_create: bool = False,
        persistent_conn_max_msg_size: int = DEFAULT_MAX_MSG_SIZE,
    ) -> None:
        assert _initialized_with_create, "Please use ControlClient.create coroutine to spawn new control instances"

        self.persistent_conn_max_msg_size = persistent_conn_max_msg_size

        self.listen_maddr = listen_maddr
        self.daemon_connector = daemon_connector
        self.handlers: Dict[str, StreamHandler] = {}

        self.unary_handlers: Dict[str, TUnaryHandler] = {}

        self._pending_messages: asyncio.Queue[p2pd_pb.PersistentConnectionRequest] = asyncio.Queue()
        self._pending_calls: Dict[CallID, asyncio.Future[bytes]] = {}
        self._handler_tasks: Dict[CallID, asyncio.Task] = {}

        self._read_task: Optional[asyncio.Task] = None
        self._write_task: Optional[asyncio.Task] = None

    @classmethod
    async def create(
        cls,
        daemon_connector: DaemonConnector,
        listen_maddr: Multiaddr = Multiaddr(DEFAULT_LISTEN_MADDR),
        use_persistent_conn: bool = True,
        persistent_conn_max_msg_size=2 << 22,
    ) -> "ControlClient":
        control = cls(
            daemon_connector,
            listen_maddr,
            _initialized_with_create=True,
            persistent_conn_max_msg_size=persistent_conn_max_msg_size,
        )

        if use_persistent_conn:
            await control._ensure_persistent_conn()

        return control

    def close(self) -> None:
        if self._read_task is not None:
            self._read_task.cancel()
        if self._write_task is not None:
            self._write_task.cancel()

    def __del__(self):
        self.close()

    async def _handler(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        pb_stream_info = p2pd_pb.StreamInfo()  # type: ignore
        await read_pbmsg_safe(reader, pb_stream_info)
        stream_info = StreamInfo.from_protobuf(pb_stream_info)
        try:
            handler = self.handlers[stream_info.proto]
        except KeyError as e:
            # should never enter here... daemon should reject the stream for us.
            writer.close()
            raise DispatchFailure(e)
        await handler(stream_info, reader, writer)

    @asynccontextmanager
    async def listen(self) -> AsyncIterator["ControlClient"]:
        proto_code = parse_conn_protocol(self.listen_maddr)
        if proto_code == protocols.P_UNIX:
            listen_path = self.listen_maddr.value_for_protocol(protocols.P_UNIX)
            server = await asyncio.start_unix_server(self._handler, path=listen_path)
        elif proto_code == protocols.P_IP4:
            host = self.listen_maddr.value_for_protocol(protocols.P_IP4)
            port = int(self.listen_maddr.value_for_protocol(protocols.P_TCP))
            server = await asyncio.start_server(self._handler, port=port, host=host)
        else:
            raise ValueError(f"Protocol not supported: {protocols.protocol_with_code(proto_code)}")

        async with server:
            yield self

    async def _read_from_persistent_conn(self, reader: asyncio.StreamReader):
        while True:
            resp = p2pd_pb.PersistentConnectionResponse()
            try:
                await read_pbmsg_safe(reader, resp)
            except asyncio.IncompleteReadError:
                break

            call_id = UUID(bytes=resp.callId)

            if resp.HasField("callUnaryResponse"):
                if call_id in self._pending_calls and resp.callUnaryResponse.HasField("response"):
                    self._pending_calls[call_id].set_result(resp.callUnaryResponse.response)
                elif call_id in self._pending_calls and resp.callUnaryResponse.HasField("error"):
                    remote_exc = P2PHandlerError(resp.callUnaryResponse.error.decode(errors="ignore"))
                    self._pending_calls[call_id].set_exception(remote_exc)
                else:
                    logger.debug(f"Received unexpected unary call: {resp}")

            elif resp.HasField("requestHandling"):
                handler_task = asyncio.create_task(self._handle_persistent_request(call_id, resp.requestHandling))
                self._handler_tasks[call_id] = handler_task

            elif call_id in self._handler_tasks and resp.HasField("cancel"):
                self._handler_tasks[call_id].cancel()

            elif call_id in self._pending_calls and resp.HasField("daemonError"):
                daemon_exc = P2PDaemonError(resp.daemonError.message)
                self._pending_calls[call_id].set_exception(daemon_exc)

            elif call_id in self._pending_calls:
                self._pending_calls[call_id].set_result(None)

            else:
                logger.debug(f"Received unexpected response from daemon: {resp}")

    async def _write_to_persistent_conn(self, writer: asyncio.StreamWriter):
        with closing(writer):
            while True:
                msg = await self._pending_messages.get()
                await write_pbmsg(writer, msg)

    async def _handle_persistent_request(self, call_id: UUID, request: p2pd_pb.CallUnaryRequest):
        if request.proto not in self.unary_handlers:
            logger.warning(f"Protocol {request.proto} not supported")
            return

        try:
            remote_id = PeerID(request.peer)
            response_payload: bytes = await self.unary_handlers[request.proto](request.data, remote_id)
            response = p2pd_pb.CallUnaryResponse(response=response_payload)

        except Exception as e:
            response = p2pd_pb.CallUnaryResponse(error=repr(e).encode())

        payload = p2pd_pb.PersistentConnectionRequest(callId=call_id.bytes, unaryResponse=response)
        if payload.ByteSize() <= self.persistent_conn_max_msg_size:
            await self._pending_messages.put(payload)
        else:
            error_msg = p2pd_pb.PersistentConnectionRequest(
                callId=call_id.bytes,
                callUnaryResponse=p2pd_pb.CallUnaryResponse(
                    error=b"response size exceeds message size limit",
                ),
            )
            await self._pending_messages.put(error_msg)

        self._handler_tasks.pop(call_id)

    async def _cancel_unary_call(self, call_id: UUID):
        await self._pending_messages.put(
            p2pd_pb.PersistentConnectionRequest(
                callId=call_id.bytes,
                cancel=p2pd_pb.Cancel(),
            ),
        )

    async def _ensure_persistent_conn(self):
        reader, writer = await self.daemon_connector.open_persistent_connection()

        self._read_task = asyncio.create_task(self._read_from_persistent_conn(reader))
        self._write_task = asyncio.create_task(self._write_to_persistent_conn(writer))

    async def add_unary_handler(self, proto: str, handler: TUnaryHandler, balanced: bool = False) -> None:
        if proto in self.unary_handlers:
            raise P2PDaemonError(f"Handler for protocol {proto} already registered")
        self.unary_handlers[proto] = handler

        call_id = uuid4()
        req = p2pd_pb.PersistentConnectionRequest(
            callId=call_id.bytes,
            addUnaryHandler=p2pd_pb.AddUnaryHandlerRequest(proto=proto, balanced=balanced),
        )

        self._pending_calls[call_id] = asyncio.Future()
        await self._pending_messages.put(req)
        await self._pending_calls[call_id]

    async def remove_unary_handler(self, proto: str) -> None:
        if proto not in self.unary_handlers:
            raise P2PDaemonError(f"Handler for protocol {proto} is not registered")

        call_id = uuid4()
        req = p2pd_pb.PersistentConnectionRequest(
            callId=call_id.bytes,
            removeUnaryHandler=p2pd_pb.RemoveUnaryHandlerRequest(proto=proto),
        )

        self._pending_calls[call_id] = asyncio.Future()
        await self._pending_messages.put(req)
        await self._pending_calls[call_id]

        del self.unary_handlers[proto]

    async def call_unary_handler(self, peer_id: PeerID, proto: str, data: bytes) -> bytes:
        call_id = uuid4()
        call_unary_req = p2pd_pb.CallUnaryRequest(
            peer=peer_id.to_bytes(),
            proto=proto,
            data=data,
        )
        req = p2pd_pb.PersistentConnectionRequest(
            callId=call_id.bytes,
            callUnary=call_unary_req,
        )

        if req.ByteSize() > self.persistent_conn_max_msg_size:
            raise P2PDaemonError(f"Message size exceeds set limit {self.persistent_conn_max_msg_size}")

        try:
            self._pending_calls[call_id] = asyncio.Future()
            await self._pending_messages.put(req)
            return await self._pending_calls[call_id]

        except asyncio.CancelledError:
            await self._cancel_unary_call(call_id)
            raise

        finally:
            self._pending_calls.pop(call_id, None)

    async def identify(self) -> Tuple[PeerID, Tuple[Multiaddr, ...]]:
        reader, writer = await self.daemon_connector.open_connection()
        req = p2pd_pb.Request(type=p2pd_pb.Request.IDENTIFY)
        await write_pbmsg(writer, req)

        resp = p2pd_pb.Response()  # type: ignore
        await read_pbmsg_safe(reader, resp)
        writer.close()

        raise_if_failed(resp)
        peer_id_bytes = resp.identify.id
        maddrs_bytes = resp.identify.addrs

        maddrs = tuple(Multiaddr(maddr_bytes) for maddr_bytes in maddrs_bytes)
        peer_id = PeerID(peer_id_bytes)

        return peer_id, maddrs

    async def connect(self, peer_id: PeerID, maddrs: Iterable[Multiaddr]) -> None:
        reader, writer = await self.daemon_connector.open_connection()

        maddrs_bytes = [i.to_bytes() for i in maddrs]
        connect_req = p2pd_pb.ConnectRequest(peer=peer_id.to_bytes(), addrs=maddrs_bytes)
        req = p2pd_pb.Request(type=p2pd_pb.Request.CONNECT, connect=connect_req)
        await write_pbmsg(writer, req)

        resp = p2pd_pb.Response()  # type: ignore
        await read_pbmsg_safe(reader, resp)
        writer.close()
        raise_if_failed(resp)

    async def list_peers(self) -> Tuple[PeerInfo, ...]:
        req = p2pd_pb.Request(type=p2pd_pb.Request.LIST_PEERS)
        reader, writer = await self.daemon_connector.open_connection()
        await write_pbmsg(writer, req)
        resp = p2pd_pb.Response()  # type: ignore
        await read_pbmsg_safe(reader, resp)
        writer.close()
        raise_if_failed(resp)

        peers = tuple(PeerInfo.from_protobuf(pinfo) for pinfo in resp.peers)
        return peers

    async def disconnect(self, peer_id: PeerID) -> None:
        disconnect_req = p2pd_pb.DisconnectRequest(peer=peer_id.to_bytes())
        req = p2pd_pb.Request(type=p2pd_pb.Request.DISCONNECT, disconnect=disconnect_req)
        reader, writer = await self.daemon_connector.open_connection()
        await write_pbmsg(writer, req)
        resp = p2pd_pb.Response()  # type: ignore
        await read_pbmsg_safe(reader, resp)
        writer.close()
        raise_if_failed(resp)

    async def stream_open(
        self, peer_id: PeerID, protocols: Sequence[str]
    ) -> Tuple[StreamInfo, asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await self.daemon_connector.open_connection()

        stream_open_req = p2pd_pb.StreamOpenRequest(peer=peer_id.to_bytes(), proto=list(protocols))
        req = p2pd_pb.Request(type=p2pd_pb.Request.STREAM_OPEN, streamOpen=stream_open_req)
        await write_pbmsg(writer, req)

        resp = p2pd_pb.Response()  # type: ignore
        await read_pbmsg_safe(reader, resp)
        raise_if_failed(resp)

        pb_stream_info = resp.streamInfo
        stream_info = StreamInfo.from_protobuf(pb_stream_info)

        return stream_info, reader, writer

    async def stream_handler(self, proto: str, handler_cb: StreamHandler, balanced: bool = False) -> None:
        self.handlers[proto] = handler_cb

        reader, writer = await self.daemon_connector.open_connection()

        req = p2pd_pb.Request(
            type=p2pd_pb.Request.STREAM_HANDLER,
            streamHandler=p2pd_pb.StreamHandlerRequest(
                addr=self.listen_maddr.to_bytes(),
                proto=[proto],
                balanced=balanced,
            ),
        )
        await write_pbmsg(writer, req)

        resp = p2pd_pb.Response()  # type: ignore
        await read_pbmsg_safe(reader, resp)
        writer.close()
        raise_if_failed(resp)

    async def remove_stream_handler(self, proto: str) -> None:
        reader, writer = await self.daemon_connector.open_connection()

        req = p2pd_pb.Request(
            type=p2pd_pb.Request.REMOVE_STREAM_HANDLER,
            removeStreamHandler=p2pd_pb.RemoveStreamHandlerRequest(
                addr=self.listen_maddr.to_bytes(),
                proto=[proto],
            ),
        )
        await write_pbmsg(writer, req)

        resp = p2pd_pb.Response()  # type: ignore
        await read_pbmsg_safe(reader, resp)
        writer.close()
        raise_if_failed(resp)

        del self.handlers[proto]
