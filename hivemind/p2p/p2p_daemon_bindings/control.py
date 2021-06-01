"""
Originally taken from: https://github.com/mhchia/py-libp2p-daemon-bindings
Licence: MIT
Author: Kevin Mai-Husan Chia
"""

import asyncio
from contextlib import asynccontextmanager
from typing import (AsyncIterator, Awaitable, Callable, Dict, Iterable,
                    Sequence, Tuple)

from multiaddr import Multiaddr, protocols

from hivemind.p2p.p2p_daemon_bindings.datastructures import (PeerID, PeerInfo,
                                                             StreamInfo)
from hivemind.p2p.p2p_daemon_bindings.utils import (DispatchFailure,
                                                    raise_if_failed,
                                                    read_pbmsg_safe,
                                                    write_pbmsg)
from hivemind.proto import p2pd_pb2 as p2pd_pb
from hivemind.utils.logging import get_logger

StreamHandler = Callable[[StreamInfo, asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]

SUPPORT_CONN_PROTOCOLS = (
    protocols.P_IP4,
    # protocols.P_IP6,
    protocols.P_UNIX,
)
SUPPORTED_PROTOS = (
    protocols.protocol_with_code(proto) for proto in SUPPORT_CONN_PROTOCOLS
)
logger = get_logger(__name__)


def parse_conn_protocol(maddr: Multiaddr) -> int:
    proto_codes = set(proto.code for proto in maddr.protocols())
    proto_cand = proto_codes.intersection(SUPPORT_CONN_PROTOCOLS)
    if len(proto_cand) != 1:
        raise ValueError(
            f"connection protocol should be only one protocol out of {SUPPORTED_PROTOS}"
            f", maddr={maddr}"
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
            logger.debug(f"DaemonConnector {self} opens connection to {self.control_maddr}")
            return await asyncio.open_unix_connection(control_path)
        elif self.proto_code == protocols.P_IP4:
            host = self.control_maddr.value_for_protocol(protocols.P_IP4)
            port = int(self.control_maddr.value_for_protocol(protocols.P_TCP))
            return await asyncio.open_connection(host, port)
        else:
            raise ValueError(
                f"Protocol not supported: {protocols.protocol_with_code(self.proto_code)}"
            )


class ControlClient:
    DEFAULT_LISTEN_MADDR = "/unix/tmp/p2pclient.sock"

    def __init__(
            self, daemon_connector: DaemonConnector, listen_maddr: Multiaddr = Multiaddr(DEFAULT_LISTEN_MADDR)
    ) -> None:
        self.listen_maddr = listen_maddr
        self.daemon_connector = daemon_connector
        self.handlers: Dict[str, StreamHandler] = {}

    async def _handler(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        pb_stream_info = p2pd_pb.StreamInfo()  # type: ignore
        await read_pbmsg_safe(reader, pb_stream_info)
        stream_info = StreamInfo.from_protobuf(pb_stream_info)
        logger.debug(f"New incoming stream: {stream_info}")
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
            raise ValueError(
                f"Protocol not supported: {protocols.protocol_with_code(proto_code)}"
            )

        async with server:
            logger.info(f"DaemonConnector {self} starts listening to {self.listen_maddr}")
            yield self

        logger.info(f"DaemonConnector {self} closed")

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
        connect_req = p2pd_pb.ConnectRequest(
            peer=peer_id.to_bytes(), addrs=maddrs_bytes
        )
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
        req = p2pd_pb.Request(
            type=p2pd_pb.Request.DISCONNECT, disconnect=disconnect_req
        )
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

        stream_open_req = p2pd_pb.StreamOpenRequest(
            peer=peer_id.to_bytes(), proto=list(protocols)
        )
        req = p2pd_pb.Request(
            type=p2pd_pb.Request.STREAM_OPEN, streamOpen=stream_open_req
        )
        await write_pbmsg(writer, req)

        resp = p2pd_pb.Response()  # type: ignore
        await read_pbmsg_safe(reader, resp)
        raise_if_failed(resp)

        pb_stream_info = resp.streamInfo
        stream_info = StreamInfo.from_protobuf(pb_stream_info)

        return stream_info, reader, writer

    async def stream_handler(self, proto: str, handler_cb: StreamHandler) -> None:
        reader, writer = await self.daemon_connector.open_connection()

        listen_path_maddr_bytes = self.listen_maddr.to_bytes()
        stream_handler_req = p2pd_pb.StreamHandlerRequest(
            addr=listen_path_maddr_bytes, proto=[proto]
        )
        req = p2pd_pb.Request(
            type=p2pd_pb.Request.STREAM_HANDLER, streamHandler=stream_handler_req
        )
        await write_pbmsg(writer, req)

        resp = p2pd_pb.Response()  # type: ignore
        await read_pbmsg_safe(reader, resp)
        writer.close()
        raise_if_failed(resp)

        # if success, add the handler to the dict
        self.handlers[proto] = handler_cb
