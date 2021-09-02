"""
Originally taken from: https://github.com/mhchia/py-libp2p-daemon-bindings
Licence: MIT
Author: Kevin Mai-Husan Chia
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterable, Sequence, Tuple

from multiaddr import Multiaddr

from hivemind.p2p.p2p_daemon_bindings.control import (
    DEFAULT_MAX_MSG_SIZE,
    ControlClient,
    DaemonConnector,
    StreamHandler,
    TUnaryHandler,
)
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID, PeerInfo, StreamInfo


class Client:
    control: ControlClient

    def __init__(self, *, _initialized_with_create=False) -> None:
        assert _initialized_with_create, "Please use Client.create coroutine to spawn new client instances"
        self.control = None

    @classmethod
    async def create(
        cls,
        control_maddr: Multiaddr = None,
        listen_maddr: Multiaddr = None,
        *,
        persistent_conn_max_msg_size: int = DEFAULT_MAX_MSG_SIZE,
    ) -> "Client":
        client = cls(_initialized_with_create=True)

        daemon_connector = DaemonConnector(control_maddr=control_maddr)
        client.control = await ControlClient.create(
            daemon_connector=daemon_connector,
            listen_maddr=listen_maddr,
            persistent_conn_max_msg_size=persistent_conn_max_msg_size,
        )

        return client

    def close(self) -> None:
        self.control.close()

    def __del__(self):
        self.close()

    @asynccontextmanager
    async def listen(self) -> AsyncIterator["Client"]:
        """
        Starts to listen incoming connections for handlers registered via stream_handler.
        :return:
        """
        async with self.control.listen():
            yield self

    async def add_unary_handler(self, proto: str, handler: TUnaryHandler):
        await self.control.add_unary_handler(proto, handler)

    async def call_unary_handler(self, peer_id: PeerID, proto: str, data: bytes) -> bytes:
        return await self.control.call_unary_handler(peer_id, proto, data)

    async def identify(self) -> Tuple[PeerID, Tuple[Multiaddr, ...]]:
        """
        Get current node peer id and list of addresses
        """
        return await self.control.identify()

    async def connect(self, peer_id: PeerID, maddrs: Iterable[Multiaddr]) -> None:
        """
        Connect to p2p node with specified addresses and peer id.
        :peer_id: node peer id you want connect to
        :maddrs: node multiaddresses you want connect to. Of course, it must be reachable.
        """
        await self.control.connect(peer_id=peer_id, maddrs=maddrs)

    async def list_peers(self) -> Tuple[PeerInfo, ...]:
        """
        Get list of peers that node connect to
        """
        return await self.control.list_peers()

    async def disconnect(self, peer_id: PeerID) -> None:
        """
        Disconnect from node with specified peer id
        :peer_id: node peer id you want disconnect from
        """
        await self.control.disconnect(peer_id=peer_id)

    async def stream_open(
        self, peer_id: PeerID, protocols: Sequence[str]
    ) -> Tuple[StreamInfo, asyncio.StreamReader, asyncio.StreamWriter]:
        """
        Open a stream to call other peer (with peer_id) handler for specified protocols
        :peer_id: other peer id
        :protocols: list of protocols for other peer handling
        :return: Returns tuple of stream info (info about connection to second peer) and reader/writer
        """
        return await self.control.stream_open(peer_id=peer_id, protocols=protocols)

    async def stream_handler(self, proto: str, handler_cb: StreamHandler) -> None:
        """
        Register a stream handler
        :param proto: protocols that handler serves
        :param handler_cb: handler callback
        :return:
        """
        await self.control.stream_handler(proto=proto, handler_cb=handler_cb)
