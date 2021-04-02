from typing import AsyncIterator, Iterable, Sequence, Tuple

import asyncio
from hivemind.p2p.p2p_daemon_bindings.control import ControlClient, DaemonConnector, StreamHandler
from contextlib import asynccontextmanager
from multiaddr import Multiaddr
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerInfo, StreamInfo, ID


class Client:
    control: ControlClient

    def __init__(
        self, control_maddr: Multiaddr = None, listen_maddr: Multiaddr = None
    ) -> None:
        daemon_connector = DaemonConnector(control_maddr=control_maddr)
        self.control = ControlClient(
            daemon_connector=daemon_connector, listen_maddr=listen_maddr
        )

    @asynccontextmanager
    async def listen(self) -> AsyncIterator["Client"]:
        """
        Starts to listen incoming connections for handlers registered via stream_handler.
        :return:
        """
        async with self.control.listen():
            yield self

    async def identify(self) -> Tuple[ID, Tuple[Multiaddr, ...]]:
        """
        Get current node peer id and list of addresses
        """
        return await self.control.identify()

    async def connect(self, peer_id: ID, maddrs: Iterable[Multiaddr]) -> None:
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

    async def disconnect(self, peer_id: ID) -> None:
        """
        Disconnect from node with specified peer id
        :peer_id:
        """
        await self.control.disconnect(peer_id=peer_id)

    async def stream_open(
        self, peer_id: ID, protocols: Sequence[str]
    ) -> Tuple[StreamInfo, asyncio.StreamReader, asyncio.StreamWriter]:
        """
        Open a stream to call other peer (with peer_id) handler for specified protocols
        :peer_id:
        :protocols:
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
