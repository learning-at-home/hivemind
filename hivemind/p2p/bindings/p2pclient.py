from typing import AsyncIterator, Iterable, Sequence, Tuple

import asyncio
from control import ControlClient, DaemonConnector, StreamHandler
from contextlib import asynccontextmanager
from multiaddr import Multiaddr
from datastructures import PeerInfo, StreamInfo, ID


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
        async with self.control.listen():
            yield self

    async def close(self) -> None:
        # await self.control.close()
        pass

    async def identify(self) -> Tuple[ID, Tuple[Multiaddr, ...]]:
        return await self.control.identify()

    async def connect(self, peer_id: ID, maddrs: Iterable[Multiaddr]) -> None:
        await self.control.connect(peer_id=peer_id, maddrs=maddrs)

    async def list_peers(self) -> Tuple[PeerInfo, ...]:
        return await self.control.list_peers()

    async def disconnect(self, peer_id: ID) -> None:
        await self.control.disconnect(peer_id=peer_id)

    async def stream_open(
        self, peer_id: ID, protocols: Sequence[str]
    ) -> Tuple[StreamInfo, asyncio.StreamReader, asyncio.StreamWriter]:
        return await self.control.stream_open(peer_id=peer_id, protocols=protocols)

    async def stream_handler(self, proto: str, handler_cb: StreamHandler) -> None:
        await self.control.stream_handler(proto=proto, handler_cb=handler_cb)
