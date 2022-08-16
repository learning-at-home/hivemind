import asyncio
from typing import AsyncIterator

import pytest

from hivemind.p2p import P2P, P2PContext, P2PDaemonError, ServicerBase
from hivemind.proto import test_pb2
from hivemind.utils.asyncio import anext


@pytest.fixture
async def server_client():
    server = await P2P.create()
    client = await P2P.create(initial_peers=await server.get_visible_maddrs())
    yield server, client

    await asyncio.gather(server.shutdown(), client.shutdown())


class UnaryUnaryServicer(ServicerBase):
    async def rpc_square(self, request: test_pb2.TestRequest, _context: P2PContext) -> test_pb2.TestResponse:
        return test_pb2.TestResponse(number=request.number**2)


@pytest.mark.asyncio
async def test_unary_unary(server_client):
    server, client = server_client
    servicer = UnaryUnaryServicer()
    await servicer.add_p2p_handlers(server)
    stub = UnaryUnaryServicer.get_stub(client, server.peer_id)

    assert await stub.rpc_square(test_pb2.TestRequest(number=10)) == test_pb2.TestResponse(number=100)


class StreamUnaryServicer(ServicerBase):
    async def rpc_sum(
        self, stream: AsyncIterator[test_pb2.TestRequest], _context: P2PContext
    ) -> test_pb2.TestResponse:
        result = 0
        async for item in stream:
            result += item.number
        return test_pb2.TestResponse(number=result)


@pytest.mark.asyncio
async def test_stream_unary(server_client):
    server, client = server_client
    servicer = StreamUnaryServicer()
    await servicer.add_p2p_handlers(server)
    stub = StreamUnaryServicer.get_stub(client, server.peer_id)

    async def generate_requests() -> AsyncIterator[test_pb2.TestRequest]:
        for i in range(10):
            yield test_pb2.TestRequest(number=i)

    assert await stub.rpc_sum(generate_requests()) == test_pb2.TestResponse(number=45)


class UnaryStreamServicer(ServicerBase):
    async def rpc_count(
        self, request: test_pb2.TestRequest, _context: P2PContext
    ) -> AsyncIterator[test_pb2.TestResponse]:
        for i in range(request.number):
            yield test_pb2.TestResponse(number=i)


@pytest.mark.asyncio
async def test_unary_stream(server_client):
    server, client = server_client
    servicer = UnaryStreamServicer()
    await servicer.add_p2p_handlers(server)
    stub = UnaryStreamServicer.get_stub(client, server.peer_id)

    stream = await stub.rpc_count(test_pb2.TestRequest(number=10))
    assert [item.number async for item in stream] == list(range(10))


class StreamStreamServicer(ServicerBase):
    async def rpc_powers(
        self, stream: AsyncIterator[test_pb2.TestRequest], _context: P2PContext
    ) -> AsyncIterator[test_pb2.TestResponse]:
        async for item in stream:
            yield test_pb2.TestResponse(number=item.number**2)
            yield test_pb2.TestResponse(number=item.number**3)


@pytest.mark.asyncio
async def test_stream_stream(server_client):
    server, client = server_client
    servicer = StreamStreamServicer()
    await servicer.add_p2p_handlers(server)
    stub = StreamStreamServicer.get_stub(client, server.peer_id)

    async def generate_requests() -> AsyncIterator[test_pb2.TestRequest]:
        for i in range(10):
            yield test_pb2.TestRequest(number=i)

    stream = await stub.rpc_powers(generate_requests())
    i = 0
    async for item in stream:
        if i % 2 == 0:
            assert item == test_pb2.TestResponse(number=(i // 2) ** 2)
        else:
            assert item == test_pb2.TestResponse(number=(i // 2) ** 3)
        i += 1


@pytest.mark.parametrize(
    "cancel_reason",
    ["close_connection", "close_generator"],
)
@pytest.mark.asyncio
async def test_unary_stream_cancel(server_client, cancel_reason):
    handler_cancelled = False

    class ExampleServicer(ServicerBase):
        async def rpc_wait(
            self, request: test_pb2.TestRequest, _context: P2PContext
        ) -> AsyncIterator[test_pb2.TestResponse]:
            try:
                yield test_pb2.TestResponse(number=request.number + 1)
                await asyncio.sleep(2)
                yield test_pb2.TestResponse(number=request.number + 2)
            except asyncio.CancelledError:
                nonlocal handler_cancelled
                handler_cancelled = True
                raise

    server, client = server_client
    servicer = ExampleServicer()
    await servicer.add_p2p_handlers(server)

    if cancel_reason == "close_connection":
        _, reader, writer = await client.call_binary_stream_handler(server.peer_id, "ExampleServicer.rpc_wait")
        await P2P.send_protobuf(test_pb2.TestRequest(number=10), writer)
        await P2P.send_protobuf(P2P.END_OF_STREAM, writer)

        response, _ = await P2P.receive_protobuf(test_pb2.TestResponse, reader)
        assert response == test_pb2.TestResponse(number=11)
        await asyncio.sleep(0.25)

        writer.close()
    elif cancel_reason == "close_generator":
        stub = ExampleServicer.get_stub(client, server.peer_id)
        iter = await stub.rpc_wait(test_pb2.TestRequest(number=10))

        assert await anext(iter) == test_pb2.TestResponse(number=11)
        await asyncio.sleep(0.25)

        await iter.aclose()
    else:
        assert False, f"Unknown cancel_reason = `{cancel_reason}`"

    await asyncio.sleep(0.25)
    assert handler_cancelled


@pytest.mark.asyncio
async def test_removing_unary_handlers(server_client):
    server1, client = server_client
    server2 = await P2P.replicate(server1.daemon_listen_maddr)
    servicer = UnaryUnaryServicer()
    stub = UnaryUnaryServicer.get_stub(client, server1.peer_id)

    for server in [server1, server2, server1]:
        await servicer.add_p2p_handlers(server)
        assert await stub.rpc_square(test_pb2.TestRequest(number=10)) == test_pb2.TestResponse(number=100)

        await servicer.remove_p2p_handlers(server)
        with pytest.raises((P2PDaemonError, ConnectionError)):
            await stub.rpc_square(test_pb2.TestRequest(number=10))

    await asyncio.gather(server2.shutdown())


@pytest.mark.asyncio
async def test_removing_stream_handlers(server_client):
    server1, client = server_client
    server2 = await P2P.replicate(server1.daemon_listen_maddr)
    servicer = UnaryStreamServicer()
    stub = UnaryStreamServicer.get_stub(client, server1.peer_id)

    for server in [server1, server2, server1]:
        await servicer.add_p2p_handlers(server)
        stream = await stub.rpc_count(test_pb2.TestRequest(number=10))
        assert [item.number async for item in stream] == list(range(10))

        await servicer.remove_p2p_handlers(server)
        with pytest.raises((P2PDaemonError, ConnectionError)):
            stream = await stub.rpc_count(test_pb2.TestRequest(number=10))
            outputs = [item.number async for item in stream]
            if not outputs:
                raise P2PDaemonError("Daemon has reset the connection")

    await asyncio.gather(server2.shutdown())
