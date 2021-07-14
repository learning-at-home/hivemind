import asyncio
from typing import AsyncIterable

import pytest

from hivemind.p2p import P2P, P2PContext, ServicerBase
from hivemind.proto import test_pb2


@pytest.fixture
async def server_client():
    server = await P2P.create()
    client = await P2P.create(initial_peers=await server.get_visible_maddrs())
    yield server, client

    await asyncio.gather(server.shutdown(), client.shutdown())


@pytest.mark.asyncio
async def test_unary_unary(server_client):
    class ExampleServicer(ServicerBase):
        async def rpc_square(self, request: test_pb2.TestRequest, _: P2PContext) -> test_pb2.TestResponse:
            return test_pb2.TestResponse(number=request.number ** 2)

    server, client = server_client
    servicer = ExampleServicer()
    await servicer.add_p2p_handlers(server)
    stub = servicer.get_stub(client, server.id)

    assert await stub.rpc_square(test_pb2.TestRequest(number=10)) == test_pb2.TestResponse(number=100)


@pytest.mark.asyncio
async def test_stream_unary(server_client):
    class ExampleServicer(ServicerBase):
        async def rpc_sum(self, request: AsyncIterable[test_pb2.TestRequest], _: P2PContext) -> test_pb2.TestResponse:
            result = 0
            async for item in request:
                result += item.number
            return test_pb2.TestResponse(number=result)

    server, client = server_client
    servicer = ExampleServicer()
    await servicer.add_p2p_handlers(server)
    stub = servicer.get_stub(client, server.id)

    async def generate_requests() -> AsyncIterable[test_pb2.TestRequest]:
        for i in range(10):
            yield test_pb2.TestRequest(number=i)

    assert await stub.rpc_sum(generate_requests()) == test_pb2.TestResponse(number=45)


@pytest.mark.asyncio
async def test_unary_stream(server_client):
    class ExampleServicer(ServicerBase):
        async def rpc_count(self, request: test_pb2.TestRequest, _: P2PContext) -> AsyncIterable[test_pb2.TestResponse]:
            for i in range(request.number):
                yield test_pb2.TestResponse(number=i)

    server, client = server_client
    servicer = ExampleServicer()
    await servicer.add_p2p_handlers(server)
    stub = servicer.get_stub(client, server.id)

    i = 0
    async for item in stub.rpc_count(test_pb2.TestRequest(number=10)):
        assert item == test_pb2.TestResponse(number=i)
        i += 1
    assert i == 10


@pytest.mark.asyncio
async def test_stream_stream(server_client):
    class ExampleServicer(ServicerBase):
        async def rpc_powers(self, request: AsyncIterable[test_pb2.TestRequest],
                             _: P2PContext) -> AsyncIterable[test_pb2.TestResponse]:
            async for item in request:
                yield test_pb2.TestResponse(number=item.number ** 2)
                yield test_pb2.TestResponse(number=item.number ** 3)

    server, client = server_client
    servicer = ExampleServicer()
    await servicer.add_p2p_handlers(server)
    stub = servicer.get_stub(client, server.id)

    async def generate_requests() -> AsyncIterable[test_pb2.TestRequest]:
        for i in range(10):
            yield test_pb2.TestRequest(number=i)

    i = 0
    async for item in stub.rpc_powers(generate_requests()):
        if i % 2 == 0:
            assert item == test_pb2.TestResponse(number=(i // 2) ** 2)
        else:
            assert item == test_pb2.TestResponse(number=(i // 2) ** 3)
        i += 1
