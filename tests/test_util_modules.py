import asyncio
import torch

import pytest
import hivemind
from hivemind.proto.dht_pb2_grpc import DHTStub
from hivemind.proto.runtime_pb2_grpc import ConnectionHandlerStub
from concurrent.futures import CancelledError


def test_mpfuture_result():
    f1, f2 = hivemind.MPFuture.make_pair()
    f1.set_result(321)
    assert f2.result() == 321
    assert f1.result() == 321

    for future in [f1, f2]:
        with pytest.raises(RuntimeError):
            future.set_result(123)
        with pytest.raises(RuntimeError):
            future.set_exception(ValueError())
        assert future.cancel() is False
        assert future.done() and not future.running() and not future.cancelled()

    f1, f2 = hivemind.MPFuture.make_pair()
    with pytest.raises(TimeoutError):
        f1.result(timeout=1e-3)

    f2.set_result(['abacaba', 123])
    assert f1.result() == ['abacaba', 123]


def test_mpfuture_exception():
    f1, f2 = hivemind.MPFuture.make_pair()
    with pytest.raises(TimeoutError):
        f1.exception(timeout=1e-3)

    f2.set_exception(NotImplementedError())

    for future in [f1, f2]:
        assert isinstance(future.exception(), NotImplementedError)
        with pytest.raises(NotImplementedError):
            future.result()
        assert future.cancel() is False
        assert future.done() and not future.running() and not future.cancelled()


def test_mpfuture_cancel():
    f1, f2 = hivemind.MPFuture.make_pair()
    assert not f2.cancelled()
    f1.cancel()
    for future in [f1, f2]:
        with pytest.raises(CancelledError):
            future.result()
        with pytest.raises(CancelledError):
            future.exception()
        with pytest.raises(RuntimeError):
            future.set_result(123)
        with pytest.raises(RuntimeError):
            future.set_exception(NotImplementedError)
        assert future.cancelled() and future.done() and not future.running()


def test_mpfuture_status():
    f1, f2 = hivemind.MPFuture.make_pair()
    assert f1.set_running_or_notify_cancel() is True
    for future in [f1, f2]:
        assert future.running() and not future.done() and not future.cancelled()
        with pytest.raises(RuntimeError):
            future.set_running_or_notify_cancel()
    f2.cancel()
    for future in [f1, f2]:
        assert not future.running() and future.done() and future.cancelled()
        assert future.set_running_or_notify_cancel() is False

    f1, f2 = hivemind.MPFuture.make_pair()
    f1.cancel()
    for future in [f1, f2]:
        assert future.set_running_or_notify_cancel() is False


@pytest.mark.asyncio
async def test_await_mpfuture():
    # await result
    f1, f2 = hivemind.MPFuture.make_pair()

    async def wait_and_assign():
        assert f2.set_running_or_notify_cancel() is True
        await asyncio.sleep(0.1)
        f2.set_result((123, 'ololo'))

    asyncio.create_task(wait_and_assign())
    for future in [f1, f2]:
        res = await future
        assert res == (123, 'ololo')

    # await cancel
    f1, f2 = hivemind.MPFuture.make_pair()

    async def wait_and_cancel():
        await asyncio.sleep(0.1)
        f1.cancel()

    asyncio.create_task(wait_and_cancel())
    for future in [f1, f2]:
        with pytest.raises(CancelledError):
            await future

    # await exception
    f1, f2 = hivemind.MPFuture.make_pair()

    async def wait_and_raise():
        await asyncio.sleep(0.1)
        f1.set_exception(SystemError())

    asyncio.create_task(wait_and_raise())
    for future in [f1, f2]:
        with pytest.raises(SystemError):
            await future


def test_vector_compression(size=(128, 128, 64), alpha=5e-08):
    torch.manual_seed(0)
    from hivemind.proto.runtime_pb2 import CompressionType
    from hivemind.utils import serialize_torch_tensor, deserialize_torch_tensor
    X = torch.randn(*size)
    assert torch.allclose(deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.NONE)), X)
    error = deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.MEANSTD_LAST_AXIS_FLOAT16))-X
    assert error.square().mean() < alpha
    error = deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.FLOAT16)) - X
    assert error.square().mean() < alpha


@pytest.mark.forked
@pytest.mark.asyncio
async def test_channel_cache():
    hivemind.ChannelCache.MAXIMUM_CHANNELS = 3
    hivemind.ChannelCache.EVICTION_PERIOD_SECONDS = 0.1

    c1 = hivemind.ChannelCache.get_stub('localhost:1337', DHTStub, aio=False)
    c2 = hivemind.ChannelCache.get_stub('localhost:1337', DHTStub, aio=True)
    c3 = hivemind.ChannelCache.get_stub('localhost:1338', DHTStub, aio=False)
    c3_again = hivemind.ChannelCache.get_stub('localhost:1338', DHTStub, aio=False)
    c1_again = hivemind.ChannelCache.get_stub('localhost:1337', DHTStub, aio=False)
    c4 = hivemind.ChannelCache.get_stub('localhost:1339', DHTStub, aio=True)
    c2_anew = hivemind.ChannelCache.get_stub('localhost:1337', DHTStub, aio=True)
    c1_yetagain = hivemind.ChannelCache.get_stub('localhost:1337', DHTStub, aio=False)

    await asyncio.sleep(0.2)
    c1_anew = hivemind.ChannelCache.get_stub(target='localhost:1337', aio=False, stub_type=DHTStub)
    c1_anew_again = hivemind.ChannelCache.get_stub(target='localhost:1337', aio=False, stub_type=DHTStub)
    c1_otherstub = hivemind.ChannelCache.get_stub(target='localhost:1337', aio=False, stub_type=ConnectionHandlerStub)
    await asyncio.sleep(0.05)
    c1_otherstub_again = hivemind.ChannelCache.get_stub(target='localhost:1337', aio=False,
                                                        stub_type=ConnectionHandlerStub)
    all_channels = [c1, c2, c3, c4, c3_again, c1_again, c2_anew, c1_yetagain, c1_anew, c1_anew_again, c1_otherstub]

    assert all(isinstance(c, DHTStub) for c in all_channels[:-1])
    assert isinstance(all_channels[-1], ConnectionHandlerStub)
    assert 'aio' in repr(c2.rpc_find)
    assert 'aio' not in repr(c1.rpc_find)

    duplicates = {(c1, c1_again), (c1, c1_yetagain), (c1_again, c1_yetagain), (c3, c3_again),
                  (c1_anew, c1_anew_again), (c1_otherstub, c1_otherstub_again)}
    for i in range(len(all_channels)):
        for j in range(i + 1, len(all_channels)):
            ci, cj = all_channels[i], all_channels[j]
            assert (ci is cj) == ((ci, cj) in duplicates), (i, j)


def test_serialize_tensor():
    tensor = torch.randn(512, 12288)

    serialized_tensor = hivemind.serialize_torch_tensor(tensor, hivemind.CompressionType.NONE)
    for chunk_size in [1024, 64 * 1024, 64 * 1024 + 1, 10 ** 9]:
        chunks = list(hivemind.split_for_streaming(serialized_tensor, chunk_size))
        assert len(chunks) == (len(serialized_tensor.buffer) - 1) // chunk_size + 1
        restored = hivemind.combine_from_streaming(chunks)
        assert torch.allclose(hivemind.deserialize_torch_tensor(restored), tensor)

    chunk_size = 30 * 1024
    serialized_tensor = hivemind.serialize_torch_tensor(tensor, hivemind.CompressionType.FLOAT16)
    chunks = list(hivemind.split_for_streaming(serialized_tensor, chunk_size))
    assert len(chunks) == (len(serialized_tensor.buffer) - 1) // chunk_size + 1
    restored = hivemind.combine_from_streaming(chunks)
    assert torch.allclose(hivemind.deserialize_torch_tensor(restored), tensor, rtol=0, atol=1e-2)

    tensor = torch.randint(0, 100, (512, 1, 1))
    serialized_tensor = hivemind.serialize_torch_tensor(tensor, hivemind.CompressionType.NONE)
    chunks = list(hivemind.split_for_streaming(serialized_tensor, chunk_size))
    assert len(chunks) == (len(serialized_tensor.buffer) - 1) // chunk_size + 1
    restored = hivemind.combine_from_streaming(chunks)
    assert torch.allclose(hivemind.deserialize_torch_tensor(restored), tensor)
