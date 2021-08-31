import asyncio
import concurrent.futures
import multiprocessing as mp
import random
import time

import numpy as np
import pytest
import torch

import hivemind
from hivemind.proto.dht_pb2_grpc import DHTStub
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.proto.runtime_pb2_grpc import ConnectionHandlerStub
from hivemind.utils import DHTExpiration, HeapEntry, MSGPackSerializer, ValueWithExpiration
from hivemind.utils.asyncio import (
    achain,
    aenumerate,
    afirst,
    as_aiter,
    amap_in_executor,
    anext,
    asingle,
    azip,
    cancel_and_wait, aiter_with_timeout,
)
from hivemind.utils.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.utils.mpfuture import InvalidStateError


@pytest.mark.forked
def test_mpfuture_result():
    future = hivemind.MPFuture()

    def _proc(future):
        with pytest.raises(RuntimeError):
            future.result()  # only creator process can await result

        future.set_result(321)

    p = mp.Process(target=_proc, args=(future,))
    p.start()
    p.join()

    assert future.result() == 321
    assert future.exception() is None
    assert future.cancel() is False
    assert future.done() and not future.running() and not future.cancelled()

    future = hivemind.MPFuture()
    with pytest.raises(concurrent.futures.TimeoutError):
        future.result(timeout=1e-3)

    future.set_result(["abacaba", 123])
    assert future.result() == ["abacaba", 123]


@pytest.mark.forked
def test_mpfuture_exception():
    future = hivemind.MPFuture()
    with pytest.raises(concurrent.futures.TimeoutError):
        future.exception(timeout=1e-3)

    def _proc(future):
        future.set_exception(NotImplementedError())

    p = mp.Process(target=_proc, args=(future,))
    p.start()
    p.join()

    assert isinstance(future.exception(), NotImplementedError)
    with pytest.raises(NotImplementedError):
        future.result()
    assert future.cancel() is False
    assert future.done() and not future.running() and not future.cancelled()


@pytest.mark.forked
def test_mpfuture_cancel():
    future = hivemind.MPFuture()
    assert not future.cancelled()
    future.cancel()
    evt = mp.Event()

    def _proc():
        with pytest.raises(concurrent.futures.CancelledError):
            future.result()
        with pytest.raises(concurrent.futures.CancelledError):
            future.exception()
        with pytest.raises(InvalidStateError):
            future.set_result(123)
        with pytest.raises(InvalidStateError):
            future.set_exception(NotImplementedError())
        assert future.cancelled() and future.done() and not future.running()
        evt.set()

    p = mp.Process(target=_proc)
    p.start()
    p.join()
    assert evt.is_set()


@pytest.mark.forked
def test_mpfuture_status():
    evt = mp.Event()
    future = hivemind.MPFuture()

    def _proc1(future):
        assert future.set_running_or_notify_cancel() is True
        evt.set()

    p = mp.Process(target=_proc1, args=(future,))
    p.start()
    p.join()
    assert evt.is_set()
    evt.clear()

    assert future.running() and not future.done() and not future.cancelled()
    with pytest.raises(InvalidStateError):
        future.set_running_or_notify_cancel()

    future = hivemind.MPFuture()
    assert future.cancel()

    def _proc2(future):
        assert not future.running() and future.done() and future.cancelled()
        assert future.set_running_or_notify_cancel() is False
        evt.set()

    p = mp.Process(target=_proc2, args=(future,))
    p.start()
    p.join()
    evt.set()

    future2 = hivemind.MPFuture()
    future2.cancel()
    assert future2.set_running_or_notify_cancel() is False


@pytest.mark.asyncio
async def test_await_mpfuture():
    # await result from the same process, but a different coroutine
    f1, f2 = hivemind.MPFuture(), hivemind.MPFuture()

    async def wait_and_assign_async():
        assert f2.set_running_or_notify_cancel() is True
        await asyncio.sleep(0.1)
        f1.set_result((123, "ololo"))
        f2.set_result((456, "pyshpysh"))

    asyncio.create_task(wait_and_assign_async())

    assert (await asyncio.gather(f1, f2)) == [(123, "ololo"), (456, "pyshpysh")]

    # await result from separate processes
    f1, f2 = hivemind.MPFuture(), hivemind.MPFuture()

    def wait_and_assign(future, value):
        time.sleep(0.1 * random.random())
        future.set_result(value)

    p1 = mp.Process(target=wait_and_assign, args=(f1, "abc"))
    p2 = mp.Process(target=wait_and_assign, args=(f2, "def"))
    for p in p1, p2:
        p.start()

    assert (await asyncio.gather(f1, f2)) == ["abc", "def"]
    for p in p1, p2:
        p.join()

    # await cancel
    f1, f2 = hivemind.MPFuture(), hivemind.MPFuture()

    def wait_and_cancel():
        time.sleep(0.01)
        f2.set_result(123456)
        time.sleep(0.1)
        f1.cancel()

    p = mp.Process(target=wait_and_cancel)
    p.start()

    with pytest.raises(asyncio.CancelledError):
        # note: it is intended that MPFuture raises Cancel
        await asyncio.gather(f1, f2)

    p.join()

    # await exception
    f1, f2 = hivemind.MPFuture(), hivemind.MPFuture()

    def wait_and_raise():
        time.sleep(0.01)
        f2.set_result(123456)
        time.sleep(0.1)
        f1.set_exception(ValueError("we messed up"))

    p = mp.Process(target=wait_and_raise)
    p.start()

    with pytest.raises(ValueError):
        # note: it is intended that MPFuture raises Cancel
        await asyncio.gather(f1, f2)

    p.join()


@pytest.mark.forked
def test_mpfuture_bidirectional():
    evt = mp.Event()
    future_from_main = hivemind.MPFuture()

    def _future_creator():
        future_from_fork = hivemind.MPFuture()
        future_from_main.set_result(("abc", future_from_fork))

        if future_from_fork.result() == ["we", "need", "to", "go", "deeper"]:
            evt.set()

    p = mp.Process(target=_future_creator)
    p.start()

    out = future_from_main.result()
    assert isinstance(out[1], hivemind.MPFuture)
    out[1].set_result(["we", "need", "to", "go", "deeper"])

    p.join()
    assert evt.is_set()


@pytest.mark.forked
def test_mpfuture_done_callback():
    receiver, sender = mp.Pipe(duplex=False)
    events = [mp.Event() for _ in range(6)]

    def _future_creator():
        future1, future2, future3 = hivemind.MPFuture(), hivemind.MPFuture(), hivemind.MPFuture()

        def _check_result_and_set(future):
            assert future.done()
            assert future.result() == 123
            events[0].set()

        future1.add_done_callback(_check_result_and_set)
        future1.add_done_callback(lambda future: events[1].set())
        future2.add_done_callback(lambda future: events[2].set())
        future3.add_done_callback(lambda future: events[3].set())

        sender.send((future1, future2))
        future2.cancel()  # trigger future2 callback from the same process

        events[0].wait()
        future1.add_done_callback(
            lambda future: events[4].set()
        )  # schedule callback after future1 is already finished
        events[5].wait()

    p = mp.Process(target=_future_creator)
    p.start()

    future1, future2 = receiver.recv()
    future1.set_result(123)

    with pytest.raises(RuntimeError):
        future1.add_done_callback(lambda future: (1, 2, 3))

    assert future1.done() and not future1.cancelled()
    assert future2.done() and future2.cancelled()
    for i in 0, 1, 4:
        events[i].wait(1)
    assert events[0].is_set() and events[1].is_set() and events[2].is_set() and events[4].is_set()
    assert not events[3].is_set()

    events[5].set()
    p.join()


@pytest.mark.forked
def test_many_futures():
    evt = mp.Event()
    receiver, sender = mp.Pipe()
    main_futures = [hivemind.MPFuture() for _ in range(1000)]
    assert len(hivemind.MPFuture._active_futures) == 1000

    def _run_peer():
        fork_futures = [hivemind.MPFuture() for _ in range(500)]
        assert len(hivemind.MPFuture._active_futures) == 500

        for i, future in enumerate(random.sample(main_futures, 300)):
            if random.random() < 0.5:
                future.set_result(i)
            else:
                future.set_exception(ValueError(f"{i}"))

        sender.send(fork_futures[:-100])
        for future in fork_futures[-100:]:
            future.cancel()

        evt.wait()

        assert len(hivemind.MPFuture._active_futures) == 200
        for future in fork_futures:
            if not future.done():
                future.set_result(123)
        assert len(hivemind.MPFuture._active_futures) == 0

    p = mp.Process(target=_run_peer)
    p.start()

    some_fork_futures = receiver.recv()
    assert len(hivemind.MPFuture._active_futures) == 700

    for future in some_fork_futures:
        future.set_running_or_notify_cancel()
    for future in random.sample(some_fork_futures, 200):
        future.set_result(321)

    evt.set()
    for future in main_futures:
        future.cancel()
    assert len(hivemind.MPFuture._active_futures) == 0
    p.join()


def test_tensor_compression(size=(128, 128, 64), alpha=5e-08, beta=0.0008):
    torch.manual_seed(0)
    X = torch.randn(*size)
    assert torch.allclose(deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.NONE)), X)
    error = deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.MEANSTD_16BIT)) - X
    assert error.square().mean() < alpha
    error = deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.FLOAT16)) - X
    assert error.square().mean() < alpha
    error = deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.QUANTILE_8BIT)) - X
    assert error.square().mean() < beta
    error = deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.UNIFORM_8BIT)) - X
    assert error.square().mean() < beta

    zeros = torch.zeros(5, 5)
    for compression_type in CompressionType.values():
        assert deserialize_torch_tensor(serialize_torch_tensor(zeros, compression_type)).isfinite().all()


@pytest.mark.forked
@pytest.mark.asyncio
async def test_channel_cache():
    hivemind.ChannelCache.MAXIMUM_CHANNELS = 3
    hivemind.ChannelCache.EVICTION_PERIOD_SECONDS = 0.1

    c1 = hivemind.ChannelCache.get_stub("localhost:1337", DHTStub, aio=False)
    c2 = hivemind.ChannelCache.get_stub("localhost:1337", DHTStub, aio=True)
    c3 = hivemind.ChannelCache.get_stub("localhost:1338", DHTStub, aio=False)
    c3_again = hivemind.ChannelCache.get_stub("localhost:1338", DHTStub, aio=False)
    c1_again = hivemind.ChannelCache.get_stub("localhost:1337", DHTStub, aio=False)
    c4 = hivemind.ChannelCache.get_stub("localhost:1339", DHTStub, aio=True)
    c2_anew = hivemind.ChannelCache.get_stub("localhost:1337", DHTStub, aio=True)
    c1_yetagain = hivemind.ChannelCache.get_stub("localhost:1337", DHTStub, aio=False)

    await asyncio.sleep(0.2)
    c1_anew = hivemind.ChannelCache.get_stub(target="localhost:1337", aio=False, stub_type=DHTStub)
    c1_anew_again = hivemind.ChannelCache.get_stub(target="localhost:1337", aio=False, stub_type=DHTStub)
    c1_otherstub = hivemind.ChannelCache.get_stub(target="localhost:1337", aio=False, stub_type=ConnectionHandlerStub)
    await asyncio.sleep(0.05)
    c1_otherstub_again = hivemind.ChannelCache.get_stub(
        target="localhost:1337", aio=False, stub_type=ConnectionHandlerStub
    )
    all_channels = [c1, c2, c3, c4, c3_again, c1_again, c2_anew, c1_yetagain, c1_anew, c1_anew_again, c1_otherstub]

    assert all(isinstance(c, DHTStub) for c in all_channels[:-1])
    assert isinstance(all_channels[-1], ConnectionHandlerStub)
    assert "aio" in repr(c2.rpc_find)
    assert "aio" not in repr(c1.rpc_find)

    duplicates = {
        (c1, c1_again),
        (c1, c1_yetagain),
        (c1_again, c1_yetagain),
        (c3, c3_again),
        (c1_anew, c1_anew_again),
        (c1_otherstub, c1_otherstub_again),
    }
    for i in range(len(all_channels)):
        for j in range(i + 1, len(all_channels)):
            ci, cj = all_channels[i], all_channels[j]
            assert (ci is cj) == ((ci, cj) in duplicates), (i, j)


def test_serialize_tensor():
    tensor = torch.randn(512, 12288)

    serialized_tensor = serialize_torch_tensor(tensor, CompressionType.NONE)
    for chunk_size in [1024, 64 * 1024, 64 * 1024 + 1, 10 ** 9]:
        chunks = list(hivemind.split_for_streaming(serialized_tensor, chunk_size))
        assert len(chunks) == (len(serialized_tensor.buffer) - 1) // chunk_size + 1
        restored = hivemind.combine_from_streaming(chunks)
        assert torch.allclose(deserialize_torch_tensor(restored), tensor)

    chunk_size = 30 * 1024
    serialized_tensor = serialize_torch_tensor(tensor, CompressionType.FLOAT16)
    chunks = list(hivemind.split_for_streaming(serialized_tensor, chunk_size))
    assert len(chunks) == (len(serialized_tensor.buffer) - 1) // chunk_size + 1
    restored = hivemind.combine_from_streaming(chunks)
    assert torch.allclose(deserialize_torch_tensor(restored), tensor, rtol=0, atol=1e-2)

    tensor = torch.randint(0, 100, (512, 1, 1))
    serialized_tensor = serialize_torch_tensor(tensor, CompressionType.NONE)
    chunks = list(hivemind.split_for_streaming(serialized_tensor, chunk_size))
    assert len(chunks) == (len(serialized_tensor.buffer) - 1) // chunk_size + 1
    restored = hivemind.combine_from_streaming(chunks)
    assert torch.allclose(deserialize_torch_tensor(restored), tensor)

    scalar = torch.tensor(1.0)
    serialized_scalar = serialize_torch_tensor(scalar, CompressionType.NONE)
    assert torch.allclose(deserialize_torch_tensor(serialized_scalar), scalar)

    serialized_scalar = serialize_torch_tensor(scalar, CompressionType.FLOAT16)
    assert torch.allclose(deserialize_torch_tensor(serialized_scalar), scalar)


def test_serialize_tuple():
    test_pairs = (
        ((1, 2, 3), [1, 2, 3]),
        (("1", False, 0), ["1", False, 0]),
        (("1", False, 0), ("1", 0, 0)),
        (("1", b"qq", (2, 5, "0")), ["1", b"qq", (2, 5, "0")]),
    )

    for first, second in test_pairs:
        assert MSGPackSerializer.loads(MSGPackSerializer.dumps(first)) == first
        assert MSGPackSerializer.loads(MSGPackSerializer.dumps(second)) == second
        assert MSGPackSerializer.dumps(first) != MSGPackSerializer.dumps(second)


def test_split_parts():
    tensor = torch.randn(910, 512)
    serialized_tensor_part = serialize_torch_tensor(tensor, allow_inplace=False)
    chunks1 = list(hivemind.utils.split_for_streaming(serialized_tensor_part, 16384))
    assert len(chunks1) == int(np.ceil(tensor.numel() * tensor.element_size() / 16384))

    chunks2 = list(hivemind.utils.split_for_streaming(serialized_tensor_part, 10_000))
    assert len(chunks2) == int(np.ceil(tensor.numel() * tensor.element_size() / 10_000))

    chunks3 = list(hivemind.utils.split_for_streaming(serialized_tensor_part, 10 ** 9))
    assert len(chunks3) == 1

    compressed_tensor_part = serialize_torch_tensor(tensor, CompressionType.FLOAT16, allow_inplace=False)
    chunks4 = list(hivemind.utils.split_for_streaming(compressed_tensor_part, 16384))
    assert len(chunks4) == int(np.ceil(tensor.numel() * 2 / 16384))

    combined1 = hivemind.utils.combine_from_streaming(chunks1)
    combined2 = hivemind.utils.combine_from_streaming(iter(chunks2))
    combined3 = hivemind.utils.combine_from_streaming(chunks3)
    combined4 = hivemind.utils.combine_from_streaming(chunks4)
    for combined in combined1, combined2, combined3:
        assert torch.allclose(tensor, deserialize_torch_tensor(combined), rtol=1e-5, atol=1e-8)

    assert torch.allclose(tensor, deserialize_torch_tensor(combined4), rtol=1e-3, atol=1e-3)

    combined_incomplete = hivemind.utils.combine_from_streaming(chunks4[:5])
    combined_incomplete2 = hivemind.utils.combine_from_streaming(chunks4[:1])
    combined_incomplete3 = hivemind.utils.combine_from_streaming(chunks4[:-1])
    for combined in combined_incomplete, combined_incomplete2, combined_incomplete3:
        with pytest.raises(RuntimeError):
            deserialize_torch_tensor(combined)
            # note: we rely on this being RuntimeError in hivemind.averaging.allreduce.AllreduceRunner


def test_generic_data_classes():
    value_with_exp = ValueWithExpiration(value="string_value", expiration_time=DHTExpiration(10))
    assert value_with_exp.value == "string_value" and value_with_exp.expiration_time == DHTExpiration(10)

    heap_entry = HeapEntry(expiration_time=DHTExpiration(10), key="string_value")
    assert heap_entry.key == "string_value" and heap_entry.expiration_time == DHTExpiration(10)

    sorted_expirations = sorted([DHTExpiration(value) for value in range(1, 1000)])
    sorted_heap_entries = sorted([HeapEntry(DHTExpiration(value), key="any") for value in range(1, 1000)[::-1]])
    assert all([entry.expiration_time == value for entry, value in zip(sorted_heap_entries, sorted_expirations)])


@pytest.mark.asyncio
async def test_asyncio_utils():
    res = [i async for i, item in aenumerate(as_aiter("a", "b", "c"))]
    assert res == list(range(len(res)))

    num_steps = 0
    async for elem in amap_in_executor(lambda x: x ** 2, as_aiter(*range(100)), max_prefetch=5):
        assert elem == num_steps ** 2
        num_steps += 1
    assert num_steps == 100

    ours = [elem async for elem in amap_in_executor(max, as_aiter(*range(7)), as_aiter(*range(-50, 50, 10)), max_prefetch=1)]
    ref = list(map(max, range(7), range(-50, 50, 10)))
    assert ours == ref

    ours = [row async for row in azip(as_aiter("a", "b", "c"), as_aiter(1, 2, 3))]
    ref = list(zip(["a", "b", "c"], [1, 2, 3]))
    assert ours == ref

    async def _aiterate():
        yield "foo"
        yield "bar"
        yield "baz"

    iterator = _aiterate()
    assert (await anext(iterator)) == "foo"
    tail = [item async for item in iterator]
    assert tail == ["bar", "baz"]
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)

    assert [item async for item in achain(_aiterate(), as_aiter(*range(5)))] == ["foo", "bar", "baz"] + list(range(5))

    assert await asingle(as_aiter(1)) == 1
    with pytest.raises(ValueError):
        await asingle(as_aiter())
    with pytest.raises(ValueError):
        await asingle(as_aiter(1, 2, 3))

    assert await afirst(as_aiter(1)) == 1
    assert await afirst(as_aiter()) is None
    assert await afirst(as_aiter(), -1) == -1
    assert await afirst(as_aiter(1, 2, 3)) == 1

    async def iterate_with_delays(delays):
        for i, delay in enumerate(delays):
            await asyncio.sleep(delay)
            yield i

    async for _ in aiter_with_timeout(iterate_with_delays([0.1] * 5), timeout=0.2):
        pass

    sleepy_aiter = iterate_with_delays([0.1, 0.1, 0.3, 0.1, 0.1])
    num_steps = 0
    with pytest.raises(asyncio.TimeoutError):
        async for _ in aiter_with_timeout(sleepy_aiter, timeout=0.2):
            num_steps += 1

    assert num_steps == 2


@pytest.mark.asyncio
async def test_cancel_and_wait():
    finished_gracefully = False

    async def coro_with_finalizer():
        nonlocal finished_gracefully

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            finished_gracefully = True
            raise

    task = asyncio.create_task(coro_with_finalizer())
    await asyncio.sleep(0.05)
    assert await cancel_and_wait(task)
    assert finished_gracefully

    async def coro_with_result():
        return 777

    async def coro_with_error():
        raise ValueError("error")

    task_with_result = asyncio.create_task(coro_with_result())
    task_with_error = asyncio.create_task(coro_with_error())
    await asyncio.sleep(0.05)
    assert not await cancel_and_wait(task_with_result)
    assert not await cancel_and_wait(task_with_error)
