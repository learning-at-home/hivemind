import asyncio
import concurrent.futures
import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

import hivemind
from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils import BatchTensorDescriptor, DHTExpiration, HeapEntry, MSGPackSerializer, ValueWithExpiration
from hivemind.utils.asyncio import (
    achain,
    aenumerate,
    aiter_with_timeout,
    amap_in_executor,
    anext,
    as_aiter,
    asingle,
    attach_event_on_finished,
    azip,
    cancel_and_wait,
    enter_asynchronously,
)
from hivemind.utils.mpfuture import InvalidStateError
from hivemind.utils.performance_ema import PerformanceEMA


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

    time.sleep(0.1)  # giving enough time for the futures to be destroyed
    assert len(hivemind.MPFuture._active_futures) == 700

    for future in some_fork_futures:
        future.set_running_or_notify_cancel()
    for future in random.sample(some_fork_futures, 200):
        future.set_result(321)

    evt.set()
    for future in main_futures:
        future.cancel()
    time.sleep(0.1)  # giving enough time for the futures to be destroyed
    assert len(hivemind.MPFuture._active_futures) == 0
    p.join()


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

    chunks3 = list(hivemind.utils.split_for_streaming(serialized_tensor_part, 10**9))
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
            # note: we rely on this being RuntimeError in hivemind.averaging.allreduce.AllReduceRunner


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
    async for elem in amap_in_executor(lambda x: x**2, as_aiter(*range(100)), max_prefetch=5):
        assert elem == num_steps**2
        num_steps += 1
    assert num_steps == 100

    ours = [
        elem
        async for elem in amap_in_executor(max, as_aiter(*range(7)), as_aiter(*range(-50, 50, 10)), max_prefetch=1)
    ]
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

    event = asyncio.Event()
    async for i in attach_event_on_finished(iterate_with_delays([0, 0, 0, 0, 0]), event):
        assert not event.is_set()
    assert event.is_set()

    event = asyncio.Event()
    sleepy_aiter = iterate_with_delays([0.1, 0.1, 0.3, 0.1, 0.1])
    with pytest.raises(asyncio.TimeoutError):
        async for _ in attach_event_on_finished(aiter_with_timeout(sleepy_aiter, timeout=0.2), event):
            assert not event.is_set()
    assert event.is_set()


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


@pytest.mark.asyncio
async def test_async_context():
    lock = mp.Lock()

    async def coro1():
        async with enter_asynchronously(lock):
            await asyncio.sleep(0.2)

    async def coro2():
        await asyncio.sleep(0.1)
        async with enter_asynchronously(lock):
            await asyncio.sleep(0.1)

    await asyncio.wait_for(asyncio.gather(coro1(), coro2()), timeout=0.5)
    # running this without enter_asynchronously would deadlock the event loop


@pytest.mark.asyncio
async def test_async_context_flooding():
    """
    test for a possible deadlock when many coroutines await the lock and overwhelm the underlying ThreadPoolExecutor

    Here's how the test below works: suppose that the thread pool has at most N workers;
    If at least N + 1 coroutines await lock1 concurrently, N of them occupy workers and the rest are awaiting workers;
    When the first of N workers acquires lock1, it lets coroutine A inside lock1 and into await sleep(1e-2);
    During that sleep, one of the worker-less coroutines will take up the worker freed by coroutine A.
    Finally, coroutine A finishes sleeping and immediately gets stuck at lock2, because there are no free workers.
    Thus, every single coroutine is either awaiting an already acquired lock, or awaiting for free workers in executor.

    """
    lock1, lock2 = mp.Lock(), mp.Lock()

    async def coro():
        async with enter_asynchronously(lock1):
            await asyncio.sleep(1e-2)
            async with enter_asynchronously(lock2):
                await asyncio.sleep(1e-2)

    num_coros = max(100, mp.cpu_count() * 5 + 1)
    # note: if we deprecate py3.7, this can be reduced to max(33, cpu + 5); see https://bugs.python.org/issue35279
    await asyncio.wait({coro() for _ in range(num_coros)})


def test_batch_tensor_descriptor_msgpack():
    tensor_descr = BatchTensorDescriptor.from_tensor(torch.ones(1, 3, 3, 7))
    tensor_descr_roundtrip = MSGPackSerializer.loads(MSGPackSerializer.dumps(tensor_descr))

    assert (
        tensor_descr.size == tensor_descr_roundtrip.size
        and tensor_descr.dtype == tensor_descr_roundtrip.dtype
        and tensor_descr.layout == tensor_descr_roundtrip.layout
        and tensor_descr.device == tensor_descr_roundtrip.device
        and tensor_descr.requires_grad == tensor_descr_roundtrip.requires_grad
        and tensor_descr.pin_memory == tensor_descr.pin_memory
        and tensor_descr.compression == tensor_descr.compression
    )


@pytest.mark.parametrize("max_workers", [1, 2, 10])
def test_performance_ema_threadsafe(
    max_workers: int,
    interval: float = 0.01,
    num_updates: int = 100,
    alpha: float = 0.05,
    bias_power: float = 0.7,
    tolerance: float = 0.05,
):
    def run_task(ema):
        task_size = random.randint(1, 4)
        with ema.update_threadsafe(task_size):
            time.sleep(task_size * interval * (0.9 + 0.2 * random.random()))
            return task_size

    with ThreadPoolExecutor(max_workers) as pool:
        ema = PerformanceEMA(alpha=alpha)
        start_time = time.perf_counter()
        futures = [pool.submit(run_task, ema) for i in range(num_updates)]
        total_size = sum(future.result() for future in futures)
        end_time = time.perf_counter()
        target = total_size / (end_time - start_time)
        assert ema.samples_per_second >= (1 - tolerance) * target * max_workers ** (bias_power - 1)
        assert ema.samples_per_second <= (1 + tolerance) * target
