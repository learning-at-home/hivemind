import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterable, AsyncIterator, Awaitable, Callable, Optional, Tuple, TypeVar, Union

import uvloop

from hivemind.utils.logging import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


def switch_to_uvloop() -> asyncio.AbstractEventLoop:
    """stop any running event loops; install uvloop; then create, set and return a new event loop"""
    try:
        asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop
    except RuntimeError as error_no_event_loop:
        pass  # this allows running DHT from background threads with no event loop
    uvloop.install()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


async def anext(aiter: AsyncIterator[T]) -> Union[T, StopAsyncIteration]:
    """equivalent to next(iter) for asynchronous iterators. Modifies aiter in-place!"""
    return await aiter.__anext__()


async def aiter(*args: T) -> AsyncIterator[T]:
    """create an asynchronous iterator from a sequence of values"""
    for arg in args:
        yield arg


async def azip(*iterables: AsyncIterable[T]) -> AsyncIterator[Tuple[T, ...]]:
    """equivalent of zip for asynchronous iterables"""
    iterators = [iterable.__aiter__() for iterable in iterables]
    while True:
        try:
            yield tuple(await asyncio.gather(*(itr.__anext__() for itr in iterators)))
        except StopAsyncIteration:
            break


async def achain(*async_iters: AsyncIterable[T]) -> AsyncIterator[T]:
    """equivalent to chain(iter1, iter2, ...) for asynchronous iterators."""
    for aiter in async_iters:
        async for elem in aiter:
            yield elem


async def aenumerate(aiterable: AsyncIterable[T]) -> AsyncIterable[Tuple[int, T]]:
    """equivalent to enumerate(iter) for asynchronous iterators."""
    index = 0
    async for elem in aiterable:
        yield index, elem
        index += 1


async def asingle(aiter: AsyncIterable[T]) -> T:
    """If ``aiter`` has exactly one item, returns this item. Otherwise, raises ``ValueError``."""
    count = 0
    async for item in aiter:
        count += 1
        if count == 2:
            raise ValueError("asingle() expected an iterable with exactly one item, but got two or more items")
    if count == 0:
        raise ValueError("asingle() expected an iterable with exactly one item, but got an empty iterable")
    return item


async def afirst(aiter: AsyncIterable[T], default: Optional[T] = None) -> Optional[T]:
    """Returns the first item of ``aiter`` or ``default`` if ``aiter`` is empty."""
    async for item in aiter:
        return item
    return default


async def await_cancelled(awaitable: Awaitable) -> bool:
    try:
        await awaitable
        return False
    except (asyncio.CancelledError, concurrent.futures.CancelledError):
        # In Python 3.7, awaiting a cancelled asyncio.Future raises concurrent.futures.CancelledError
        # instead of asyncio.CancelledError
        return True
    except BaseException:
        logger.exception(f"Exception in {awaitable}:")
        return False


async def cancel_and_wait(awaitable: Awaitable) -> bool:
    """
    Cancels ``awaitable`` and waits for its cancellation.
    In case of ``asyncio.Task``, helps to avoid ``Task was destroyed but it is pending!`` errors.
    In case of ``asyncio.Future``, equal to ``future.cancel()``.
    """

    awaitable.cancel()
    return await await_cancelled(awaitable)


async def amap_in_executor(
    func: Callable[..., T],
    *iterables: AsyncIterable,
    max_prefetch: Optional[int] = None,
    executor: Optional[ThreadPoolExecutor] = None,
) -> AsyncIterator[T]:
    """iterate from an async iterable in a background thread, yield results to async iterable"""
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue(max_prefetch)

    async def _put_items():
        async for args in azip(*iterables):
            await queue.put(loop.run_in_executor(executor, func, *args))
        await queue.put(None)

    task = asyncio.create_task(_put_items())
    try:
        future = await queue.get()
        while future is not None:
            yield await future
            future = await queue.get()
        await task
    finally:
        if not task.done():
            task.cancel()
