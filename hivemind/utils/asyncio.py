from typing import TypeVar, AsyncIterator, Union, AsyncIterable, Awaitable, Tuple
import asyncio

import janus
import uvloop


T = TypeVar('T')


def switch_to_uvloop() -> asyncio.AbstractEventLoop:
    """ stop any running event loops; install uvloop; then create, set and return a new event loop """
    try:
        asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop
    except RuntimeError as error_no_event_loop:
        pass  # this allows running DHT from background threads with no event loop
    uvloop.install()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


async def anext(aiter: AsyncIterator[T]) -> Union[T, StopAsyncIteration]:
    """ equivalent to next(iter) for asynchronous iterators. Modifies aiter in-place! """
    return await aiter.__anext__()


async def aiter(*args: T) -> AsyncIterator[T]:
    """ create an asynchronous iterator from a sequence of values """
    for arg in args:
        yield arg


async def azip(*iterables: AsyncIterable[T]) -> AsyncIterator[Tuple[T, ...]]:
    """ equivalent of zip for asynchronous iterables """
    iterators = [iterable.__aiter__() for iterable in iterables]
    while True:
        try:
            yield tuple(await asyncio.gather(*(itr.__anext__() for itr in iterators)))
        except StopAsyncIteration:
            break


async def achain(*async_iters: AsyncIterable[T]) -> AsyncIterator[T]:
    """ equivalent to chain(iter1, iter2, ...) for asynchronous iterators. """
    for aiter in async_iters:
        async for elem in aiter:
            yield elem


async def aenumerate(aiterable: AsyncIterable[T]) -> AsyncIterable[Tuple[int, T]]:
    """ equivalent to enumerate(iter) for asynchronous iterators. """
    index = 0
    async for elem in aiterable:
        yield index, elem
        index += 1


async def await_cancelled(awaitable: Awaitable) -> bool:
    try:
        await awaitable
        return False
    except asyncio.CancelledError:
        return True
    except BaseException:
        return False


async def async_map(func: callable, *iterables: AsyncIterable, max_prefetch: int):
    """ iterate from an async iterable in a background thread, yield results to async iterable """
    assert max_prefetch > 0
    inputs, outputs = janus.Queue(max_prefetch), janus.Queue(max_prefetch)

    async def _put_inputs():
        async for args in azip(*iterables):
            await inputs.async_q.put(args)
        await inputs.async_q.put(None)

    def _thread():
        try:
            args = inputs.sync_q.get()
            while args is not None:
                outputs.sync_q.put(func(*args))
                args = inputs.sync_q.get()
            outputs.sync_q.put(None)
        except BaseException as e:
            outputs.sync_q.put(e)

    asyncio.create_task(_put_inputs())
    asyncio.get_event_loop().run_in_executor(None, _thread)

    output = await outputs.async_q.get()
    while output is not None:
        if isinstance(output, Exception):
            raise output
        else:
            yield output
            output = await outputs.async_q.get()


