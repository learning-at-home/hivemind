from typing import TypeVar, AsyncIterator, Union, AsyncIterable
import asyncio
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


async def achain(*async_iters: AsyncIterable[T]) -> AsyncIterator[T]:
    """ equivalent to chain(iter1, iter2, ...) for asynchronous iterators. """
    for aiter in async_iters:
        async for elem in aiter:
            yield elem
