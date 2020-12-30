import asyncio
import uvloop


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
