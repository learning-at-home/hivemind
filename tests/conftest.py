import asyncio
import gc

import psutil
import pytest

from hivemind.utils.crypto import RSAPrivateKey
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.utils.mpfuture import MPFuture

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


@pytest.fixture
def event_loop():
    """
    This overrides the ``event_loop`` fixture from pytest-asyncio
    (e.g. to make it compatible with ``asyncio.subprocess``).

    This fixture is identical to the original one but does not call ``loop.close()`` in the end.
    Indeed, at this point, the loop is already stopped (i.e. next tests are free to create new loops).
    However, finalizers of objects created in the current test may reference the current loop and fail if it is closed.
    For example, this happens while using ``asyncio.subprocess`` (the ``asyncio.subprocess.Process`` finalizer
    fails if the loop is closed, but works if the loop is only stopped).
    """

    yield asyncio.get_event_loop()


@pytest.fixture(autouse=True, scope="session")
def cleanup_children():
    yield

    with RSAPrivateKey._process_wide_key_lock:
        RSAPrivateKey._process_wide_key = None

    gc.collect()  # Call .__del__() for removed objects

    children = psutil.Process().children(recursive=True)
    if children:
        gone, alive = psutil.wait_procs(children, timeout=0.1)
        logger.debug(f"Cleaning up {len(alive)} leftover child processes")
        for child in alive:
            child.terminate()
        gone, alive = psutil.wait_procs(alive, timeout=1)
        for child in alive:
            child.kill()

    MPFuture.reset_backend()
