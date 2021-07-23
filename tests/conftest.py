import gc
from contextlib import suppress

import psutil
import pytest

from hivemind.utils.logging import get_logger
from hivemind.utils.mpfuture import MPFuture


logger = get_logger(__name__)


@pytest.fixture(autouse=True, scope="session")
def cleanup_children():
    yield

    gc.collect()  # Call .__del__() for removed objects

    children = psutil.Process().children(recursive=True)
    if children:
        logger.info(f"Cleaning up {len(children)} leftover child processes")
        for child in children:
            with suppress(psutil.NoSuchProcess):
                child.terminate()
        psutil.wait_procs(children, timeout=1)
        for child in children:
            with suppress(psutil.NoSuchProcess):
                child.kill()

    # Broken code or killing of child processes may leave the MPFuture backend corrupted
    MPFuture.reset_backend()
