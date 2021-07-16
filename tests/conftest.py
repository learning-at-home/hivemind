import gc
from contextlib import suppress

import psutil
import pytest

from hivemind.utils import get_logger

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
