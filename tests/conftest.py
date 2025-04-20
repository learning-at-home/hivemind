import gc

import psutil
import pytest

from hivemind.utils.crypto import RSAPrivateKey
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.utils.mpfuture import MPFuture

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


@pytest.fixture(autouse=True, scope="session")
def cleanup_children():
    yield

    with RSAPrivateKey._process_wide_key_lock:
        RSAPrivateKey._process_wide_key = None

    gc.collect()  # Call .__del__() for removed objects

    MPFuture.reset_backend()

    children = psutil.Process().children(recursive=True)
    if children:
        _gone, alive = psutil.wait_procs(children, timeout=1)
        logger.debug(f"Cleaning up {len(alive)} leftover child processes")
        for child in alive:
            child.terminate()
        _gone, alive = psutil.wait_procs(alive, timeout=1)
        for child in alive:
            child.kill()
