import multiprocessing as mp

import pytest
import psutil

from hivemind.utils.mpfuture import MPFuture


@pytest.fixture(autouse=True, scope='session')
def cleanup_after_test():
    """ reset shared memory manager for isolation, terminate any leftover processes after the test is finished """
    old_values = MPFuture._global_mpfuture_senders, MPFuture._active_pid, MPFuture._active_futures
    manager = mp.managers.SyncManager()
    manager.start()
    try:
        MPFuture._global_mpfuture_senders = manager.dict()
        MPFuture._active_pid = MPFuture._active_futures = None

        yield

        for child in psutil.Process().children(recursive=True):
            child.terminate()
        manager.join(1)
        manager.shutdown()
    finally:
        MPFuture._global_mpfuture_senders, MPFuture._active_pid, MPFuture._active_futures = old_values