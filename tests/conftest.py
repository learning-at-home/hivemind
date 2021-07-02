import multiprocessing as mp

import pytest
import psutil

from hivemind.utils.mpfuture import MPFuture


@pytest.fixture(autouse=True, scope='session')
def cleanup_after_test():
    old_values = MPFuture.global_mpfuture_senders, MPFuture.active_pid, MPFuture.active_futures
    try:
        with mp.managers.SyncManager() as manager:
            MPFuture.global_mpfuture_senders = manager.dict()
            MPFuture.active_pid = MPFuture.active_futures = None

            yield

        for child in psutil.Process().children(recursive=True):
            child.terminate()
    finally:
        MPFuture.global_mpfuture_senders, MPFuture.active_pid, MPFuture.active_futures = old_values