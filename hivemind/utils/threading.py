import os
from concurrent.futures import Future, ThreadPoolExecutor

EXECUTOR_PID, GLOBAL_EXECUTOR = None, None


def run_in_background(func: callable, *args, **kwargs) -> Future:
    """ run func(*args, **kwargs) in background and return Future for its outputs """
    global EXECUTOR_PID, GLOBAL_EXECUTOR
    if os.getpid() != EXECUTOR_PID:
        GLOBAL_EXECUTOR = ThreadPoolExecutor(max_workers=os.environ.get("HIVEMIND_THREADS", float('inf')))
        EXECUTOR_PID = os.getpid()
    return GLOBAL_EXECUTOR.submit(func, *args, **kwargs)
