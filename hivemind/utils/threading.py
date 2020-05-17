import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, TimeoutError
import time
from typing import Optional, List

num_threads = os.environ.get("HIVEMIND_THREADS", float('inf'))
GLOBAL_EXECUTOR = ThreadPoolExecutor(max_workers=int(num_threads if not isinstance(num_threads, float)))


def run_in_background(func: callable, *args, **kwargs) -> Future:
    """ run func(*args, **kwargs) in background and return Future for its outputs """

    return GLOBAL_EXECUTOR.submit(func, *args, **kwargs)


def run_forever(func: callable, *args, **kwargs):
    """ A function that runs a :func: in background forever. Returns a future that catches exceptions """

    def repeat():
        while True:
            func(*args, **kwargs)

    return run_in_background(repeat)


def run_and_await_k(jobs: List[callable], k: int,
                    timeout_after_k: Optional[float] = 0, timeout_total: Optional[float] = None):
    """
    Runs all :jobs: asynchronously, awaits for at least k of them to finish
    :param jobs: functions to call asynchronously
    :param k: how many functions should finish for call to be successful
    :param timeout_after_k: after reaching k finished jobs, wait for this long before cancelling
    :param timeout_total: if specified, terminate cancel jobs after this many seconds
    :returns: a list of either results or exceptions for each job
    """
    jobs = list(jobs)
    assert k <= len(jobs), f"Can't await {k} out of {len(jobs)} jobs."
    start_time = time.time()
    future_to_ix = {run_in_background(job): i for i, job in enumerate(jobs)}
    outputs = [None] * len(jobs)
    success_count = 0

    try:
        # await first k futures for as long as it takes
        for future in as_completed(list(future_to_ix.keys()), timeout=timeout_total):
            success_count += int(not future.exception())
            outputs[future_to_ix.pop(future)] = future.result() if not future.exception() else future.exception()
            if success_count >= k:
                break  # we have enough futures to succeed
            if len(outputs) + len(future_to_ix) < k:
                failed = len(jobs) - len(outputs) - len(future_to_ix)
                raise ValueError(f"Couldn't get enough results: too many jobs failed ({failed} / {len(outputs)})")

        # await stragglers for at most self.timeout_after_k_min or whatever time is left
        if timeout_after_k is not None and timeout_total is not None:
            time_left = min(timeout_after_k, timeout_total - time.time() + start_time)
        else:
            time_left = timeout_after_k if timeout_after_k is not None else timeout_total
        for future in as_completed(list(future_to_ix.keys()), timeout=time_left):
            success_count += int(not future.exception())
            outputs[future_to_ix.pop(future)] = future.result() if not future.exception() else future.exception()

    except TimeoutError:
        if len(outputs) < k:
            raise TimeoutError(f"Couldn't get enough results: time limit exceeded (got {len(outputs)} of {k})")
    finally:
        for future, index in future_to_ix.items():
            future.cancel()
            outputs[index] = future.result() if not future.exception() else future.exception()
    return outputs
