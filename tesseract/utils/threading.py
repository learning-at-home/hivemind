import time
from concurrent.futures import Future, TimeoutError
from itertools import count
from threading import Thread, Event, Lock


def run_in_background(func: callable, *args, **kwargs):
    """ run f(*args, **kwargs) in background and return Future for its outputs """
    future = Future()

    def _run():
        try:
            future.set_result(func(*args, **kwargs))
        except Exception as e:
            future.set_exception(e)

    Thread(target=_run).start()
    return future


def repeated(func: callable, n_times=None):
    """ A function that runs a :func: forever or for a specified number of times; use with run_run_in_background """

    def repeat():
        for i in count():
            if n_times is not None and i > n_times:
                break
            func()

    return repeat


def add_event_callback(event: Event, callback, timeout=None):
    """ Add callback that will be executed asynchronously when event is set """
    return Thread(target=lambda: (event.wait(timeout), callback())).start()


class CountdownEvent(Event):
    def __init__(self, count_to: int, initial=0):
        """ An event that must be incremented :count_to: times before it is considered set """
        super().__init__()
        self.value = initial
        self.count_to = count_to
        self.lock = Lock()
        self.increment(by=0)  # trigger set/unset depending on initial value

    def increment(self, by=1):
        with self.lock:
            self.value += by
            if self.value >= self.count_to:
                super().set()
            else:
                super().clear()
            return self.value

    def clear(self):
        return self.increment(by=-self.value)


def await_first(*events: Event, k=1, timeout=None):
    """
    wait until first k (default=1) events are set, return True if event was set fast
    # Note: after k successes we manually *set* all events to avoid memory leak.
    """
    events_done = CountdownEvent(count_to=k)
    for event in events:
        add_event_callback(event, callback=events_done.increment, timeout=timeout)

    if events_done.wait(timeout=timeout):
        [event.set() for event in events]
        return True
    else:
        raise TimeoutError()


def run_and_await_k(jobs: callable, k, timeout_after_k=0, timeout_total=None):
    """
    Runs all :jobs: asynchronously, awaits for at least k of them to finish
    :param jobs: functions to call
    :param k: how many functions should finish
    :param timeout_after_k: after reaching k finished jobs, wait for this long before cancelling
    :param timeout_total: if specified, terminate cancel jobs after this many seconds
    :returns: a list of either results or exceptions for each job
    """
    assert k <= len(jobs)
    start_time = time.time()
    min_successful_jobs = CountdownEvent(count_to=k)
    max_failed_jobs = CountdownEvent(count_to=len(jobs) - k + 1)

    def _run_and_increment(run_job: callable):
        try:
            result = run_job()
            min_successful_jobs.increment()
            return result
        except Exception as e:
            max_failed_jobs.increment()
            return e

    def _run_and_await(run_job: callable):
        # call function asynchronously. Increment counter after finished
        future = run_in_background(_run_and_increment, run_job)

        try:  # await for success counter to reach k OR for fail counter to reach n - k + 1
            await_first(min_successful_jobs, max_failed_jobs,
                        timeout=None if timeout_total is None else timeout_total - time.time() + start_time)
        except TimeoutError as e:  # counter didn't reach k jobs in timeout_total
            return future.result() if future.done() else e

        try:  # await for subsequent jobs if asked to
            return future.result(timeout=timeout_after_k)
        except TimeoutError as e:
            future.cancel()
            return e

        except Exception as e:  # job failed with exception. Ignore it.
            return e

    results = [run_in_background(_run_and_await, f) for f in jobs]
    results = [result.result() for result in results]
    if min_successful_jobs.is_set():
        return results
    elif max_failed_jobs.is_set():
        raise ValueError("Could not get enough results: too many jobs failed.")
    else:
        raise TimeoutError("Could not get enough results: reached timeout_total.")
