import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError
from itertools import count
from threading import Event
from threading import Lock
from threading import Thread


def run_in_background(func: callable, *args, **kwargs) -> Future:
    """ run f(*args, **kwargs) in background and return Future for its outputs """
    future = Future()

    def _run():
        try:
            future.set_result(func(*args, **kwargs))
        except BaseException as e:
            future.set_exception(e)

    Thread(target=_run).start()
    return future


def run_forever(func: callable, *args, **kwargs):
    """ A function that runs a :func: in background forever. Returns a future that catches exceptions """

    def repeat():
        while True:
            func(*args, **kwargs)

    return run_in_background(repeat)
