import asyncio
import os
from concurrent.futures import Future
from threading import Thread
from typing import Awaitable

from hivemind.utils import switch_to_uvloop


class RemoteExpertWorker:
    """Local thread for managing async tasks related to RemoteExpert"""

    _event_thread = None
    _event_loop_fut = None
    _pid = None

    @classmethod
    def _run_event_loop(cls):
        try:
            loop = switch_to_uvloop()
            cls._event_loop_fut.set_result(loop)
        except Exception as e:
            cls._event_loop_fut.set_exception(e)
        loop.run_forever()

    @classmethod
    def run_coroutine(cls, coro: Awaitable, return_future: bool = False):
        if cls._event_thread is None or cls._pid != os.getpid():
            cls._pid = os.getpid()
            cls._event_loop_fut = Future()
            cls._event_thread = Thread(target=cls._run_event_loop, daemon=True)
            cls._event_thread.start()

        loop = cls._event_loop_fut.result()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future if return_future else future.result()
