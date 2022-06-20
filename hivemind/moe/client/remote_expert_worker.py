import os
from concurrent.futures import Future
from queue import Queue
from threading import Thread
from typing import Awaitable, Optional

from hivemind.utils import switch_to_uvloop


class RemoteExpertWorker:
    """Local thread for managing async tasks related to RemoteExpert"""

    _task_queue: Queue = Queue()
    _event_thread: Optional[Thread] = None
    _pid: int = -1

    @classmethod
    def _run(cls):
        loop = switch_to_uvloop()

        async def receive_tasks():
            while True:
                cor, future = cls._task_queue.get()
                try:
                    result = await cor
                except Exception as e:
                    future.set_exception(e)
                    continue
                if not future.cancelled():
                    future.set_result(result)

        loop.run_until_complete(receive_tasks())

    @classmethod
    def run_coroutine(cls, coro: Awaitable, return_future: bool = False):
        if cls._event_thread is None or cls._pid != os.getpid():
            cls._pid = os.getpid()
            cls._event_thread = Thread(target=cls._run, daemon=True)
            cls._event_thread.start()

        future = Future()
        cls._task_queue.put((coro, future))

        if return_future:
            return future

        result = future.result()
        return result
