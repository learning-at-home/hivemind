from __future__ import annotations
import time
import multiprocessing as mp
import multiprocessing.connection
import concurrent.futures._base as base

import asyncio
from functools import lru_cache
from typing import Optional, Tuple

from hivemind.utils.threading import run_in_background


class MPFuture(base.Future):
    """ Multiprocessing version of concurrent.futures.Future. Can also be awaited like asyncio.Future """

    TERMINAL_STATES = {base.FINISHED, base.CANCELLED, base.CANCELLED_AND_NOTIFIED}

    def __init__(self, connection: mp.connection.Connection):
        """ manually create MPFuture. Please use MPFuture.make_pair instead """
        self._state, self._result, self._exception = base.PENDING, None, None
        self.connection = connection

    @classmethod
    def make_pair(cls) -> Tuple[MPFuture, MPFuture]:
        """ Create a pair of linked futures to be used in two processes """
        connection1, connection2 = mp.Pipe()
        return cls(connection1), cls(connection2)

    def _send_updates(self):
        """ Send updates to a paired MPFuture """
        try:
            self.connection.send((self._state, self._result, self._exception))
            if self._state in self.TERMINAL_STATES:
                self._shutdown_trigger.set_result(True)
                self.connection.close()
            return True
        except BrokenPipeError:
            return False

    def _recv_updates(self, timeout: Optional[float]):
        """ Await updates from a paired MPFuture """
        try:
            future = base.wait([run_in_background(self.connection.poll, timeout), self._shutdown_trigger],
                               return_when=base.FIRST_COMPLETED)[0].pop()
            if future is self._shutdown_trigger:
                raise BrokenPipeError()
            if not future.result():
                raise TimeoutError()
            self._state, result, exception = self.connection.recv()
            self._result = result if result is not None else self._result
            self._exception = exception if exception is not None else self._exception
            if self._state in self.TERMINAL_STATES:
                self.connection.close()
        except TimeoutError as e:
            raise e
        except (BrokenPipeError, OSError, EOFError) as e:
            if self._state in (base.PENDING, base.RUNNING):
                self._state, self._exception = base.FINISHED, e

    def _await_terminal_state(self, timeout: Optional[float]):
        """ Await updates until future is either finished, cancelled or got an exception """
        time_left = float('inf') if timeout is None else timeout
        time_before = time.monotonic()
        while self._state not in self.TERMINAL_STATES and time_left > 0:
            self._recv_updates(time_left if timeout else None)
            time_spent = time.monotonic() - time_before
            time_left, time_before = time_left - time_spent, time_before + time_spent

    def _sync_updates(self):
        """ Apply queued updates from a paired MPFuture without waiting for new ones """
        try:
            self._recv_updates(timeout=0)
        except TimeoutError:
            pass

    def set_result(self, result):
        self._sync_updates()
        if self._state in self.TERMINAL_STATES:
            raise RuntimeError(f"Can't set_result to a future that is in {self._state}")
        self._state, self._result = base.FINISHED, result
        return self._send_updates()

    def set_exception(self, exception: BaseException):
        self._sync_updates()
        if self._state in self.TERMINAL_STATES:
            raise RuntimeError(f"Can't set_exception to a future that is in {self._state}")
        self._state, self._exception = base.FINISHED, exception
        self._send_updates()

    def set_running_or_notify_cancel(self):
        self._sync_updates()
        if self._state == base.PENDING:
            self._state = base.RUNNING
            return self._send_updates()
        elif self._state == base.CANCELLED:
            return False
        else:
            raise RuntimeError(f"Can't set_running_or_notify_cancel to a future that is in {self._state}")

    def cancel(self):
        self._sync_updates()
        if self._state in self.TERMINAL_STATES:
            return False
        self._state, self._exception = base.CANCELLED, base.CancelledError()
        return self._send_updates()

    def result(self, timeout: Optional[float] = None):
        self._await_terminal_state(timeout)
        if self._exception is not None:
            raise self._exception
        return self._result

    def exception(self, timeout=None):
        self._await_terminal_state(timeout)
        if self._state == base.CANCELLED:
            raise base.CancelledError()
        return self._exception

    def done(self) -> bool:
        self._sync_updates()
        return self._state in self.TERMINAL_STATES

    def running(self):
        self._sync_updates()
        return self._state == base.RUNNING

    def cancelled(self):
        self._sync_updates()
        return self._state == base.CANCELLED

    def add_done_callback(self, callback):
        raise NotImplementedError(f"MPFuture doesn't support callbacks.")

    def remove_done_callback(self, callback):
        raise NotImplementedError(f"MPFuture doesn't support callbacks.")

    def get_loop(self):
        raise NotImplementedError(f"MPFuture doesn't support get_loop")

    @property
    @lru_cache()
    def _shutdown_trigger(self):
        return base.Future()

    def __repr__(self):
        self._sync_updates()
        if self._state == base.FINISHED:
            if self._exception:
                return "<MPFuture at 0x{:x} state=finished raised {}>".format(id(self), type(self._exception))
            else:
                return "<MPFuture at 0x{:x} state=finished returned {}>".format(id(self), type(self._result))
        else:
            return "<MPFuture at 0x{:x} state={}>".format(id(self), self._state)

    def __await__(self):
        yield from asyncio.get_running_loop().run_in_executor(None, self._await_terminal_state, None).__await__()
        if self._exception:
            raise self._exception
        return self._result

    def __del__(self):
        self._shutdown_trigger.set_result(True)
        if hasattr(self, 'connection'):
            self.connection.close()
