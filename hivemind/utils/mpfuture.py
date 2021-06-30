from __future__ import annotations

import asyncio
import concurrent.futures._base as base
import multiprocessing as mp
import multiprocessing.connection
import os
import threading
from typing import Tuple, Generic, TypeVar, Dict, Optional, Any, Callable

import torch

from hivemind.utils.logging import get_logger


logger = get_logger(__name__)

# flavour types
ResultType = TypeVar('ResultType')
PID, UID, State, PipeEnd = int, int, Any, mp.connection.Connection
ALL_STATES = base.PENDING, base.RUNNING, base.FINISHED, base.CANCELLED, base.CANCELLED_AND_NOTIFIED
TERMINAL_STATES = {base.FINISHED, base.CANCELLED, base.CANCELLED_AND_NOTIFIED}

try:
    from concurrent.futures import InvalidStateError
except ImportError:
    class InvalidStateError(Exception):
        """Raised when attempting to change state of a future in a terminal state (e.g. finished)"""


INITIALIZER_LOCK = mp.Lock()
PIPE_WAITER: Optional[threading.Thread] = None
MPFUTURE_PIPES: Dict[PID, Tuple[PipeEnd, PipeEnd]] = mp.Manager().dict()
ACTIVE_FUTURES: Optional[Dict[PID, MPFuture]] = None
ACTIVE_PID: Optional[PID] = None


def _initialize_mpfuture_backend():
    global ACTIVE_PID, ACTIVE_FUTURES, PIPE_WAITER
    pid = os.getpid()
    logger.debug(f"Initializing MPFuture backend for pid {pid}")
    assert pid != ACTIVE_PID and pid not in MPFUTURE_PIPES, "already initialized"

    with INITIALIZER_LOCK:
        ACTIVE_PID, ACTIVE_FUTURES, MPFUTURE_PIPES[pid] = pid, {}, mp.Pipe(duplex=False)
        PIPE_WAITER = threading.Thread(target=_process_updates_in_background, name=f'{__name__}.BACKEND', daemon=True)
        PIPE_WAITER.start()


def _send_update(pid: PID, uid: UID, message_type: State, payload: Any = None):
    pipes = MPFUTURE_PIPES.get(pid)
    if pipes:
        receiver_pipe, sender_pipe = pipes
        sender_pipe.send((uid, message_type, payload))
    else:
        logger.warning(f"Could not update MPFuture(pid={pid}, uid={uid}): unknown pid.")


def _process_updates_in_background():
    pid = os.getpid()
    receiver_pipe, sender_pipe = MPFUTURE_PIPES[pid]
    while True:
        try:
            uid, message_type, payload = receiver_pipe.recv()
            if uid not in ACTIVE_FUTURES:
                logger.debug(f"Ignoring update to future with uid={uid}: the future is no longer active.")
            elif message_type == Exception:
                base.Future.set_exception(ACTIVE_FUTURES[uid], payload)
            elif message_type == base.FINISHED:
                base.Future.set_result(ACTIVE_FUTURES[uid], payload)
            elif message_type == base.CANCELLED:
                base.Future.cancel(ACTIVE_FUTURES[uid])
            else:
                raise ValueError(f"Unexpected message type {message_type}")

        except BrokenPipeError:
            logger.debug(f"MPFuture backend was shut down (pid={pid}).")
        except Exception as e:
            logger.warning(f"Internal error (type={e}, pid={pid}): could not retrieve update for MPFuture.")
            logger.exception(e)


class MPFuture(base.Future, Generic[ResultType]):
    """
    Multiprocessing-aware version of concurrent.futures.Future / asyncio.Future.
    Any process can access future status and set the result / exception. However, only the
    original process (i.e. the process that created the future) can retrieve the result or exception.

    This primitive works between processes created through inheritance (e.g. fork), *not* for arbitrary processes.
    For independently spawned processes, please instead use mp.Pipe / mp.connection.Connection.
    """

    def __init__(self, loop: Optional[asyncio.BaseEventLoop] = None):
        self._shared_state_code = torch.empty([], dtype=torch.uint8).share_memory_()
        self._state, self._result, self._exception = base.PENDING, None, None
        self._origin_pid, self._uid = os.getpid(), id(self)
        # note: self._uid is only unique inside process that spawned it
        super().__init__()
        if ACTIVE_PID != self._origin_pid:
            _initialize_mpfuture_backend()
        assert self._uid not in ACTIVE_FUTURES
        ACTIVE_FUTURES[self._uid] = self

        try:
            self._loop = loop or asyncio.get_event_loop()
            self._aio_event = asyncio.Event()
        except RuntimeError:
            self._loop = self._aio_event = None

    @property
    def _state(self) -> State:
        return ALL_STATES[self._shared_state_code.item()]

    @_state.setter
    def _state(self, new_state):
        self._shared_state_code[...] = ALL_STATES.index(new_state)
        if self._state in TERMINAL_STATES and self._aio_event is not None:
            asyncio.run_coroutine_threadsafe(self._set_event(), self.get_loop())

    async def _set_event(self):
        self._aio_event.set()

    def set_result(self, result: ResultType):
        if os.getpid() == self._origin_pid:
            super().set_result(result)
        elif self._state in TERMINAL_STATES:
            raise InvalidStateError(f"Can't set_result to a future that is {self._state} ({self})")
        else:
            _send_update(self._origin_pid, self._uid, base.FINISHED, result)

    def set_exception(self, exception: BaseException):
        if os.getpid() == self._origin_pid:
            super().set_exception(exception)
        elif self._state in TERMINAL_STATES:
            raise InvalidStateError(f"Can't set_exception to a future that is {self._state} ({self})")
        else:
            _send_update(self._origin_pid, self._uid, Exception, exception)

    def set_running_or_notify_cancel(self):
        if self._state == base.PENDING:
            self._state = base.RUNNING
            return True
        elif self._state == base.CANCELLED:
            return False
        else:
            raise InvalidStateError(f"Can't set_running_or_notify_cancel when future is in {self._state} ({self})")

    def cancel(self):
        if os.getpid() == self._origin_pid:
            return super().cancel()
        elif self._state in TERMINAL_STATES:
            return False
        else:
            _send_update(self._origin_pid, self._uid, base.CANCELLED)
            return True

    def result(self, timeout: Optional[float] = None) -> ResultType:
        if self._state not in TERMINAL_STATES:
            assert os.getpid() == self._origin_pid, "only the process that created MPFuture can await result."
            return super().result(timeout)
        elif self._state == base.CANCELLED:
            raise base.CancelledError()
        elif self._exception:
            raise self._exception
        else:
            return self._result

    def exception(self, timeout: Optional[float] = None) -> BaseException:
        if self._state not in TERMINAL_STATES:
            assert os.getpid() == self._origin_pid, "only the process that created MPFuture can await exception."
            return super().exception(timeout)
        elif self._state == base.CANCELLED:
            raise base.CancelledError()
        return self._exception

    def done(self) -> bool:
        return self._state in TERMINAL_STATES

    def running(self):
        return self._state == base.RUNNING

    def cancelled(self):
        return self._state == base.CANCELLED

    def add_done_callback(self, callback: Callable):
        assert os.getpid() == self._origin_pid, "only the process that created MPFuture can set callbacks."
        return super().add_done_callback(callback)

    def remove_done_callback(self, callback: Callable):
        assert os.getpid() == self._origin_pid, "only the process that created MPFuture can set callbacks."
        return super().add_done_callback(callback)

    def get_loop(self) -> Optional[asyncio.BaseEventLoop]:
        return self._loop

    def __await__(self):
        if not self._aio_event:
            raise RuntimeError("Can't await: MPFuture was created with no event loop.")
        yield from self._aio_event.wait().__await__()
        try:
            return super().result(timeout=0)
        except base.CancelledError:
            raise asyncio.CancelledError()

    def __del__(self):
        if getattr(self, '_origin_pid', None) == os.getpid():
            del ACTIVE_FUTURES[self._uid]
        if getattr(self, '_aio_event', None):
            self._aio_event.set()

    def __getstate__(self):
        return dict(_shared_state_code=self._shared_state_code,
                    _origin_pid=self._origin_pid, _uid=self._uid,
                    _result=self._result, _exception=self._exception)

    def __setstate__(self, state):
        self._shared_state_code = state['_shared_state_code']
        self._origin_pid, self._uid = state['_origin_pid'], state['_uid']
        self._result, self._exception = state['_result'], state['_exception']
        self._waiters, self._done_callbacks = [], []
        self._condition = threading.Condition()
        self._aio_event = self._loop = None
