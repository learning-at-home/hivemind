from __future__ import annotations

import asyncio
import concurrent.futures._base as base
import contextlib
import multiprocessing as mp
import multiprocessing.connection
import os
import threading
import uuid
from enum import Enum, auto
from typing import Generic, TypeVar, Dict, Optional, Any, Callable

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
    # Python 3.7 doesn't raise concurrent.futures.InvalidStateError for repeating set_result/set_exception calls and
    # doesn't even define this error. In this module, we simulate the Python 3.8+ behavior,
    # defining and raising this error if necessary.
    class InvalidStateError(Exception):
        """Raised when attempting to change state of a future in a terminal state (e.g. finished)"""


class UpdateType(Enum):
    RESULT = auto()
    EXCEPTION = auto()
    CANCEL = auto()


class MPFuture(base.Future, Generic[ResultType]):
    """
    A version of concurrent.futures.Future / asyncio.Future that can be fulfilled from a separate process.
    Any process can access future status and set the result / exception and check for state.
    However, only the original process (i.e. the process that created the future) can await the result or exception.

    :note: This is an internal primitive that is not guaranteed to work outside of hivemind applications.
     More specifically, there are two known limitations:
       - MPFuture works between processes created through inheritance (e.g. fork), *not* for independent processes
       - Different executors (non-origin processes) cannot call set_result / set_exception / cancel simultaneously
    """
    lock = mp.Lock()  # global lock that prevents simultaneous initialization and writing
    pipe_waiter_thread: Optional[threading.Thread] = None  # process-specific thread that receives results/exceptions
    global_mpfuture_receivers: Dict[PID, PipeEnd] = mp.Manager().dict()
    active_futures: Optional[Dict[PID, MPFuture]] = None  # pending or running futures originated from current process
    active_pid: Optional[PID] = None  # pid of currently active process; used to handle forks natively

    def __init__(self, use_lock: bool = True,  loop: Optional[asyncio.BaseEventLoop] = None):
        self._origin_pid, self._uid = os.getpid(), uuid.uuid4().int
        self._shared_state_code = torch.empty([], dtype=torch.uint8).share_memory_()
        self._state_cache = {}  # mapping from global to cached local future used that makes updates immediately
        # available on setter side; dictionary-based cache works because future can visit any state at most once

        base.Future.__init__(self)
        self._state, self._result, self._exception = base.PENDING, None, None
        self._use_lock = use_lock

        if self.active_pid != self._origin_pid:
            self._initialize_mpfuture_backend()
        assert self._uid not in self.active_futures
        self.active_futures[self._uid] = self

        try:
            self._loop = loop or asyncio.get_event_loop()
            self._aio_event = asyncio.Event()
        except RuntimeError:
            self._loop = self._aio_event = None

    @property
    def _state(self) -> State:
        shared_state = ALL_STATES[self._shared_state_code.item()]
        return self._state_cache.get(shared_state, shared_state)

    @_state.setter
    def _state(self, new_state: State):
        self._shared_state_code[...] = ALL_STATES.index(new_state)
        if self._state in TERMINAL_STATES and self._loop is not None and not self._aio_event.is_set():
            self._set_event_threadsafe()

    def _set_event_threadsafe(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop == self.get_loop():
            asyncio.create_task(self._event_setter())
        else:
            asyncio.run_coroutine_threadsafe(self._event_setter(), self._loop)

    async def _event_setter(self):
        self._aio_event.set()

    @property
    def _sender_pipe(self) -> mp.connection.Connection:
        """ a pipe that can be used to send updates to the MPFuture creator """
        return self.global_mpfuture_receivers[self._origin_pid]

    @classmethod
    def _initialize_mpfuture_backend(cls):
        pid = os.getpid()
        logger.debug(f"Initializing MPFuture backend for pid {pid}")
        assert pid != cls.active_pid and pid not in cls.global_mpfuture_receivers, "already initialized"

        recv_pipe, send_pipe = mp.Pipe(duplex=False)
        with cls.lock:
            cls.active_pid, cls.active_futures, cls.global_mpfuture_receivers[pid] = pid, {}, send_pipe
            cls.pipe_waiter_thread = threading.Thread(target=cls._process_updates_in_background, args=[recv_pipe],
                                                      name=f'{__name__}.BACKEND', daemon=True)
            cls.pipe_waiter_thread.start()

    @classmethod
    def _process_updates_in_background(cls, receiver_pipe: mp.connection.Connection):
        pid = os.getpid()
        while True:
            try:
                uid, update_type, payload = receiver_pipe.recv()
                if uid not in cls.active_futures:
                    logger.debug(f"Ignoring update to future with uid={uid}: the future is no longer active.")
                elif update_type == UpdateType.RESULT:
                    cls.active_futures.pop(uid).set_result(payload)
                elif update_type == UpdateType.EXCEPTION:
                    cls.active_futures.pop(uid).set_exception(payload)
                elif update_type == UpdateType.CANCEL:
                    cls.active_futures.pop(uid).cancel()
                else:
                    raise RuntimeError(f"MPFuture received unexpected update type {update_type}")
            except (BrokenPipeError, EOFError):
                logger.debug(f"MPFuture backend was shut down (pid={pid}).")
            except Exception as e:
                logger.warning(f"Internal error (type={e}, pid={pid}): could not retrieve update for MPFuture.")
                logger.exception(e)

    def _send_update(self, update_type: UpdateType, payload: Any = None):
        """ this method sends result, exception or cancel to the MPFuture origin. """
        with self.lock if self._use_lock else contextlib.nullcontext():
            self._sender_pipe.send((self._uid, update_type, payload))

    def set_result(self, result: ResultType):
        if os.getpid() == self._origin_pid:
            self.active_futures.pop(self._uid, None)
            super().set_result(result)
        elif self._state in TERMINAL_STATES:
            raise InvalidStateError(f"Can't set_result to a future that is {self._state} ({self._uid})")
        else:
            self._state_cache[self._state], self._result = base.FINISHED, result
            self._send_update(UpdateType.RESULT, result)

    def set_exception(self, exception: BaseException):
        if os.getpid() == self._origin_pid:
            self.active_futures.pop(self._uid, None)
            super().set_exception(exception)
        elif self._state in TERMINAL_STATES:
            raise InvalidStateError(f"Can't set_exception to a future that is {self._state} ({self._uid})")
        else:
            self._state_cache[self._state], self._exception = base.FINISHED, exception
            self._send_update(UpdateType.EXCEPTION, exception)

    def cancel(self):
        if os.getpid() == self._origin_pid:
            self.active_futures.pop(self._uid, None)
            return super().cancel()
        elif self._state in [base.RUNNING, base.FINISHED]:
            return False
        else:
            self._state_cache[self._state] = base.CANCELLED
            self._send_update(UpdateType.CANCEL)

    def set_running_or_notify_cancel(self):
        if self._state == base.PENDING:
            self._state = base.RUNNING
            return True
        elif self._state == base.CANCELLED:
            return False
        else:
            raise InvalidStateError(f"Can't set_running_or_notify_cancel when future is in {self._state} ({self._uid})")

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
            self.active_futures.pop(self._uid, None)
        if getattr(self, '_aio_event', None):
            self._aio_event.set()

    def __getstate__(self):
        return dict(_shared_state_code=self._shared_state_code, _origin_pid=self._origin_pid, _uid=self._uid,
                    _use_lock=self._use_lock, _result=self._result, _exception=self._exception)

    def __setstate__(self, state):
        self._shared_state_code = state['_shared_state_code']
        self._origin_pid, self._uid = state['_origin_pid'], state['_uid']
        self._result, self._exception = state['_result'], state['_exception']
        self._use_lock = state['_use_lock']

        self._waiters, self._done_callbacks = [], []
        self._condition = threading.Condition()
        self._aio_event = self._loop = None
        self._state_cache = {}
