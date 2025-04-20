from __future__ import annotations

import asyncio
import concurrent.futures._base as base
import multiprocessing as mp
import os
import threading
import uuid
from concurrent.futures import InvalidStateError
from contextlib import nullcontext
from enum import Enum, auto
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Dict, Generic, Optional, TypeVar
from weakref import ref

from hivemind.utils.logging import get_logger

logger = get_logger(__name__)

# flavour types
ResultType = TypeVar("ResultType")
PID, UID, State, PipeEnd = int, int, str, mp.connection.Connection
ALL_STATES = base.PENDING, base.RUNNING, base.FINISHED, base.CANCELLED, base.CANCELLED_AND_NOTIFIED
TERMINAL_STATES = {base.FINISHED, base.CANCELLED, base.CANCELLED_AND_NOTIFIED}


class SharedStateCode:
    """
    A wrapper around a shared memory byte that is used to represent state.
    """

    def __init__(self, name=None):
        # track parameter was added in Python 3.13, so we don't use it for compatibility
        self._shm = SharedMemory(name=name, create=name is None, size=1)
        self.name = self._shm.name

    def item(self):
        return self._shm.buf[0]

    def __setitem__(self, _, value):
        self._shm.buf[0] = value

    def close(self):
        self._shm.close()

    def unlink(self):
        try:
            self._shm.unlink()
        except Exception as e:
            logger.debug(f"Failed to unlink shared memory: {e}")


class DummyStateCode:
    """
    A non-shared version of SharedStateCode that can be used as a fallback
    when shared memory is not accessible.
    """

    def __init__(self):
        self.name = None
        self._state = ALL_STATES.index(base.PENDING)

    def item(self):
        return self._state

    def __setitem__(self, _, value):
        pass

    def close(self):
        pass

    def unlink(self):
        pass


class UpdateType(Enum):
    RESULT = auto()
    EXCEPTION = auto()
    CANCEL = auto()


class MPFuture(base.Future, Generic[ResultType]):
    """
    A version of concurrent.futures.Future / asyncio.Future that can be fulfilled from a separate process.
    Any process can access future status and set the result / exception and check for state.
    However, only the original process (i.e. the process that created the future) can await the result or exception.

    :param use_lock: if True, operations with MPFuture use a global lock to prevent concurrent writes to the same pipe;
      If set to False, writing to this future ignores global lock, slightly improving performance, but making user
      responsible for avoiding concurrent set_result / set_exception calls to futures with the same process of origin.

    :note: This is an internal primitive that is not guaranteed to work outside of hivemind applications.
     More specifically, there are two known limitations:
       - MPFuture works between processes created through inheritance (e.g. fork), *not* for independent processes
       - MPFuture is deterministic if only one process can call set_result/set_exception/set_running_or_notify_cancel
         and only the origin process can call result/exception/cancel.
    """

    _initialization_lock = mp.Lock()  # global lock that prevents simultaneous initialization of two processes
    _update_lock = mp.Lock()  # global lock that prevents simultaneous writing to the same pipe
    _state_lock = mp.Lock()  # lock for creating shared state objects
    _global_sender_pipe: Optional[PipeEnd] = None  # a pipe that is used to send results/exceptions to this process
    _pipe_waiter_thread: Optional[threading.Thread] = None  # process-specific thread that receives results/exceptions
    _active_futures: Optional[Dict[UID, ref[MPFuture]]] = None  # non-done futures originated from this process
    _active_pid: Optional[PID] = None  # pid of currently active process; used to handle forks natively

    def __init__(self, *, use_lock: bool = True):
        self._maybe_initialize_mpfuture_backend()

        self._origin_pid, self._uid = os.getpid(), uuid.uuid4().int

        with MPFuture._state_lock:
            self._shared_state_code = SharedStateCode()

        self._state_cache: Dict[State, State] = {}
        # mapping from global to cached local future used that makes updates immediately
        # available on setter side; dictionary-based cache works because future can visit any state at most once

        base.Future.__init__(self)  # parent init is deferred because it uses self._shared_state_code
        self._state, self._result, self._exception = base.PENDING, None, None
        self._use_lock = use_lock

        assert self._uid not in MPFuture._active_futures
        MPFuture._active_futures[self._uid] = ref(self)
        self._sender_pipe = MPFuture._global_sender_pipe

        try:
            self._loop = asyncio.get_event_loop()
            self._aio_event = asyncio.Event()
        except RuntimeError:
            self._loop, self._aio_event = None, None

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
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        async def _event_setter():
            self._aio_event.set()

        if self._loop.is_closed():
            return  # do nothing, the loop is already closed
        elif self._loop.is_running() and running_loop == self._loop:
            asyncio.create_task(_event_setter())
        elif self._loop.is_running() and running_loop != self._loop:
            asyncio.run_coroutine_threadsafe(_event_setter(), self._loop)
        else:
            self._loop.run_until_complete(_event_setter())

    @classmethod
    def _maybe_initialize_mpfuture_backend(cls):
        pid = os.getpid()
        if pid != MPFuture._active_pid:
            with MPFuture._initialization_lock:
                if pid != MPFuture._active_pid:
                    # note: the second if is intentional, see https://en.wikipedia.org/wiki/Double-checked_locking
                    logger.debug(f"Initializing MPFuture backend for pid {pid}")

                    receiver_pipe, cls._global_sender_pipe = mp.Pipe(duplex=False)
                    cls._active_pid, cls._active_futures = pid, {}
                    cls._pipe_waiter_thread = threading.Thread(
                        target=cls._process_updates_in_background,
                        args=[receiver_pipe],
                        name=f"{__name__}.BACKEND",
                        daemon=True,
                    )
                    cls._pipe_waiter_thread.start()

    @staticmethod
    def reset_backend():
        """Last-resort function to reset internals of MPFuture. All current MPFuture instances will be broken"""
        MPFuture._active_pid = None
        MPFuture._initialization_lock = mp.Lock()
        MPFuture._update_lock = mp.Lock()
        MPFuture._state_lock = mp.Lock()

    @classmethod
    def _process_updates_in_background(cls, receiver_pipe: mp.connection.Connection):
        pid = os.getpid()
        while True:
            try:
                if cls._pipe_waiter_thread is not threading.current_thread():
                    break  # backend was reset, a new background thread has started

                uid, update_type, payload = receiver_pipe.recv()
                future = None
                future_ref = cls._active_futures.pop(uid, None)
                if future_ref is not None:
                    future = future_ref()

                if future is None:
                    # The MPFuture instance is already destroyed in this process
                    # (the caller is not interested in the result)
                    continue
                if update_type == UpdateType.RESULT:
                    future.set_result(payload)
                elif update_type == UpdateType.EXCEPTION:
                    future.set_exception(payload)
                elif update_type == UpdateType.CANCEL:
                    future.cancel()
                else:
                    raise RuntimeError(f"Received unexpected update type {update_type}")
            except (BrokenPipeError, EOFError, ConnectionError):
                logger.debug(f"Update pipe was was shut down unexpectedly (pid={pid})")
            except Exception as e:
                logger.exception(f"Could not retrieve update: caught {repr(e)} (pid={pid})")

    def _send_update(self, update_type: UpdateType, payload: Any = None):
        """This method sends result, exception or cancel to the MPFuture origin."""
        try:
            with MPFuture._update_lock if self._use_lock else nullcontext():
                self._sender_pipe.send((self._uid, update_type, payload))
        except (ConnectionError, BrokenPipeError, EOFError, OSError) as e:
            logger.debug(f"No updates were sent: pipe to origin process was broken ({e})", exc_info=True)

    def set_result(self, result: ResultType):
        if os.getpid() == self._origin_pid:
            super().set_result(result)
            MPFuture._active_futures.pop(self._uid, None)
        elif self._state in TERMINAL_STATES:
            raise InvalidStateError(f"Can't set_result to a future that is {self._state} ({self._uid})")
        else:
            self._state_cache[self._state], self._result = base.FINISHED, result
            self._send_update(UpdateType.RESULT, result)

    def set_exception(self, exception: Optional[BaseException]):
        if os.getpid() == self._origin_pid:
            super().set_exception(exception)
            MPFuture._active_futures.pop(self._uid, None)
        elif self._state in TERMINAL_STATES:
            raise InvalidStateError(f"Can't set_exception to a future that is {self._state} ({self._uid})")
        else:
            self._state_cache[self._state], self._exception = base.FINISHED, exception
            self._send_update(UpdateType.EXCEPTION, exception)

    def cancel(self) -> bool:
        if os.getpid() == self._origin_pid:
            MPFuture._active_futures.pop(self._uid, None)
            return super().cancel()
        elif self._state in [base.RUNNING, base.FINISHED]:
            return False
        else:
            self._state_cache[self._state] = base.CANCELLED
            self._send_update(UpdateType.CANCEL)
            return True

    def set_running_or_notify_cancel(self):
        if self._state == base.PENDING:
            self._state = base.RUNNING
            return True
        elif self._state == base.CANCELLED:
            return False
        else:
            raise InvalidStateError(
                f"Can't set_running_or_notify_cancel when future is in {self._state} ({self._uid})"
            )

    def result(self, timeout: Optional[float] = None) -> ResultType:
        if self._state not in TERMINAL_STATES:
            if os.getpid() != self._origin_pid:
                raise RuntimeError("Only the process that created MPFuture can await result")
            return super().result(timeout)
        elif self._state == base.CANCELLED:
            raise base.CancelledError()
        elif self._exception:
            raise self._exception
        else:
            return self._result

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        if self._state not in TERMINAL_STATES:
            if os.getpid() != self._origin_pid:
                raise RuntimeError("Only the process that created MPFuture can await exception")
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

    def add_done_callback(self, callback: Callable[[MPFuture], None]):
        if os.getpid() != self._origin_pid:
            raise RuntimeError("Only the process that created MPFuture can set callbacks")
        return super().add_done_callback(callback)

    def __await__(self):
        if not self._aio_event:
            raise RuntimeError("Can't await: MPFuture was created with no event loop")
        yield from self._aio_event.wait().__await__()
        try:
            return super().result()
        except base.CancelledError:
            raise asyncio.CancelledError()

    def __del__(self):
        if getattr(self, "_origin_pid", None) == os.getpid() and MPFuture._active_futures is not None:
            MPFuture._active_futures.pop(self._uid, None)
        if getattr(self, "_aio_event", None):
            self._aio_event.set()
        if hasattr(self, "_shared_state_code"):
            try:
                self._shared_state_code.close()
                if getattr(self, "_origin_pid", None) == os.getpid():
                    self._shared_state_code.unlink()
            except Exception as e:
                logger.debug(f"Error closing shared memory: {e}")

    def __getstate__(self):
        return dict(
            _sender_pipe=self._sender_pipe,
            _shared_state_code_name=self._shared_state_code.name,
            _origin_pid=self._origin_pid,
            _uid=self._uid,
            _use_lock=self._use_lock,
            _result=self._result,
            _exception=self._exception,
        )

    def __setstate__(self, state):
        self._sender_pipe = state["_sender_pipe"]
        try:
            self._shared_state_code = SharedStateCode(name=state["_shared_state_code_name"])
        except Exception as e:
            # If the origin process garbage-collects all instances of MPFuture using the same shmem segment,
            # the segment is freed, and we'll get an error when trying to attach to it. In this case,
            # we create a dummy state code that doesn't use shared memory.
            logger.debug(f"Failed to attach to shared memory: {e}. Creating a dummy state code.")
            self._shared_state_code = DummyStateCode()

        self._origin_pid, self._uid = state["_origin_pid"], state["_uid"]
        self._result, self._exception = state["_result"], state["_exception"]
        self._use_lock = state["_use_lock"]

        self._waiters, self._done_callbacks = [], []
        self._condition = threading.Condition()
        self._aio_event, self._loop = None, None
        self._state_cache = {}
