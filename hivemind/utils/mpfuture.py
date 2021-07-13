from __future__ import annotations

import asyncio
import concurrent.futures._base as base
from contextlib import nullcontext
import multiprocessing as mp
import multiprocessing.connection
import os
import threading
import uuid
from weakref import ref
from enum import Enum, auto
from typing import Generic, TypeVar, Dict, Optional, Any, Callable, Type, Tuple

from hivemind.utils.logging import get_logger


logger = get_logger(__name__)

# flavour types
ResultType = TypeVar("ResultType")
PID, UID, State, PipeEnd = int, int, str, mp.connection.Connection
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


class MessageType(Enum):
    RESULT = auto()
    EXCEPTION = auto()
    RUNNING = auto()
    CANCEL = auto()
    STATE_REQUEST = auto()
    STATE_RESPONSE = auto()


class MPFuture(base.Future, Generic[ResultType]):
    """
    A version of concurrent.futures.Future / asyncio.Future that can be fulfilled from a separate process.
    Any process can access future status and set the result / exception and check for state.
    However, only the original process (i.e. the process that created the future) can await the result or exception.

    :param synchronize: TODO
    :param use_lock: if True, operations with MPFuture use a global lock to prevent concurrent writes to the same pipe;
      If set to False, writing to this future ignores global lock, slightly improving performance, but making user
      responsible for avoiding concurrent set_result / set_exception calls to futures with the same process of origin.
    :param loop: if specified, overrides default asyncio event loop for the purpose of awaiting MPFuture

    :note: This is an internal primitive that is not guaranteed to work outside of hivemind applications.
     More specifically, there are two known limitations:
       - MPFuture works between processes created through inheritance (e.g. fork), *not* for independent processes
       - MPFuture is deterministic if only one process can call set_result/set_exception/set_running_or_notify_cancel
         and only the origin process can call result/exception/cancel.
    """

    _initialization_lock = mp.Lock()  # global lock that prevents simultaneous initialization of two processes
    _update_lock = mp.Lock()  # global lock that prevents simultaneous writing to the same pipe
    _process_wide_pipe: Optional[PipeEnd] = None  # a pipe that is used to send results/exceptions to this process
    _pipe_waiter_thread: Optional[threading.Thread] = None  # process-specific thread that receives results/exceptions
    _active_futures: Optional[Dict[UID, Type[ref][MPFuture]]] = None  # non-done futures originated from this process
    _status_requests: Optional[Dict[UID, Tuple[MPFuture, threading.Event]]] = None  # futures to be updated by origin
    _active_pid: Optional[PID] = None  # pid of currently active process; used to handle forks natively

    SOFT_UPDATE_TIMEOUT = 0.1  # seconds spent awaiting status update before warning is printed
    HARD_UPDATE_TIMEOUT = 10.0  # seconds spent awaiting status update before future is automatically cancelled

    def __init__(self, synchronize: bool = True, use_lock: bool = True, loop: Optional[asyncio.BaseEventLoop] = None):
        base.Future.__init__(self)
        self.synchronize = synchronize
        self._origin_pid, self._uid = os.getpid(), uuid.uuid4().int
        self._state, self._result, self._exception = base.PENDING, None, None
        self._use_lock = use_lock

        self._initialize_backend_if_necessary()
        assert self._uid not in MPFuture._active_futures
        MPFuture._active_futures[self._uid] = ref(self)
        self._sender_pipe = MPFuture._process_wide_pipe

        try:
            self._loop = loop or asyncio.get_event_loop()
            self._aio_event = asyncio.Event()
        except RuntimeError:
            self._loop, self._aio_event = None, None

    def _set_event_if_necessary(self):
        if self._aio_event is None or self._aio_event.is_set():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        async def _event_setter():
            self._aio_event.set()

        if self._loop.is_running() and loop == self.get_loop():
            asyncio.create_task(_event_setter())
        elif self._loop.is_running() and loop != self.get_loop():
            asyncio.run_coroutine_threadsafe(_event_setter(), self._loop)
        else:
            self._loop.run_until_complete(_event_setter())

    @classmethod
    def _initialize_backend_if_necessary(cls):
        pid = os.getpid()
        if MPFuture._active_pid != pid:
            with MPFuture._initialization_lock:
                if MPFuture._active_pid != pid:
                    # note: the second if is intentional, see https://en.wikipedia.org/wiki/Double-checked_locking
                    logger.debug(f"Initializing MPFuture backend for pid {pid}")
                    receiver_pipe, cls._process_wide_pipe = mp.Pipe(duplex=False)
                    cls._active_pid, cls._active_futures, cls._status_requests = pid, {}, {}
                    cls._pipe_waiter_thread = threading.Thread(
                        target=cls._process_updates_in_background,
                        args=[receiver_pipe],
                        name=f"{__name__}.BACKEND",
                        daemon=True,
                    )
                    cls._pipe_waiter_thread.start()

    @classmethod
    def _process_updates_in_background(cls, receiver_pipe: mp.connection.Connection):
        pid = os.getpid()
        while True:
            try:
                uid, msg_type, payload = receiver_pipe.recv()
                future = None
                future_ref = cls._active_futures.get(uid)
                if future_ref is not None:
                    future = future_ref()

                if msg_type == MessageType.STATE_REQUEST:
                    future_state = None if future is None else future.__getstate__()
                    payload.send((uid, MessageType.STATE_RESPONSE, future_state))

                elif msg_type == MessageType.STATE_RESPONSE:
                    future, state_updated_event = cls._status_requests.get(uid) or (None, None)
                    if future is None:
                        logger.debug("Received a state update for a future that does not await status update.")
                    else:
                        if payload is not None:
                            future.__setstate__(payload)
                        else:
                            base.Future.cancel(future)
                        state_updated_event.set()

                elif future is None:
                    logger.debug(
                        f"Received {msg_type} for MPFuture uid={uid}, but future is already done or destroyed"
                    )
                elif msg_type == MessageType.RESULT:
                    future.set_result(payload)
                elif msg_type == MessageType.EXCEPTION:
                    future.set_exception(payload)
                elif msg_type == MessageType.RUNNING:
                    try:
                        future.set_running_or_notify_cancel()
                    except (InvalidStateError, RuntimeError) as e:
                        logger.debug(f"could set MPFuture (uid={uid}) to running due to {e}")
                elif msg_type == MessageType.CANCEL:
                    future.cancel()
                else:
                    raise RuntimeError(f"Received unexpected update type {msg_type}")

                if future is None or future.done():
                    cls._active_futures.pop(uid, None)

            except (BrokenPipeError, EOFError, ConnectionError):
                logger.debug(f"Update pipe was was shut down unexpectedly (pid={pid})")
            except Exception as e:
                logger.exception(f"Could not retrieve update: caught {repr(e)} (pid={pid})")

    def _send_update(self, update_type: MessageType, payload: Any = None):
        """This method sends result, exception or cancel to the MPFuture origin."""
        try:
            with MPFuture._update_lock if self._use_lock else nullcontext():
                self._sender_pipe.send((self._uid, update_type, payload))
        except (ConnectionError, BrokenPipeError, EOFError) as e:
            logger.debug(f"No updates were sent: pipe to origin process is no longer operational ({e}).")

    def _synchronize_if_necessary(self):
        if not self.synchronize or os.getpid() == self._origin_pid or self._state in TERMINAL_STATES:
            return

        self._initialize_backend_if_necessary()

        maybe_existing_request = self._status_requests.get(self._uid)
        if maybe_existing_request is not None:
            _, status_updated = maybe_existing_request
            status_updated.wait(MPFuture.HARD_UPDATE_TIMEOUT)
            return

        # otherwise create a new request for synchronization

        try:
            status_updated = threading.Event()
            self._status_requests[self._uid] = (self, status_updated)
            with MPFuture._update_lock if self._use_lock else nullcontext():
                self._sender_pipe.send((self._uid, MessageType.STATE_REQUEST, self._process_wide_pipe))
            status_updated.wait(MPFuture.SOFT_UPDATE_TIMEOUT)
            if not status_updated.is_set():
                logger.warning(f"Status update took over {MPFuture.SOFT_UPDATE_TIMEOUT}, expect performance issues")
                status_updated.wait(MPFuture.HARD_UPDATE_TIMEOUT - MPFuture.SOFT_UPDATE_TIMEOUT)
                if not status_updated.is_set():
                    self.set_exception(
                        TimeoutError(
                            f"Status update took over {MPFuture.HARD_UPDATE_TIMEOUT} seconds, "
                            f"mpfuture is cancelled"
                        )
                    )
                    status_updated.set()  # this triggers any concurrent _synchronize_if_necessary calls to finish
        except (ConnectionError, BrokenPipeError, EOFError) as e:
            logger.error(f"MPFuture was cancelled because sender pipe is broken. Origin process is probably down.")
            if not self.cancel():
                self.set_exception(e)
        finally:
            self._status_requests.pop(self._uid, None)

    def set_result(self, result: ResultType):
        if self._state in TERMINAL_STATES:
            raise InvalidStateError(f"Can't set_result to a future that is {self._state} ({self._uid})")
        elif os.getpid() == self._origin_pid:
            MPFuture._active_futures.pop(self._uid, None)
            self._set_event_if_necessary()
        else:
            self._send_update(MessageType.RESULT, result)
        super().set_result(result)

    def set_exception(self, exception: Optional[BaseException]):
        if self._state in TERMINAL_STATES:
            raise InvalidStateError(f"Can't set_exception to a future that is {self._state} ({self._uid})")
        elif os.getpid() == self._origin_pid:
            MPFuture._active_futures.pop(self._uid, None)
            self._set_event_if_necessary()
        else:
            self._send_update(MessageType.EXCEPTION, exception)
        super().set_exception(exception)

    def cancel(self) -> bool:
        if self._state in [base.RUNNING, base.FINISHED]:
            return False
        elif os.getpid() == self._origin_pid:
            MPFuture._active_futures.pop(self._uid, None)
            self._set_event_if_necessary()
        else:
            self._send_update(MessageType.CANCEL)
        return super().cancel()

    def set_running_or_notify_cancel(self):
        """if synchronize is set to False, this future will ignore any state changes from origin"""
        self._synchronize_if_necessary()
        try:
            is_running = super().set_running_or_notify_cancel()
            if is_running and os.getpid() != self._origin_pid:
                self._send_update(MessageType.RUNNING)
            return is_running
        except RuntimeError as e:
            raise InvalidStateError(str(e))

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
        self._synchronize_if_necessary()
        return self._state in TERMINAL_STATES

    def running(self):
        self._synchronize_if_necessary()
        return self._state == base.RUNNING

    def cancelled(self):
        self._synchronize_if_necessary()
        return self._state == base.CANCELLED

    def add_done_callback(self, callback: Callable[[MPFuture], None]):
        if os.getpid() != self._origin_pid:
            raise RuntimeError("Only the process that created MPFuture can set callbacks")
        return super().add_done_callback(callback)

    def get_loop(self) -> Optional[asyncio.BaseEventLoop]:
        return self._loop

    def __await__(self):
        if not self._aio_event:
            raise RuntimeError("Can't await: MPFuture was created with no event loop")
        yield from self._aio_event.wait().__await__()
        try:
            return super().result(timeout=0)
        except base.CancelledError:
            raise asyncio.CancelledError()

    def __del__(self):
        if getattr(self, "_origin_pid", None) == os.getpid():
            MPFuture._active_futures.pop(self._uid, None)
        if getattr(self, "_aio_event", None):
            self._aio_event.set()

    def __getstate__(self):
        return dict(
            synchronize=self.synchronize,
            _sender_pipe=self._sender_pipe,
            _state=self._state,
            _origin_pid=self._origin_pid,
            _uid=self._uid,
            _use_lock=self._use_lock,
            _result=self._result,
            _exception=self._exception,
        )

    def __setstate__(self, state):
        self.synchronize = state["synchronize"]
        self._sender_pipe = state["_sender_pipe"]
        self._state, self._origin_pid, self._uid = state["_state"], state["_origin_pid"], state["_uid"]
        self._result, self._exception = state["_result"], state["_exception"]
        self._use_lock = state["_use_lock"]

        self._waiters, self._done_callbacks = [], []
        self._condition = threading.Condition()
        self._aio_event, self._loop = None, None
