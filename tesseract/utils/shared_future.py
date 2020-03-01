import multiprocessing as mp
import multiprocessing.connection
from concurrent.futures import Future, CancelledError
from warnings import warn


class SharedFuture(Future):
    """ Multiprocessing version of concurrent.futures.Future, interacts between two processes via Pipe """
    STATES = 'pending', 'running', 'cancelled', 'finished', 'exception'
    STATE_PENDING, STATE_RUNNING, STATE_CANCELLED, STATE_FINISHED, STATE_EXCEPTION = STATES

    def __init__(self, connection: mp.connection.Connection):
        """ manually create MPFuture. Please use MPFuture.make_pair instead """
        self.connection = connection
        self.state = self.STATE_PENDING
        self._result = None
        self._exception = None

    @classmethod
    def make_pair(cls):
        """ Create a pair of linked futures to be used in two processes """
        connection1, connection2 = mp.Pipe()
        return cls(connection1), cls(connection2)

    def _recv(self, timeout):
        if self.state in (self.STATE_PENDING, self.STATE_RUNNING):
            if not self.connection.poll(timeout):
                raise TimeoutError()
            try:
                status, payload = self.connection.recv()
            except BrokenPipeError as e:
                status, payload = self.STATE_EXCEPTION, e

            assert status in self.STATES
            self.state = status

            if status == self.STATE_FINISHED:
                self._result = payload
            elif status == self.STATE_EXCEPTION:
                self._exception = payload
            elif status in (self.STATE_RUNNING, self.STATE_CANCELLED):
                pass  # only update self.state
            else:
                raise ValueError("Result status should not be self.STATE_PENDING")

    def set_result(self, result):
        try:
            self.state, self._result = self.STATE_FINISHED, result
            self.connection.send((self.STATE_FINISHED, result))
            return True
        except BrokenPipeError:
            return False

    def set_exception(self, exception: BaseException):
        try:
            self.state, self._exception = self.STATE_EXCEPTION, exception
            self.connection.send((self.STATE_EXCEPTION, exception))
            return True
        except BrokenPipeError:
            return False

    def set_running_or_notify_cancel(self):
        return True

    def cancel(self):
        raise NotImplementedError()

    def result(self, timeout=None):
        self._recv(timeout)
        if self.state == self.STATE_FINISHED:
            return self._result
        elif self.state == self.STATE_EXCEPTION:
            raise self._exception
        else:
            assert self.state == self.STATE_CANCELLED
            raise CancelledError()

    def exception(self, timeout=None):
        self._recv(timeout)
        return self._exception

    def done(self):
        return self.state in (self.STATE_FINISHED, self.STATE_EXCEPTION, self.STATE_CANCELLED)

    def running(self):
        return self.state == self.STATE_RUNNING

    def cancelled(self):
        warn("cancelled not implemented")
        return False

    def add_done_callback(self, callback):
        raise NotImplementedError()

    def __repr__(self):
        try:
            self._recv(timeout=0)
        except TimeoutError:
            pass
        if self.state == self.STATE_FINISHED:
            return "<MPFuture at 0x{:x} state=finished returned {}>".format(id(self), type(self._result))
        elif self.state == self.STATE_EXCEPTION:
            return "<MPFuture at 0x{:x} state=finished raised {}>".format(id(self), type(self._exception))
        else:
            return "<MPFuture at 0x{:x} state={}>".format(id(self), self.state)
