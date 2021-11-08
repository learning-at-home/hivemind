import time
from contextlib import contextmanager
from threading import Lock
from typing import Optional


class PerformanceEMA:
    """
    A running estimate of performance (operations/sec) using adjusted exponential moving average
    :param alpha: Smoothing factor in range [0, 1], [default: 0.1].
    """

    def __init__(self, alpha: float = 0.1, eps: float = 1e-20, paused: bool = False):
        self.alpha, self.eps, self.num_updates = alpha, eps, 0
        self.ema_seconds_per_sample, self.samples_per_second = 0, eps
        self.timestamp = time.perf_counter()
        self.paused = paused
        self.lock = Lock()

    def update(self, task_size: float, interval: Optional[float] = None) -> float:
        """
        :param task_size: how many items were processed since last call
        :param interval: optionally provide the time delta it took to process this task
        :returns: current estimate of performance (samples per second), but at most
        """
        assert task_size > 0, f"Can't register processing {task_size} samples"
        if not self.paused:
            self.timestamp, old_timestamp = time.perf_counter(), self.timestamp
            interval = interval if interval is not None else self.timestamp - old_timestamp
        else:
            assert interval is not None, "If PerformanceEMA is paused, please specify the time interval"
        self.ema_seconds_per_sample = (
            self.alpha * interval / task_size + (1 - self.alpha) * self.ema_seconds_per_sample
        )
        self.num_updates += 1
        adjusted_seconds_per_sample = self.ema_seconds_per_sample / (1 - (1 - self.alpha) ** self.num_updates)
        self.samples_per_second = 1 / max(adjusted_seconds_per_sample, self.eps)
        return self.samples_per_second

    def reset_timer(self):
        """Reset the time since the last update so that the next task performance is counted from current time"""
        self.timestamp = time.perf_counter()

    @contextmanager
    def pause(self):
        """While inside this context, EMA will not count the time passed towards the performance estimate"""
        self.paused, was_paused = True, self.paused
        try:
            yield
        finally:
            self.paused = was_paused
            self.reset_timer()

    def __repr__(self):
        return f"{self.__class__.__name__}(ema={self.samples_per_second:.5f}, num_updates={self.num_updates})"

    @contextmanager
    def update_threadsafe(self, task_size: float):
        """
        Update the EMA throughput of a code that runs inside the context manager, supports multiple concurrent threads.

        :param task_size: how many items were processed since last call
        """
        start_timestamp = time.perf_counter()
        yield
        with self.lock:
            self.update(task_size, interval=time.perf_counter() - max(start_timestamp, self.timestamp))
            # note: we define interval as such to support two distinct scenarios:
            # (1) if this is the first call to measure_threadsafe after a pause, count time from entering this context
            # (2) if there are concurrent calls to measure_threadsafe, respect the timestamp updates from these calls
