from contextlib import contextmanager

from hivemind.utils import get_dht_time


class PerformanceEMA:
    """
    A running estimate of performance (operations/sec) using adjusted exponential moving average
    :param alpha: Smoothing factor in range [0, 1], [default: 0.1].
    """

    def __init__(self, alpha: float = 0.1, eps: float = 1e-20):
        self.alpha, self.eps, self.num_updates = alpha, eps, 0
        self.ema_seconds_per_sample, self.samples_per_second = 0, eps
        self.timestamp = get_dht_time()
        self.paused = False

    def update(self, num_processed: int) -> float:
        """
        :param num_processed: how many items were processed since last call
        :returns: current estimate of performance (samples per second), but at most
        """
        assert not self.paused, "PerformanceEMA is currently paused"
        assert num_processed > 0, f"Can't register processing {num_processed} samples"
        self.timestamp, old_timestamp = get_dht_time(), self.timestamp
        seconds_per_sample = max(0, self.timestamp - old_timestamp) / num_processed
        self.ema_seconds_per_sample = self.alpha * seconds_per_sample + (1 - self.alpha) * self.ema_seconds_per_sample
        self.num_updates += 1
        adjusted_seconds_per_sample = self.ema_seconds_per_sample / (1 - (1 - self.alpha) ** self.num_updates)
        self.samples_per_second = 1 / max(adjusted_seconds_per_sample, self.eps)
        return self.samples_per_second

    @contextmanager
    def pause(self):
        """While inside this context, EMA will not count the time passed towards the performance estimate"""
        self.paused, was_paused = True, self.paused
        try:
            yield
        finally:
            self.timestamp = get_dht_time()
            self.paused = was_paused
