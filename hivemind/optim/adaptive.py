from typing import Optional, Sequence

from hivemind.optim.collaborative import CollaborativeOptimizer
from hivemind import DHT, TrainingAverager


class CollaborativeAdaptiveOptimizer(CollaborativeOptimizer):
    """
    Behaves exactly as CollaborativeOptimizer except:
     - averages adaptive learning rates of an optimizer
     - doesn't average gradients
    :param average_opt_statistics: average optimizer statistics with corresponding names in statedict
    :param kwargs: options for CollaborativeOptimizer
    """
    def __init__(self, average_opt_statistics: Sequence[str], **kwargs):
        self.average_opt_statistics = average_opt_statistics
        super().__init__(self, **kwargs)

    def _make_averager(self, **kwargs):
        return TrainingAverager(self.opt, dht=self.dht, average_parameters=True, average_gradients=False,
                                average_opt_statistics=self.average_opt_statistics,
                                prefix=f"{self.prefix}_averaging", allreduce_timeout=self.averaging_timeout,
                                listen=not self.client_mode, **kwargs
                                )

