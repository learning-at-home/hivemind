from typing import Sequence

import torch.optim

from hivemind.optim.collaborative import CollaborativeOptimizer
from hivemind.optim.training_averager import TrainingAverager


class CollaborativeAdaptiveOptimizer(CollaborativeOptimizer):
    """
    Behaves exactly as CollaborativeOptimizer except:

    * averages adaptive learning rates of an optimizer
    * doesn't average gradients

    :param average_opt_statistics: average optimizer statistics with corresponding names in statedict
    :param kwargs: options for CollaborativeOptimizer
    """

    def __init__(self, opt: torch.optim.Optimizer, average_opt_statistics: Sequence[str], **kwargs):
        super().__init__(opt, average_opt_statistics=average_opt_statistics, **kwargs)

    def _make_averager(self, average_opt_statistics, **kwargs):
        return TrainingAverager(
            self.opt,
            dht=self.dht,
            average_parameters=True,
            average_gradients=False,
            average_opt_statistics=average_opt_statistics,
            prefix=f"{self.prefix}_averaging",
            allreduce_timeout=self.averaging_timeout,
            client_mode=self.client_mode,
            **kwargs,
        )
