""" A wrapper for pytorch optimizers that performs decentralized averaging in background """
from typing import Optional, Iterable

import torch

import hivemind
from hivemind.client.averaging import DecentralizedAverager
from hivemind.utils import get_logger, nested_flatten

logger = get_logger(__name__)

# TODO this file is a work in progress, it should be updated after DecentralizedAverager is complete


class DecentralizedOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer, *, dht: hivemind.dht.DHT,
                 average_optimizer: bool = False, averaged_tensors: Optional[Iterable[torch.Tensor]] = None, **kwargs):
        """
        A wrapper for torch optimizer that will periodically average model parameters with other peers.
        :param optimizer: an normal pytorch optimizer to be wrapped
        :param dht: a DHT node that will be used to find groups
        :param average_optimizer: if True, all-reduce will aggregate both model parameters and optimizer,
           otherwise average only model parameters (or averaged_tensors, if specified)
        :param averaged_tensors: manually specify all tensors that should be averaged
        :param kwargs: see DecentralizedAverager parameters
        """
        self.optimizer, self.dht, self._averager, self._called_step = optimizer, dht, None, False
        assert not average_optimizer or (averaged_tensors is None), "Please use either average_optimizer=True or" \
                                                                    " averaged_tensors, but not both."
        self.averager_opts = average_optimizer, averaged_tensors, kwargs

    @property
    def averager(self) -> DecentralizedAverager:
        if not self._called_step:
            raise ValueError("DecentralizedOptimizer.averager will be created when you call .step() for the first time")
        if self._averager is not None:
            return self._averager

        average_optimizer, averaged_tensors, kwargs = self.averager_opts
        if averaged_tensors is None:
            averaged_tensors = [param for param_group in self.optimizer.param_groups for param in param_group['params']]
            if average_optimizer:
                found_optimizer_stats = False
                for value in nested_flatten(self.optimizer.state_dict()):
                    if isinstance(value, torch.Tensor) and value not in averaged_tensors:
                        averaged_tensors.append(value)
                        found_optimizer_stats = True
                if not found_optimizer_stats:
                    logger.warning("Using average_optimizer=True, but found no optimizer statistics. Make sure your "
                                   "optimizer has tensors in its .state_dict().")
        else:
            averaged_tensors = list(averaged_tensors)

        self._averager = DecentralizedAverager(averaged_tensors, dht=self.dht, start=True, **kwargs)
        return self._averager

    def step(self, *args, **kwargs):
        step_result = self.optimizer.step(*args, **kwargs)
        if self.averager.found_group:
            self.averager.all_reduce(inplace=True)  # TODO background averaging a-la hogwild
        if not self.averager.looking_for_group:
            self.averager.look_for_group()
        return step_result

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def __repr__(self):
        return f"DecentralizedOptimizer({repr(self.optimizer)})"

    def __del__(self):
        logger.info("Deleting DecentralizedOptimizer, shutting down background averaging process")
        if self._averager is not None:
            self._averager.shutdown()