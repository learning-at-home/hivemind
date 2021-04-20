import time
from threading import Thread, Lock, Event
from typing import Optional

import torch

from hivemind.dht import DHT
from hivemind.client.averaging import DecentralizedAverager
from hivemind.optim.base import DecentralizedOptimizerBase
from hivemind.utils import get_logger, get_dht_time

logger = get_logger(__name__)


class ParameterAveragingOptimizer(DecentralizedOptimizerBase):
    """
    A simple optimizer that trains a shared model by averaging model parameters with peers in the background.

    :param opt: a pytorch optimizer configured to update model parameters.
    :param dht: a running hivemind DHT daemon connected to other peers
    :param averaging_period: if specified, optimizer will attempt to average weights at regular intervals of this many
      seconds. (averaging step will only occur if the optimizer ran at least one step in that interval)
    :param prefix: all DHT keys that point to optimization metadata will have this prefix
    :param target_group_size: maximum group size for averaging (see DecentralizedAverager)
    :param timeout: if DecentralizedAverager step is unable to form group in this many seconds, cancel step
    :param kwargs: additional parameters passed to DecentralizedAverager

    :note: using DecentralizedOptimizer with adaptive learning rates may result in poor convergence due to
      out-of-sync adaptive learning rates (such as adam second momentum or schedule step). Please ensure that these
      statistics are synchronized or use a more advanced DecentralizedOptimizer version, if applicable.
    :note: the base optimizer cannot add param groups after the DecentralizedOptimizer is created
    """

    def __init__(self, opt: torch.optim.Optimizer, dht: DHT, prefix: str, target_group_size: int,
                 averaging_period: float = 0, timeout: Optional[float] = None, verbose: bool = False, **kwargs):
        super().__init__(opt, dht)
        with torch.no_grad():
            averaged_tensors = tuple(p.cpu().float().clone().requires_grad_(False)
                                     for group in self.param_groups for p in group['params'])
        self.averager = DecentralizedAverager(averaged_tensors, dht, start=True, prefix=prefix,
                                              target_group_size=target_group_size, **kwargs)
        self.lock_parameters, self.update_event, self.stop_event = Lock(), Event(), Event()
        self.background_averaging_thread = Thread(
            name=f'{self.__class__.__name__}', daemon=True, target=self._average_parameters_in_background,
            args=[self.lock_parameters, self.update_event, self.stop_event, self.averager, self.opt],
            kwargs=dict(averaging_period=averaging_period, timeout=timeout, verbose=verbose))
        self.background_averaging_thread.start()

    def step(self, *args, **kwargs):
        self.update_event.set()
        with self.lock_parameters:
            return self.opt.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        return self.opt.zero_grad(*args, **kwargs)

    def __del__(self):
        self.stop_event.set()
        self.update_event.set()

    def shutdown(self):
        self.averager.shutdown()

    @staticmethod
    @torch.no_grad()
    def _average_parameters_in_background(
            lock_parameters: Lock, update_event: Event, stop_event: Event, averager: DecentralizedAverager,
            opt: torch.optim.Optimizer, averaging_period: float, verbose: bool, **kwargs):
        """ Iteratively find groups of peers, average parameters with these peers and update local model parameters. """
        while not stop_event.is_set():
            update_event.wait()
            update_event.clear()
            if stop_event.is_set():
                break

            if averaging_period:
                current_time = get_dht_time()
                # note: we use global DHT time to make sure peers start averaging at the ~same time (to form groups)
                time_to_nearest_interval = max(0.0, averaging_period - current_time % averaging_period)
                time.sleep(time_to_nearest_interval)

            with lock_parameters, averager.get_tensors() as averaged_tensors:
                local_tensors = tuple(p for group in opt.param_groups for p in group['params'])
                assert len(local_tensors) == len(averaged_tensors), "The number of optimized parameters should not change."

                for local_tensor, averaged_tensor in zip(local_tensors, averaged_tensors):
                    averaged_tensor[...] = local_tensor.cpu().float()

                old_local_tensors = tuple(local_tensor.cpu().detach().clone() for local_tensor in local_tensors)

            try:
                if verbose:
                    logger.info(f"Starting a new averaging round with current parameters.")
                group_info = averager.step(**kwargs)

                if group_info is not None:
                    with lock_parameters, averager.get_tensors() as averaged_tensors:
                        for local_tensor, old_local_tensor, averaged_tensor in zip(
                            local_tensors, old_local_tensors, averaged_tensors
                        ):
                            local_tensor[...] += averaged_tensor.to(dtype=local_tensor.dtype) - old_local_tensor
                    if verbose:
                        logger.info(f"Finished averaging round in with {len(group_info)} peers.")
                else:
                    if verbose:
                        logger.warning(f"Averaging round failed: could not find group.")
            except Exception as e:
                logger.error(f"Averaging round failed: caught {e}.")


class DecentralizedSGD(ParameterAveragingOptimizer):
    """
    Decentralized Stochastic Gradient Descent algorithm like in Lian et al (2017) [1] based on Moshpit All-Reduce [2].

    :param opt: a pytorch optimizer configured to update model parameters.
    :param dht: a running hivemind DHT daemon connected to other peers
    :param averaging_period: if specified, optimizer will attempt to average weights at regular intervals of this many
      seconds. (averaging step will only occur if the optimizer ran at least one step in that interval)
    :param prefix: all DHT keys that point to optimization metadata will have this prefix
    :param target_group_size: maximum group size for averaging (see DecentralizedAverager)
    :param kwargs: additional parameters passed to DecentralizedAverager

    - [1] Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Parallel Stochastic Gradient
     Descent - https://proceedings.neurips.cc/paper/2017/hash/f75526659f31040afeb61cb7133e4e6d-Abstract.html
    - [2] Moshpit SGD: Communication-Efficient Decentralized Training on Heterogeneous Unreliable Devices
     https://arxiv.org/abs/2103.03239
    """

    def __init__(self, params, lr: float, *, dht: DHT, prefix: str, target_group_size: int, averaging_period: float = 0,
                 momentum: float = 0, dampening: float = 0, weight_decay: float = 0, nesterov: bool = False, **kwargs):
        opt = torch.optim.SGD(params, lr, momentum, dampening, weight_decay, nesterov)
        super().__init__(opt, dht, prefix, target_group_size, averaging_period, **kwargs)
