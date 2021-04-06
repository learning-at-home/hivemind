from threading import Thread, Lock, Event
from typing import Optional, Callable, Any

import torch

from hivemind.dht import DHT
from hivemind.client.averaging import DecentralizedAverager
from hivemind.client.optim.base import DecentralizedOptimizerBase
from hivemind.utils import MPFuture, get_logger

logger = get_logger(__name__)


class DecentralizedOptimizer(DecentralizedOptimizerBase):
    """
    A simple optimizer that trains a shared model by averaging model parameters with peers in the background.

    :param opt: a pytorch optimizer configured to update model parameters.
    :param dht: a running hivemind DHT daemon connected to other peers
    :param prefix: all keys that point to a DHT metadata will have this prefix
    :param target_group_size: maximum group size for averaging (see DecentralizedAverager)
    :param kwargs: additional parameters passed to DecentralizedAverager

    :note: using DecentralizedOptimizer with adaptive learning rates may result in poor convergence due to
      out-of-sync adaptive learning rates (such as adam second momentum or schedule step). Please ensure that these
      statistics are synchronized or use a more advanced DecentralizedOptimizer version, if applicable.
    :note: the base optimizer cannot add param groups after the DecentralizedOptimizer is created
    """
    averager_step: Optional[MPFuture] = None

    def __init__(self, opt: torch.optim.Optimizer, dht: DHT, prefix: str, target_group_size: int,
                 timeout: Optional[float] = None, verbose: bool = False, **kwargs):
        super().__init__(opt, dht)
        with torch.no_grad():
            averaged_tensors = tuple(p.cpu().float().clone().requires_grad_(False)
                                     for group in self.param_groups for p in group['params'])
        self.averager = DecentralizedAverager(averaged_tensors, dht, start=True, prefix=prefix,
                                              target_group_size=target_group_size, **kwargs)
        self.lock_parameters, self.update_event, self.stop_event = Lock(), Event(), Event()
        self.background_averaging_thread = Thread(daemon=True, target=_average_parameters_in_background,
                                                  args=[self.update_event, self.stop_event, self.averager, self.opt],
                                                  kwargs=dict(verbose=verbose, timeout=timeout))
        self.background_averaging_thread.start()

    def step(self, closure: Optional[Callable[[], Any]] = None):
        self.update_event.set()
        with self.lock_parameters:
            return self.opt.step(closure)

    def zero_grad(self, set_to_none: Optional[bool] = False):
        return self.opt.zero_grad()

    def __del__(self):
        self.stop_event.set()
        while self.background_averaging_thread.is_alive():
            self.update_event.set()


def _average_parameters_in_background(update_event: Event, stop_event: Event, averager: DecentralizedAverager,
                                      opt: torch.optim.Optimizer, verbose: bool, **kwargs):
    """ Iteratively find groups of peers, average parameters with these peers and update local model parameters. """
    while True:
        update_event.wait()
        update_event.clear()
        if stop_event.is_set():
            break

        with torch.no_grad(), averager.get_tensors() as averaged_tensors:
            local_tensors = tuple(p for group in opt.param_groups for p in group['params'])
            assert len(local_tensors) == len(averaged_tensors), "The number of optimized parameters should not change."
            for local_tensor, averaged_tensor in zip(local_tensors, averaged_tensors):
                averaged_tensor[...] = local_tensor.cpu().float()

        try:
            if verbose:
                logger.info(f"Starting a new averaging round with current parameters.")
            group_info = averager.step(**kwargs)

            if group_info is not None:
                with torch.no_grad(), averager.get_tensors() as averaged_tensors:
                    for local_tensor, averaged_tensor in zip(local_tensors, averaged_tensors):
                        local_tensor[...] = averaged_tensor.to(dtype=local_tensor.dtype)
                if verbose:
                    logger.info(f"Finished averaging round in with {len(group_info)} peers.")
            else:
                logger.info(f"Averaging round failed: could not find group.")
        except Exception as e:
            logger.info(f"Averaging round failed: caught {e}.")
