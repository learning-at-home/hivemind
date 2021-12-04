import time
from threading import Event, Lock, Thread
from typing import Optional, Sequence, Tuple

import torch

from hivemind.dht import DHT
from hivemind.optim.base import DecentralizedOptimizerBase
from hivemind.optim.training_averager import TrainingAverager
from hivemind.utils import get_dht_time, get_logger

logger = get_logger(__name__)


class DecentralizedOptimizer(DecentralizedOptimizerBase):
    """
    A simple optimizer that trains a shared model by averaging with peers in variety of ways. Supports
    parameter/gradient averaging and syncing adaptive learning rates or any other internal statistics of optimizer.

    :param opt: a pytorch optimizer configured to update model parameters.
    :param dht: a running hivemind DHT daemon connected to other peers
    :param average_parameters: whether to average model parameters
    :param average_gradients: whether to average gradients
    :param average_opt_statistics: if specified, average optimizer states with corresponding names in state_dict
    :param averaging_steps_period: performs averaging after this many optimizer steps
    :param averaging_time_period: if specified, optimizer will attempt to average weights at regular intervals of this
      many seconds. (averaging step will only occur if the optimizer ran `averaging_steps_period` steps in that interval)
    :param prefix: all DHT keys that point to optimization metadata will have this prefix
    :param target_group_size: maximum group size for averaging (see DecentralizedAverager)
    :param timeout: if DecentralizedAverager step is unable to form group in this many seconds, cancel step
    :param kwargs: additional parameters passed to TrainingAverager
    :note: if you're using an optimizer with adaptive learning rates (such as Adam), make sure to specify
      necessary fields' names in `average_opt_statistics`. Otherwise you may encounter poor convergence.
    :note: the base optimizer cannot add param groups after the DecentralizedOptimizer is created
    """

    def __init__(
        self,
        opt: torch.optim.Optimizer,
        dht: DHT,
        *,
        prefix: str,
        target_group_size: int,
        average_parameters: bool,
        average_gradients: bool,
        average_opt_statistics: Sequence[str] = (),
        averaging_steps_period: int = 1,
        averaging_time_period: float = 0,
        timeout: Optional[float] = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(opt, dht)
        assert averaging_steps_period > 0 and averaging_time_period >= 0, "Averaging period must be positive."
        self.local_step, self.averaging_step_period = 0, averaging_steps_period

        self.averager = TrainingAverager(
            opt,
            average_parameters=average_parameters,
            average_gradients=average_gradients,
            average_opt_statistics=average_opt_statistics,
            dht=dht,
            start=True,
            prefix=prefix,
            target_group_size=target_group_size,
            **kwargs,
        )
        self.lock_parameters, self.update_event, self.stop_event = Lock(), Event(), Event()
        self.lock_parameters.acquire()  # this lock is only released when averager can modify tensors in background

        self.background_averaging_thread = Thread(
            name=f"{self.__class__.__name__}",
            daemon=True,
            target=self._average_parameters_in_background,
            args=[self.lock_parameters, self.update_event, self.stop_event, self.averager],
            kwargs=dict(averaging_period=averaging_time_period, timeout=timeout, verbose=verbose),
        )
        self.background_averaging_thread.start()

    def step(self, *args, **kwargs):
        loss = self.opt.step(*args, **kwargs)
        if self.lock_parameters.locked():
            self.lock_parameters.release()
        try:
            self.local_step += 1
            if self.local_step % self.averaging_step_period == 0:
                self.update_event.set()
            self.averager.pending_updates_done.wait()

            if not self.averager.client_mode:
                self.averager.state_sharing_priority = get_dht_time()

            return loss
        finally:
            self.lock_parameters.acquire()

    def zero_grad(self, *args, **kwargs):
        return self.opt.zero_grad(*args, **kwargs)

    def __del__(self):
        self.stop_event.set()
        self.update_event.set()

    def shutdown(self):
        self.stop_event.set()
        self.update_event.set()
        self.averager.shutdown()

    @staticmethod
    @torch.no_grad()
    def _average_parameters_in_background(
        lock_parameters: Lock,
        update_event: Event,
        stop_event: Event,
        averager: TrainingAverager,
        averaging_period: float,
        verbose: bool,
        **kwargs,
    ):
        """Iteratively find groups of peers, average parameters with these peers and update local model parameters."""
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

            if verbose:
                logger.info(f"Starting a new averaging round with current parameters")
            try:
                group_info = averager.step(lock_parameters, **kwargs)
                if verbose:
                    if group_info is not None:
                        logger.info(f"Finished averaging round in with {len(group_info)} peers")
                    else:
                        logger.warning(f"Averaging round failed: could not find group")
            except Exception as e:
                logger.error(f"Averaging round failed: caught {e}")


class DecentralizedSGD(DecentralizedOptimizer):
    """
    Decentralized Stochastic Gradient Descent algorithm like in Lian et al (2017) [1] based on Moshpit All-Reduce [2].

    :param dht: a running hivemind DHT daemon connected to other peers
    :param prefix: all DHT keys that point to optimization metadata will have this prefix
    :param target_group_size: maximum group size for averaging (see DecentralizedAverager)
    :param kwargs: additional parameters passed to DecentralizedOptimizer

    - [1] Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Parallel Stochastic Gradient
     Descent - https://proceedings.neurips.cc/paper/2017/hash/f75526659f31040afeb61cb7133e4e6d-Abstract.html
    - [2] Moshpit SGD: Communication-Efficient Decentralized Training on Heterogeneous Unreliable Devices
     https://arxiv.org/abs/2103.03239
    """

    def __init__(
        self,
        params,
        lr: float,
        *,
        dht: DHT,
        prefix: str,
        target_group_size: int,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        **kwargs,
    ):
        opt = torch.optim.SGD(params, lr, momentum, dampening, weight_decay, nesterov)
        super().__init__(
            opt,
            dht,
            prefix=prefix,
            target_group_size=target_group_size,
            average_parameters=True,
            average_gradients=False,
            **kwargs,
        )


class DecentralizedAdam(DecentralizedOptimizer):
    """
    Decentralized Adam/AmsGrad as proposed in [1], [2]

    :param dht: a running hivemind DHT daemon connected to other peers
    :param prefix: all DHT keys that point to optimization metadata will have this prefix
    :param target_group_size: maximum group size for averaging (see DecentralizedAverager)
    :param averaging_steps_period: performs averaging after this many optimizer steps
    :param kwargs: additional parameters passed to DecentralizedOptimizer

    - [1] On the Convergence of Decentralized Adaptive Gradient Methods
    - [2] Toward Communication Efficient Adaptive Gradient Method - https://dl.acm.org/doi/abs/10.1145/3412815.3416891
    """

    def __init__(
        self,
        params,
        lr: float,
        *,
        dht: DHT,
        prefix: str,
        target_group_size: int,
        averaging_steps_period: int,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        **kwargs,
    ):
        opt = torch.optim.Adam(params, lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        opt_statistics = ("max_exp_avg_sq",) if amsgrad else ("exp_avg_sq",)

        super().__init__(
            opt,
            dht,
            prefix=prefix,
            target_group_size=target_group_size,
            average_parameters=True,
            average_gradients=False,
            average_opt_statistics=opt_statistics,
            averaging_steps_period=averaging_steps_period,
            **kwargs,
        )
