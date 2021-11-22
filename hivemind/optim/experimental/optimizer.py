from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Union

import torch

from hivemind.averaging.control import StepControl
from hivemind.dht import DHT
from hivemind.optim.experimental.grad_averager import GradientAverager
from hivemind.optim.experimental.progress_tracker import ProgressTracker
from hivemind.optim.experimental.state_averager import (
    LRSchedulerBase,
    OptimizerFactory,
    Parameters,
    ParamGroups,
    SchedulerFactory,
    TorchOptimizer,
    TrainingStateAverager,
)
from hivemind.optim.grad_scaler import GradScaler
from hivemind.utils import get_dht_time, get_logger

logger = get_logger(__name__)


class Optimizer(torch.optim.Optimizer):
    """
    Hivemind Optimizer wraps your regular PyTorch Optimizer for training in a swarm of peers. It can be configured with
     synchronous, delayed or asynchronous updates to trade between optimization guarantees and compute utilization.

    The Optimizer is meant as a drop-in replacement for your regular PyTorch code:

    >>> model = transformers.AutoModel("albert-xxlarge-v2")
    >>> dht = hivemind.DHT(initial_peers=INITIAL_PEERS, start=True)
    >>> opt = hivemind.Optimizer(model.parameters(), optim_cls=torch.optim.Adam, prefix="run_42",
    >>>                          target_batch_size=4096, batch_size_per_step=4)
    >>> while True:
    >>>     loss = compute_loss_on_batch(model, batch_size=4)
    >>>     opt.zero_grad()
    >>>     loss.backward()
    >>>     opt.step()  # <-- train collaboratively with any peers that use the same prefix (run_42)

    However, unlike regular optimizers, calling opt.step with hivemind.Optimizer can do one of the following:
    - accumulate a minibatch of data towards the (global) target batch size without changing parameters (yet),
    - after accumulating the target batch size, all-reduce gradients with peers and perform optimizer step,
    - if, for any reason, your peer lags behind the rest of the swarm, it will load state from up-to-date peers.

    :note: Hivemind.Optimizer can be used the same way any other pytorch optimizer, but there is one limitation:
      learning rate schedulers, curriculum and other time-dependent features should use opt.global_step (and not the
      number of local forward-backward cycles). This is because any device can join midway through training, when
      other peers have already made some progress and changed their learning rate accordingly.

    :param dht: a running hivemind.DHT instance connected to other peers
    :param prefix: a unique name of this experiment, used as a common prefix for all DHT keys
    :param target_batch_size: perform optimizer step after all peers collectively accumulate this many samples
    :param batch_size_per_step: before each call to .step, user should accumulate gradients over this many samples
    :param optimizer: a standard pytorch optimizer, preferably a large-batch one such as LAMB, LARS, etc.
    :param params: optional, a list/tuple of parameters or structured param groups for the optimizer
    :param scheduler: if specified, use this scheduler to update optimizer learning rate
    :note: If you are using ColloptaborativeOptimizer with lr_scheduler, it is recommended to pass this scheduler
      explicitly into this class. Otherwise, scheduler may not be synchronized between peers.

    :param matchmaking_time: when looking for group, wait for peers to join for up to this many secodns
    :param averaging_timeout: if an averaging step hangs for this long, it will be cancelled.
    :param load_state_timeout: wait for at most this many seconds before giving up on load_state_from_peers
    :param reuse_grad_buffers: if True, use model's .grad buffers for gradient accumulation.
      This is more memory efficient, but it requires that the user does *NOT* call model/opt zero_grad at all
    :param epoch_tolerance: a peer can temporarily be delayed by this many steps without being deemed out of sync
    :param delay_optimizer_step: if True, run optimizer step in background and apply results in a future step
    :param client_mode: if True, runs training without incoming connections, in a firewall-compatible mode
    :param averager_opts: additional keyword arguments forwarded to both GradientAverager and TrainingStateAverager
    :param tracker_opts: additional keyword arguments forwarded to ProgressTracker
    :param verbose: if True, report internal events such as accumilating gradients and running background tasks

    Internally, hivemind.Optimizer consists of 4 components:
    - DHT, a decentralized key-value storage used for coordination across the swarm
    - GradientAverager that is responsible for aggregating gradients with peers for global steps (can be disabled)
    - TrainingStateAverager holds parameters and optimizer/scheduler statistics, keeping them weakly synchronized
     by averaging with peers. It can also download these variable from other peers if your peer is out of sync.
    - ProgressTracker that uses DHT to track the global training progress: the number of steps or samples accumulated

    """

    def __init__(
        self,
        *,
        dht: DHT,
        prefix: str,
        target_batch_size: int,
        batch_size_per_step: Optional[int] = None,
        optimizer: Union[TorchOptimizer, OptimizerFactory],
        params: Optional[Union[Parameters, ParamGroups]] = None,
        scheduler: Optional[Union[LRSchedulerBase, SchedulerFactory]] = None,
        matchmaking_time: Optional[float] = 15.0,
        averaging_timeout: Optional[float] = 300.0,
        load_state_timeout: float = 600.0,
        reuse_grad_buffers: bool = False,
        epoch_tolerance: int = 1,
        delay_optimizer_step: bool = False,
        client_mode: bool = None,
        averager_opts: Optional[dict] = None,
        tracker_opts: Optional[dict] = None,
        verbose: bool = False,
    ):
        self.dht, self.prefix, self.epoch_tolerance = dht, prefix, epoch_tolerance
        self.averaging_timeout, self.load_state_timeout = averaging_timeout, load_state_timeout
        self.batch_size_per_step, self.target_batch_size = batch_size_per_step, target_batch_size
        self.matchmaking_time, self.delay_optimizer_step = matchmaking_time, delay_optimizer_step

        self.client_mode = client_mode if client_mode is not None else self.dht.client_mode
        self.status_loglevel = logging.INFO if verbose else logging.DEBUG
        self.scheduled_round: Optional[StepControl] = None

        self.state_averager = self._make_state_averager(
            optimizer=optimizer, params=params, scheduler=scheduler, **averager_opts or {}
        )
        self.grad_averager = self._make_gradient_averager(reuse_grad_buffers=reuse_grad_buffers, **averager_opts or {})
        self.tracker = self._make_progress_tracker(target_batch_size, **tracker_opts or {})
        self._schema_hash = self._compute_schema_hash()
        self._parent_pid = os.getpid()

        self._step_supports_amp_scaling = self.grad_averager.reuse_grad_buffers
        # note: the line above is used by pytorch AMP GradScaler to enable custom behavior needed when reusing gradient
        # buffers over multiple steps (to avoid repeated unscaling). Without reuse_grad_buffers, this is not needed.

    def _make_state_averager(self, **kwargs) -> TrainingStateAverager:
        return TrainingStateAverager(
            dht=self.dht,
            prefix=f"{self.prefix}_state_averager",
            allreduce_timeout=self.averaging_timeout,
            status_loglevel=self.status_loglevel,
            client_mode=self.client_mode,
            offload_optimizer=True,
            custom_gradients=True,
            start=True,
            **kwargs,
        )

    def _make_gradient_averager(self, **kwargs) -> GradientAverager:
        assert hasattr(self, "state_averager"), "must initialize state averager first"
        grad_averager = GradientAverager(
            dht=self.dht,
            prefix=f"{self.prefix}_grad_averager",
            parameters=self.state_averager.main_parameters,
            allreduce_timeout=self.averaging_timeout,
            client_mode=self.client_mode,
            start=True,
            **kwargs,
        )
        optimized_param_groups = self.state_averager.optimizer.param_groups
        optimized_parameters = [param for group in optimized_param_groups for param in group["params"]]
        with grad_averager.get_tensors() as averaged_gradients:
            assert len(averaged_gradients) == len(optimized_parameters)
            for opt_param, averaged_grad in zip(optimized_parameters, averaged_gradients):
                opt_param.grad = averaged_grad
        return grad_averager

    def _make_progress_tracker(self, target_batch_size: int, **kwargs) -> ProgressTracker:
        return ProgressTracker(
            dht=self.dht,
            prefix=self.prefix,
            target_batch_size=target_batch_size,
            client_mode=self.client_mode,
            status_loglevel=self.status_loglevel,
            start=True,
            **kwargs,
        )

    def _compute_schema_hash(self) -> int:
        optimized_param_groups = self.state_averager.optimizer.param_groups
        optimized_parameters = [param for group in optimized_param_groups for param in group["params"]]
        param_shapes = tuple(tuple(param.shape) for param in optimized_parameters)
        grad_ids = tuple(id(param.grad) for param in optimized_parameters)
        return hash((grad_ids, param_shapes))

    def is_alive(self) -> bool:
        return self.state_averager.is_alive()

    @property
    def local_epoch(self) -> int:
        return self.state_averager.local_epoch

    @property
    def should_load_state_from_peers(self) -> bool:
        """If true, peer will discard local progress and attempt to download state from peers."""
        return self.local_epoch < self.tracker.global_epoch - self.epoch_tolerance

    def step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        grad_scaler: Optional[HivemindGradScaler] = None,
        **kwargs,
    ):
        """
        Report accumulating gradients w.r.t. batch_size additional samples, optionally update model parameters

        :param closure: A closure that reevaluates the model and returns the loss
        :param batch_size: optional override for batch_size_per_step from init
        :param grad_scaler: if amp is enabled, this **must** be a hivemind-aware gradient scaler
        :note: this .step is different from normal pytorch optimizers in several key ways. See __init__ for details.
        """
        if grad_scaler is not None and not isinstance(grad_scaler, GradScaler):
            raise ValueError("hivemind.Optimizer requires a hivemind-aware gradient scaler (hivemind.GradScaler).")
        if self.batch_size_per_step is None and batch_size is None:
            raise ValueError("Please either set batch_size_per_step parameter at init or when calling .step")
        batch_size = batch_size if batch_size is not None else self.batch_size_per_step

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.should_load_state_from_peers:
            self.load_state_from_peers()
            return loss

        if grad_scaler is not None and not grad_scaler.are_grads_finite(self):
            logger.log(self.status_loglevel, "Encountered incorrect value in fp16 grads, resetting local gradients")
            self.tracker.report_local_progress(self.local_epoch, samples_accumulated=0)
            self.grad_averager.reset_accumulated_grads_()
            return loss

        self.grad_averager.accumulate_grads_(batch_size)
        self.tracker.report_local_progress(self.local_epoch, self.grad_averager.local_samples_accumulated)
        self.state_averager.step(apply_delayed_updates=True)

        if self.tracker.estimated_next_update_time - get_dht_time() <= self.matchmaking_time:
            if self.scheduled_round is None or self.scheduled_round.triggered or self.scheduled_round.done():
                eta_seconds = self.tracker.estimated_next_update_time - get_dht_time()
                eta_seconds = max(eta_seconds, self.grad_averager.matchmaking_kwargs["min_matchmaking_time"])
                logger.log(self.status_loglevel, f"Pre-scheduling next averaging round in {eta_seconds:.2f}s.")
                scheduled_time = self.tracker.estimated_next_update_time
                if self.client_mode:
                    scheduled_time = get_dht_time() + self.averaging_timeout
                self.scheduled_round = self.grad_averager.schedule_step(scheduled_time, timeout=self.averaging_timeout)

        if not self.tracker.ready_to_update_epoch:
            return loss

        with self.tracker.pause_updates():
            logger.log(self.status_loglevel, f"Beginning global optimizer step #{self.tracker.global_epoch}")
            # divide accumulators by local steps to recover the true average grad w.r.t. local_samples_accumulated
            if grad_scaler is not None:
                with grad_scaler.running_global_step():
                    assert grad_scaler.unscale_(self.opt)

            if self.scheduled_round is not None and self.scheduled_round.triggered or self.scheduled_round.done():
                logger.log(self.status_loglevel, f"Discarding failed matchmaking results: {self.scheduled_round}")
                self.scheduled_round = None

            need_averaging = self.tracker.global_progress.num_peers > 1
            if need_averaging:
                try:
                    group_info = self.grad_averager.step(
                        control=self.scheduled_round, reset_accumulators=True, timeout=self.averaging_timeout
                    )
                    logger.log(self.status_loglevel, f"Averaged gradients with {len(group_info)} peers")
                except BaseException as e:
                    logger.log(self.status_loglevel, f"Averaging failed with {repr(e)}")

            else:
                if self.scheduled_round is not None:
                    self.scheduled_round.cancel()
                logger.log(self.status_loglevel, f"Skipped averaging: there are no other peers")

            assert self._schema_hash == self._compute_schema_hash(), "parameters or gradients changed during iteration"
            with self.grad_averager.use_averaged_gradients(replace_model_gradients=False):
                # note: we do not need to replace because the offloaded optimizer is already using averaged grads

                self.state_averager.step(
                    increment_epoch=True,
                    optimizer_step=True,
                    delay_optimizer_step=self.delay_optimizer_step,
                    grad_scaler=grad_scaler,
                    averaging_round=need_averaging,
                    delay_averaging=True,
                    averaging_opts=dict(
                        scheduled_time=get_dht_time() + self.matchmaking_time, timeout=self.averaging_timeout
                    )
                    if need_averaging
                    else None,
                )

            self.grad_averager.reset_accumulated_grads_()
            self.tracker.update_epoch(new_epoch=self.state_averager.local_epoch)
            logger.log(self.status_loglevel, f"Optimizer step done! Beginning next epoch {self.local_epoch}.")
        return loss

    def step_aux(self, **kwargs):
        """
        Find and assist other peers in averaging without sending local gradients.

        :note: this .step is different from normal pytorch optimizers in several key ways. See __init__ for details.
        """
        raise NotImplementedError("Auxiliary step for hivemind.Optimizer is not implemented yet.")

    def zero_grad(self, set_to_none: bool = False):
        """Reset gradients from model. If these gradients are reused for accumulators, raise an error."""
        if self.grad_averager.reuse_grad_buffers:
            raise ValueError(
                f"When running {self.__class__.__name__} with reuse_grad_buffers=True, user should never "
                f"call zero_grad manually. Gradients will be refreshed internally."
            )
        for param in self.grad_averager.parameters:
            if param.grad is None:
                pass
            elif set_to_none:
                param.grad = None
            else:
                param.grad.zero_()

    def load_state_from_peers(self, **kwargs):
        """Attempt to fetch the newest collaboration state from other peers"""
        if self.scheduled_round is not None and not self.scheduled_round.done():
            self.scheduled_round.cancel()

        with self.tracker.pause_updates():
            while True:
                try:
                    self.state_averager.load_state_from_peers(timeout=self.load_state_timeout, **kwargs)
                    break
                except KeyboardInterrupt:
                    raise
                except BaseException as e:
                    logger.exception(f"Failed to load state from peers: {e}, retrying ...")
                    continue

            if self.tracker.global_epoch - self.epoch_tolerance <= self.local_epoch < self.tracker.global_epoch:
                logger.log(self.status_loglevel, f"Catching up with collaboration step {self.tracker.global_epoch}.")
                self.state_averager.local_epoch = self.tracker.global_epoch

            self.tracker.report_local_progress(local_epoch=self.local_epoch, samples_accumulated=0)
            self.grad_averager.reset_accumulated_grads_()

    def state_dict(self) -> dict:
        state_dict = self.state_averager.optimizer.state_dict()
        state_dict["state"]["local_epoch"] = self.local_epoch
        return state_dict

    def load_state_dict(self, state_dict: dict):
        if "local_epoch" in state_dict["state"]:
            self.state_averager.local_epoch = state_dict["state"].pop("local_epoch")
        return self.state_averager.optimizer.load_state_dict(state_dict)

    @property
    def state(self):
        return dict(self.state_averager.optimizer.state, local_epoch=self.local_epoch)

    @property
    def opt(self) -> TorchOptimizer:
        # for compatibility with HivemindGradScaler
        return self.state_averager.optimizer

    @property
    def param_groups(self) -> ParamGroups:
        next_index = 0
        param_groups = tuple(dict(param_group) for param_group in self.state_averager.optimizer.param_groups)
        for param_group in param_groups:
            num_params = len(param_group["params"])
            main_params_for_group = self.state_averager.main_parameters[next_index : next_index + num_params]
            param_group["params"] = main_params_for_group
            next_index += num_params
        assert next_index == len(self.state_averager.main_parameters)
        return param_groups

    def add_param_group(self, param_group: dict) -> None:
        raise ValueError(
            f"{self.__class__.__name__} does not support calling add_param_group after creation."
            f"Please provide all parameter groups at init."
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(prefix={self.prefix}, epoch={self.local_epoch})"

    def shutdown(self):
        logger.debug("Sending goodbye to peers...")
        self.tracker.shutdown()
        logger.debug("Shutting down averager...")
        self.state_averager.step(wait_for_delayed_update=True)
        self.state_averager.shutdown()
        self.grad_averager.shutdown()
        logger.debug(f"{self.__class__.__name__} is shut down.")

    def __del__(self):
        if self.is_alive() and self._parent_pid == os.getpid():
            self.shutdown()
