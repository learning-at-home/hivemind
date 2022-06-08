from __future__ import annotations

import logging
import os
import time
from functools import partial
from typing import Callable, Optional, Sequence, Union

import torch

from hivemind.averaging.control import AveragingStage, StepControl
from hivemind.compression import CompressionBase, NoCompression
from hivemind.dht import DHT
from hivemind.optim.grad_averager import GradientAverager, GradientAveragerFactory
from hivemind.optim.grad_scaler import GradScaler
from hivemind.optim.progress_tracker import LocalTrainingProgress, ProgressTracker
from hivemind.optim.state_averager import (
    LRSchedulerBase,
    OptimizerFactory,
    Parameters,
    ParamGroups,
    SchedulerFactory,
    TorchOptimizer,
    TrainingStateAverager,
)
from hivemind.utils import PerformanceEMA, get_dht_time, get_logger

logger = get_logger(__name__)


class Optimizer(torch.optim.Optimizer):
    """
    hivemind.Optimizer wraps your regular PyTorch Optimizer for training collaboratively with peers.

    By default, Optimizer is configured to be exactly **equivalent to synchronous training** with target_batch_size.
    There are advanced options make training semi-asynchronous (delay_optimizer_step and delay_gradient_averaging)
    or even fully asynchronous (use_local_updates=True).

    :example: The Optimizer can be used as a drop-in replacement for a regular PyTorch Optimizer:

    >>> model = transformers.AutoModel("albert-xxlarge-v2")
    >>> dht = hivemind.DHT(initial_peers=INITIAL_PEERS, start=True)
    >>> opt = hivemind.Optimizer(dht=dht, run_id="run_42", batch_size_per_step=4, target_batch_size=4096,
    >>>                          params=model.parameters(), optimizer=lambda params: torch.optim.Adam(params))
    >>> while True:
    >>>     loss = compute_loss_on_batch(model, batch_size=4)
    >>>     opt.zero_grad()
    >>>     loss.backward()
    >>>     opt.step()  # <-- train collaboratively with any peers that use the same prefix (run_42)

    By default, peers will perform the following steps:

     * accumulate a minibatch of gradients towards the (global) target batch size, without updating parameters yet;
     * after peers collectively accumulate target_batch_size, average gradients with peers and perform optimizer step;
     * if your peer lags behind the rest of the swarm, it will download parameters and optimizer state from others;

    Unlike regular training, your device may join midway through training, when other peers already made some progress.
    For this reason, any learning rate schedulers, curriculum and other **time-dependent features should be based on**
    ``optimizer.local_epoch`` (and not the number ot calls to opt.step). Otherwise, peers that joined training late
    may end up having different learning rates. To do so automatically, specify ``scheduler=...`` parameter below.

    :What is an epoch?: Optimizer uses the term ``epoch`` to describe intervals between synchronizations. One epoch
      coresponds to processing certain number of training samples (``target_batch_size``) in total across all peers.
      Like in PyTorch LR Scheduler, **epoch does not necessarily correspond to a full pass over the training data.**
      At the end of epoch, peers perform synchronous actions such as averaging gradients for a global optimizer update,
      updating the learning rate scheduler or simply averaging parameters (if using local updates).
      The purpose of this is to ensure that changing the number of peers does not require changing hyperparameters.
      For instance, if the number of peers doubles, they will run all-reduce more frequently to adjust for faster training.

    :Configuration guide: This guide will help you set up your first collaborative training run. It covers the most
      important basic options, but ignores features that require significant changes to the training code.

    >>> dht = hivemind.DHT(initial_peers=INITIAL_PEERS, client_mode=IF_BEHIND_FIREWALL_OR_VERY_UNRELIABLE, start=True)
    >>> opt = hivemind.Optimizer(
    >>>    dht=dht, run_id="a_unique_name_that_every_participant_will_see_when_training",
    >>>    batch_size_per_step=ACTUAL_BATCH_SIZE_OF_THIS_PEER, target_batch_size=LARGE_GLOBAL_BATCH,
    >>>    # ^--- Each global optimzier step will use gradients from 1x-1.1x of target_batch_size (due to latency);
    >>>    # It is recommended to train with very large batch sizes to reduce the % of time spent on communication.
    >>>
    >>>    params=params, optimizer=lambda params: AnyPyTorchOptimizer(params, **hyperparams_for_target_batch_size),
    >>>    # tune learning rate for your target_batch_size. Here's a good reference: https://arxiv.org/abs/1904.00962
    >>>    scheduler=lambda opt: AnyPyTorchScheduler(opt, **hyperparams_for_target_batch_size),
    >>>    # scheduler.step will be called automatically each time when peers collectively accumulate target_batch_size
    >>>
    >>>    offload_optimizer=True,  # saves GPU memory, but increases RAM usage; Generally a good practice to use this.
    >>>    delay_grad_averaging=OPTIONAL, delay_optimizer_step=OPTIONAL, # train faster, but with 1 round of staleness;
    >>>    # setting both to True is equivalent to Delayed Parameter Updates (see https://arxiv.org/abs/2101.06840)
    >>>
    >>>    grad_compression=hivemind.Float16Compression(),  state_averaging_compression=hivemind.Float16Compression(),
    >>>    # ^-- it is usually fine to use pure 16-bit or even lower precision during communication with no precaution;
    >>>    # See hivemind/examples/albert for an working example of mixed 8/16-bit compression.
    >>>
    >>>    matchmaking_time=15.0, # 3-5s for small local runs, 10-15s for training over the internet or with many peers
    >>>    averaging_timeout=60.0,  # around of 2x the actual time it takes to run all-reduce
    >>>    verbose=True  # periodically report the training progress to the console (e.g. "Averaged with N peers")
    >>> )  # and you're done!


    :param dht: a running hivemind.DHT instance connected to other peers.
    :param run_id: a unique identifier of this training run, used as a common prefix for all DHT keys.
      **Note:** peers with the same run_id should *generally* train the same model and use compatible configurations.
      Some options can be safely changed by individual peers: ``batch_size_per_step``, ``client_mode``, ``auxiliary``,
      ``reuse_grad_buffers``, ``offload_optimizer``, and ``verbose``. In some cases, other options may also be tuned
      individually by each peer, but they should be changed with caution to avoid deadlocks or convergence issues.

    :param target_batch_size: global batch size that must be accumulated before the swarm transitions to the next epoch.
      The actual batch may be *slightly* larger due asynchrony (e.g. peers submit more gradients in the last second).
    :param batch_size_per_step: you should accumulate gradients over this many samples between calls to optimizer.step.

    :param params: parameters or param groups for the optimizer; required if optimizer is a callable(params).
    :param optimizer: a callable(parameters) -> pytorch.optim.Optimizer or a pre-initialized PyTorch optimizer.
      **Note:** some advanced options like offload_optimizer, delay_optimizer_step, or delay_grad_averaging require
      and require the callable and will not work if hivemind.optimizer is created with a pre-existing PyTorch Optimizer.
    :param scheduler: callable(optimizer) -> PyTorch LRScheduler or a pre-initialized PyTorch scheduler.
      The learning rate scheduler will adjust learning rate based on global epoch, not the number of
      local calls to optimizer.step; this is required to keep different peers synchronized.

    :param matchmaking_time: when looking for group, wait for peers to join for up to this many seconds.
      Increase if you see "averaged gradients with N peers" where N is below 0.9x the real siee on >=25% of epochs.
      When training with low-latency network, decreasing matchmaking_time allows training with smaller batch sizes.
    :param averaging_timeout: if an averaging step hangs for this long, it will be cancelled automatically.
      Increase averaging_timeout if you see "Proceeding with local gradients" at least 25% of the time.
      Do not set this timeout too high, as it may cause your optimizer to hang after some types of network errors.
    :param allreduce_timeout: timeout for a single attempt to run all-reduce, default: equal to averaging_timeout.
    :param load_state_timeout: wait for at most this many seconds before giving up on load_state_from_peers.
    :param reuse_grad_buffers: if True, use model's .grad buffers for gradient accumulation.
      This is more memory efficient, but it requires that the user does *NOT* call model/opt zero_grad at all

    :param offload_optimizer: offload the optimizer to host memory, saving GPU memory for parameters and gradients
    :param delay_optimizer_step: run optimizer in background, apply results in future .step; requires offload_optimizer
    :param delay_grad_averaging: average gradients in background; requires offload_optimizer and delay_optimizer_step

    :param delay_state_averaging: if enabled (default), average parameters and extra tensors in a background thread;
      if set to False, average parameters synchronously within the corresponding hivemind.Optimizer.step call.

    :param average_state_every: average state (parameters, chosen opt tensors) with peers every this many **epochs**.
      This reduces the communication overhead increasing, but can cause parameters to diverge if too large.
      The maximal average_state_every=num_epochs depends on how often peers diverge from each other. If peers
      hardly ever skip averaging rounds, they can average state less frequently. In turn, network failures, lossy
      gradient compression and local_updates cause parameters to diverge faster and requires more frequent averaging.

    :param use_local_updates: if enabled, peers will update parameters on each .step using local gradients;
      if not enabled (default), accumulate gradients to target_batch_size, and then call .step with averaged gradients.
      Even if use_local_updates=True, learning rate scheduler will still be called once per target_batch_size.

    :param client_mode: if True, this peer will not accept incoming connections (firewall-compatible mode)
    :param auxiliary: if True, optimizer.step will only assist other peers in averaging (for cpu-only workers)

    :param grad_compression: compression strategy used for averaging gradients, default = no compression
    :param grad_averager_factory: if provided, creates gradient averager with required averaging strategy
    :param state_averaging_compression: compression for averaging params and state tensors, default = no compression
    :param load_state_compression: compression strategy for loading state from peers, default = no compression
    :param average_opt_statistics: names of optimizer statistics from state dict that should be averaged with peers
    :param extra_tensors: if specified, these extra tensors will also be averaged and shared in load_state_from_peers.

    :param averager_opts: additional keyword arguments forwarded to both GradientAverager and TrainingStateAverager
    :param tracker_opts: additional keyword arguments forwarded to ProgressTracker
    :param performance_ema_alpha: moving average alpha in ProgressTracker, TrainingStateAverager and Optimizer
    :param verbose: if True, report internal events such as accumilating gradients and running background tasks

    :note: in a large-scale training, peers will inevitably fail and you will see error messages. hivemind.Optimizer
      is designed to recover from such failures, but will sometimes need a minute or two to re-adjust.

    """

    def __init__(
        self,
        *,
        dht: DHT,
        run_id: str,
        target_batch_size: int,
        batch_size_per_step: Optional[int] = None,
        optimizer: Union[TorchOptimizer, OptimizerFactory],
        params: Optional[Union[Parameters, ParamGroups]] = None,
        scheduler: Optional[Union[LRSchedulerBase, SchedulerFactory]] = None,
        matchmaking_time: Optional[float] = 15.0,
        averaging_timeout: Optional[float] = 60.0,
        allreduce_timeout: Optional[float] = None,
        next_chunk_timeout: Optional[float] = None,
        load_state_timeout: float = 600.0,
        reuse_grad_buffers: bool = False,
        offload_optimizer: Optional[bool] = None,
        delay_optimizer_step: Optional[bool] = None,
        delay_grad_averaging: bool = False,
        delay_state_averaging: bool = True,
        average_state_every: int = 1,
        use_local_updates: bool = False,
        client_mode: bool = None,
        auxiliary: bool = False,
        grad_compression: CompressionBase = NoCompression(),
        grad_averager_factory: Optional[GradientAveragerFactory] = None,
        state_averaging_compression: CompressionBase = NoCompression(),
        load_state_compression: CompressionBase = NoCompression(),
        average_opt_statistics: Sequence[str] = (),
        extra_tensors: Sequence[torch.Tensor] = (),
        averager_opts: Optional[dict] = None,
        tracker_opts: Optional[dict] = None,
        performance_ema_alpha: float = 0.1,
        shutdown_timeout: float = 5,
        verbose: bool = False,
    ):
        self._parent_pid = os.getpid()

        client_mode = client_mode if client_mode is None else dht.client_mode
        delay_optimizer_step = delay_optimizer_step if delay_optimizer_step is not None else delay_grad_averaging
        if offload_optimizer is None:
            offload_optimizer = params is not None and not use_local_updates
        allreduce_timeout = allreduce_timeout if allreduce_timeout is not None else averaging_timeout
        next_chunk_timeout = next_chunk_timeout if next_chunk_timeout is not None else matchmaking_time
        assert not delay_grad_averaging or delay_optimizer_step, "delay_grad_averaging requires delay_optimizer_step"
        assert not (client_mode and auxiliary), "Client-mode peers cannot serve as auxiliaries"
        assert not auxiliary or batch_size_per_step is None, "Auxiliary peers should not accumulate batches"
        if callable(optimizer) and params is not None:
            if scheduler is not None and (not callable(scheduler) or isinstance(scheduler, LRSchedulerBase)):
                raise ValueError("For this mode, please provide scheduler factory: callable(optimizer) -> scheduler")
        elif all(hasattr(optimizer, attr) for attr in ("param_groups", "step", "zero_grad")):
            if offload_optimizer or delay_optimizer_step or delay_grad_averaging:
                raise ValueError(
                    "To enable offload_optimizer or delayed updates, please initialize Optimizer as "
                    "hivemind.Optimizer(..., params=params, optimizer=lambda params: create_opt(params)"
                )
        else:
            raise ValueError(
                "Please initialize the optimizer in one of the following two ways:\n"
                "(A) hivemind.Optimizer(..., params=params, optimizer=lambda params: create_opt(params)\n"
                "(B) hivemind.Optimizer(..., optimizer=pre_initialize_optimizer)"
            )
        if use_local_updates:
            assert not reuse_grad_buffers, "if local_updates is True, gradients will not be accumulated"
            assert not delay_grad_averaging, "if local_updates is True, gradients will not be averaged"
            assert (
                grad_averager_factory is None
            ), "if local_updates is True, provided grad_averager_factory will not be used"

        self.dht, self.run_id, self.client_mode, self.auxiliary = dht, run_id, client_mode, auxiliary
        self.batch_size_per_step, self.target_batch_size = batch_size_per_step, target_batch_size
        self.delay_state_averaging, self.average_state_every = delay_state_averaging, average_state_every
        self.matchmaking_time, self.offload_optimizer = matchmaking_time, offload_optimizer
        self.delay_grad_averaging, self.delay_optimizer_step = delay_grad_averaging, delay_optimizer_step

        self.averaging_timeout, self.allreduce_timeout = averaging_timeout, allreduce_timeout
        self.load_state_timeout, self.shutdown_timeout = load_state_timeout, shutdown_timeout
        self.next_chunk_timeout = next_chunk_timeout

        self.status_loglevel = logging.INFO if verbose else logging.DEBUG
        self.scheduled_grads: Optional[StepControl] = None
        self.scheduled_state: Optional[StepControl] = None

        self.tracker = self._make_progress_tracker(
            target_batch_size, performance_ema_alpha=performance_ema_alpha, **tracker_opts or {}
        )
        self.state_averager = self._make_state_averager(
            optimizer=optimizer,
            params=params,
            scheduler=scheduler,
            delta_rule_averaging=use_local_updates and self.delay_state_averaging,
            compression=state_averaging_compression,
            state_compression=load_state_compression,
            average_opt_statistics=average_opt_statistics,
            performance_ema_alpha=performance_ema_alpha,
            extra_tensors=extra_tensors,
            **averager_opts or {},
        )
        if not use_local_updates:
            self.grad_averager = self._make_gradient_averager(
                grad_averager_factory, reuse_grad_buffers=reuse_grad_buffers, compression=grad_compression
            )
        else:
            self.grad_averager = None

        self._should_check_synchronization_on_update = True  # used in self.should_load_state_from_peers
        self._schema_hash = self._compute_schema_hash()

        self.delay_before_state_averaging = PerformanceEMA(alpha=performance_ema_alpha)
        # measures the average time from the beginning of self._update_global_epoch to the call to state_averager
        # used for pre-scheduling the averaging round in state_averager

        self._step_supports_amp_scaling = reuse_grad_buffers
        # note: the line above is used by pytorch AMP GradScaler to enable custom behavior needed when reusing gradient
        # buffers over multiple steps (to avoid repeated unscaling). Without reuse_grad_buffers, this is not needed.

    def _make_state_averager(self, **kwargs) -> TrainingStateAverager:
        return TrainingStateAverager(
            dht=self.dht,
            prefix=f"{self.run_id}_state_averager",
            min_matchmaking_time=self.matchmaking_time,
            allreduce_timeout=self.allreduce_timeout,
            shutdown_timeout=self.shutdown_timeout,
            offload_optimizer=self.offload_optimizer,
            custom_gradients=self.offload_optimizer,
            status_loglevel=self.status_loglevel,
            next_chunk_timeout=self.next_chunk_timeout,
            client_mode=self.client_mode,
            auxiliary=self.auxiliary,
            start=True,
            **kwargs,
        )

    def _make_gradient_averager(self, factory: Optional[GradientAveragerFactory], **kwargs) -> GradientAverager:
        assert hasattr(self, "state_averager"), "must initialize state averager first"
        factory = factory if factory is not None else GradientAverager
        grad_averager = factory(
            dht=self.dht,
            prefix=f"{self.run_id}_grad_averager",
            parameters=self.state_averager.main_parameters,
            min_matchmaking_time=self.matchmaking_time,
            allreduce_timeout=self.allreduce_timeout,
            shutdown_timeout=self.shutdown_timeout,
            next_chunk_timeout=self.next_chunk_timeout,
            client_mode=self.client_mode,
            auxiliary=self.auxiliary,
            start=True,
            **kwargs,
        )
        if self.offload_optimizer:
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
            prefix=self.run_id,
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

        # offloaded optimizer requires that gradient tensors are reused between iterations
        grad_ids = tuple(id(param.grad) for param in optimized_parameters) if self.offload_optimizer else None
        return hash((grad_ids, param_shapes))

    def is_alive(self) -> bool:
        return self.state_averager.is_alive()

    @property
    def local_epoch(self) -> int:
        """
        This worker's current epoch, kept synchronized with peers. If peer's local_epoch lags behind others, it will
        automatically re-synchronize by downloading state from another peer.
        An epoch corresponds to accumulating target_batch_size across all active devices.
        """
        return self.state_averager.local_epoch

    @property
    def local_progress(self) -> LocalTrainingProgress:
        return self.tracker.local_progress

    @property
    def use_local_updates(self) -> bool:
        return self.grad_averager is None

    @property
    def use_gradient_averaging(self) -> bool:
        return self.grad_averager is not None

    def step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        grad_scaler: Optional[GradScaler] = None,
    ):
        """
        Update training progress after accumulating another local batch size. Depending on the configuration, this will
        report progress to peers, run global or local optimizer step, average parameters or schedule background tasks.

        :param closure: A closure that reevaluates the model and returns the loss.
        :param batch_size: optional override for batch_size_per_step from init.
        :param grad_scaler: if amp is enabled, this **must** be a hivemind-aware gradient scaler.
        :note: this .step is different from normal pytorch optimizers in several key ways. See __init__ for details.
        """
        if grad_scaler is not None and not isinstance(grad_scaler, GradScaler):
            raise ValueError("hivemind.Optimizer requires a hivemind-aware gradient scaler (hivemind.GradScaler)")
        if self.batch_size_per_step is None and batch_size is None and not self.auxiliary:
            raise ValueError("Please either set batch_size_per_step parameter at init or when calling .step")
        if self.auxiliary and (closure is not None or batch_size is not None or grad_scaler is not None):
            raise ValueError("Auxiliary peers should not have batch size, run closures, or use grad_scaler")
        batch_size = batch_size if batch_size is not None else self.batch_size_per_step

        # if delayed updates finished before step, apply these updates; otherwise do nothing
        self.state_averager.step(apply_delayed_updates=True)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.auxiliary and self._should_load_state_from_peers():
            logger.log(self.status_loglevel, "Peer is out of sync")
            self.load_state_from_peers()
            return loss  # local gradients were computed with out-of-sync parameters, must start over

        if self.use_gradient_averaging:
            # accumulate gradients toward target batch size, then aggregate with peers and run optimizer
            if not self.auxiliary:
                grads_are_valid = self._check_and_accumulate_gradients(batch_size, grad_scaler)
                if not grads_are_valid:
                    return loss  # local gradients were reset due to overflow, must start over

            self._maybe_schedule_gradient_averaging()
            self._maybe_schedule_state_averaging()

        else:
            # use_local_updates=True: update parameters on every step independently of other peers
            if not self.auxiliary:
                if grad_scaler is not None:
                    with grad_scaler.running_global_step():
                        assert grad_scaler.unscale_(self)

                new_samples_accumulated = self.tracker.local_progress.samples_accumulated + batch_size
                self.tracker.report_local_progress(self.local_epoch, new_samples_accumulated)
                self._maybe_schedule_state_averaging()

                self.state_averager.step(
                    increment_epoch=False,
                    optimizer_step=True,
                    delay_optimizer_step=self.delay_optimizer_step,
                    grad_scaler=grad_scaler,
                )

        if self.tracker.ready_to_update_epoch:
            self._update_global_epoch(grad_scaler)

        return loss

    def _update_global_epoch(self, grad_scaler: Optional[GradScaler]) -> None:
        """Depending on the configuration: aggregate gradients and/or parameters, perform global optimizer step"""
        assert self._schema_hash == self._compute_schema_hash(), "parameters or gradients changed during iteration"
        _epoch_start_time = time.perf_counter()

        with self.tracker.pause_updates():
            wait_for_trigger = None

            if self.use_gradient_averaging:
                logger.log(self.status_loglevel, f"Beginning optimizer step #{self.local_epoch}")
                if self.delay_optimizer_step:
                    self.state_averager.step(wait_for_delayed_updates=True)

                began_averaging_gradients = self._begin_averaging_gradients(grad_scaler)
                if not began_averaging_gradients:
                    # failed to start gradient averaging due to an internal error
                    self.grad_averager.load_accumulators_into_averager_()
                elif self.delay_grad_averaging:
                    # if using delayed grad averaing, send this to state_averager as a pre-condition for optimizer step
                    wait_for_trigger = partial(self._average_gradients_and_load_into_optimizer, self.scheduled_grads)
                else:
                    # delay_grad_averaging=False, average gradients immediately
                    self._average_gradients_and_load_into_optimizer(self.scheduled_grads)

            next_epoch = max(self.local_epoch + 1, self.tracker.global_epoch)
            swarm_not_empty = self.tracker.global_progress.num_peers > 1
            should_perform_optimizer_step = not self.auxiliary and not self.use_local_updates
            should_average_state = (
                swarm_not_empty
                and next_epoch % self.average_state_every == 0
                and not self.state_averager.averaging_in_progress
            )

            if should_average_state and self.scheduled_state is not None:
                if self.scheduled_state.triggered or self.scheduled_state.done():
                    logger.log(
                        self.status_loglevel,
                        f"Not using pre-scheduled group for state averaging because it"
                        f"was already used elsewhere: {self.scheduled_state}",
                    )
                    self.scheduled_state = None
                self.delay_before_state_averaging.update(task_size=1, interval=time.perf_counter() - _epoch_start_time)

            self.state_averager.step(
                increment_epoch=True,
                wait_for_trigger=wait_for_trigger,
                optimizer_step=should_perform_optimizer_step,
                delay_optimizer_step=self.delay_optimizer_step and should_perform_optimizer_step,
                grad_scaler=grad_scaler,
                averaging_round=should_average_state,
                delay_averaging=self.delay_state_averaging and not self.auxiliary,
                averaging_control=self.scheduled_state if should_average_state else None,
                averaging_opts=dict(timeout=self.averaging_timeout) if should_average_state else None,
            )

            if not should_average_state and self.scheduled_state is not None and not self.scheduled_state.done():
                self.scheduled_state.cancel()
            self.scheduled_state = None

            self.tracker.update_epoch(new_epoch=self.state_averager.local_epoch)
            self._should_check_synchronization_on_update = True
            # the above line ensures that peers check for *strict* synchronization once per epoch

            if not self.client_mode:
                self.state_averager.state_sharing_priority = self.local_epoch

            if self.use_gradient_averaging and not self.auxiliary:
                self.grad_averager.reset_accumulated_grads_()
                if not self.client_mode:
                    self.grad_averager.state_sharing_priority = self.local_epoch

            logger.log(self.status_loglevel, f"Transitioning to epoch {self.local_epoch}")

    def _begin_averaging_gradients(self, grad_scaler: Optional[GradScaler]) -> bool:
        """Begin an all-reduce round to average gradients; return True if succeeded, False if failed"""
        if grad_scaler is not None:
            with grad_scaler.running_global_step():
                assert grad_scaler.unscale_(self)

        began_averaging_gradients = False
        if self.scheduled_grads is not None and (self.scheduled_grads.triggered or self.scheduled_grads.done()):
            logger.log(
                self.status_loglevel,
                f"Not using pre-scheduled group for state averaging because it"
                f"was already used elsewhere: {self.scheduled_state}",
            )
            self.scheduled_grads = None

        elif self.tracker.global_progress.num_peers > 1:
            try:
                self.scheduled_grads = self.grad_averager.step(
                    control=self.scheduled_grads, reset_accumulators=True, wait=False
                )
                began_averaging_gradients = True
            except BaseException as e:
                logger.exception(e)

        if not began_averaging_gradients and self.scheduled_grads is not None and not self.scheduled_grads.done():
            if self.tracker.global_progress.num_peers > 1:
                logger.log(self.status_loglevel, f"Tagging along for a pre-scheduled gradient averaging round")
                self._tag_along_with_zero_weight(self.scheduled_grads)
            else:
                logger.log(self.status_loglevel, f"Skipping pre-scheduled averaging round: there are no other peers")
                self._load_local_gradients_into_optimizer()
                self.scheduled_grads.cancel()
            self.scheduled_grads = None
        return began_averaging_gradients

    def _check_and_accumulate_gradients(self, batch_size: int, grad_scaler: Optional[GradScaler]) -> bool:
        """Check if gradients are valid, accumulate and return True; otherwise, reset and return False"""
        assert not self.use_local_updates and not self.auxiliary
        if grad_scaler is not None and not grad_scaler.are_grads_finite(self):
            logger.log(self.status_loglevel, "Encountered incorrect value in fp16 grads, resetting local gradients")
            self.tracker.report_local_progress(self.local_epoch, samples_accumulated=0)
            self.grad_averager.reset_accumulated_grads_()
            return False

        self.grad_averager.accumulate_grads_(batch_size)
        self.tracker.report_local_progress(self.local_epoch, self.grad_averager.local_samples_accumulated)
        return True

    def _maybe_schedule_gradient_averaging(self) -> None:
        """If next epoch is coming soon, schedule the next gradient averaging round at the estimated end of epoch"""
        assert self.use_gradient_averaging
        if self.tracker.estimated_next_update_time - get_dht_time() <= self.matchmaking_time:
            if self.scheduled_grads is None or self.scheduled_grads.triggered or self.scheduled_grads.done():
                eta_seconds = self.tracker.estimated_next_update_time - get_dht_time()
                eta_seconds = max(eta_seconds, self.grad_averager.matchmaking_kwargs["min_matchmaking_time"])
                logger.log(self.status_loglevel, f"Pre-scheduling gradient averaging round in {eta_seconds:.2f} sec")
                self.scheduled_grads = self.grad_averager.schedule_step(timeout=self.averaging_timeout)

    def _maybe_schedule_state_averaging(self) -> None:
        """If next epoch is coming soon, schedule the next state averaging at estimated parameter averaging start"""
        next_epoch = max(self.local_epoch + 1, self.tracker.global_epoch)
        if next_epoch % self.average_state_every != 0:
            return  # averaging is not performed at this epoch
        if self.state_averager.averaging_in_progress:
            return  # previous run is still in progress
        if self.delay_before_state_averaging.num_updates == 0:
            return  # not enough data to accurately pre-schedule

        estimated_time = self.tracker.estimated_next_update_time
        estimated_time += self.delay_before_state_averaging.ema_seconds_per_sample
        estimated_time += self.state_averager.delay_before_averaging.ema_seconds_per_sample
        eta_seconds_to_averaging = estimated_time - get_dht_time()

        if eta_seconds_to_averaging <= self.matchmaking_time:
            if self.scheduled_state is None or self.scheduled_state.triggered or self.scheduled_state.done():
                min_matchmaking_time = self.state_averager.matchmaking_kwargs["min_matchmaking_time"]
                actual_seconds = max(eta_seconds_to_averaging, min_matchmaking_time)
                logger.log(self.status_loglevel, f"Pre-scheduling state averaging round in {actual_seconds:.2f} sec")
                self.scheduled_state = self.state_averager.schedule_step(
                    gather=next_epoch, timeout=self.averaging_timeout
                )

    def _average_gradients_and_load_into_optimizer(self, maybe_step_control: Optional[StepControl]):
        """Run gradient averaging; on success, feed averaged gradients into optimizer; else, use local gradients"""
        assert self.use_gradient_averaging and maybe_step_control is None or maybe_step_control.triggered
        averaged_gradients = False

        try:
            if maybe_step_control is not None:
                group_info = maybe_step_control.result(self.averaging_timeout)
                logger.log(self.status_loglevel, f"Averaged gradients with {len(group_info)} peers")
                self._load_averaged_gradients_into_optimizer_()
                averaged_gradients = True
            else:
                logger.log(self.status_loglevel, f"Skipped averaging: there are no other peers")
        except BaseException as e:
            logger.log(self.status_loglevel, f"Averaging gradients failed with {repr(e)}")

        if not averaged_gradients:
            self._load_local_gradients_into_optimizer()

    def _load_averaged_gradients_into_optimizer_(self):
        """If required, load averaged gradients into optimizer; otherwise simply notify grad averager"""
        assert self.use_gradient_averaging

        if self.offload_optimizer:
            pass  # averaged gradients are already baked into optimizer, see _make_gradient_averager
        else:
            # copy averaged gradients into optimizer .grad buffers
            optimized_param_groups = self.state_averager.optimizer.param_groups
            optimized_parameters = [param for group in optimized_param_groups for param in group["params"]]
            with torch.no_grad(), self.grad_averager.get_tensors() as averaged_gradients:
                assert len(averaged_gradients) == len(optimized_parameters)
                for opt_param, averaged_grad in zip(optimized_parameters, averaged_gradients):
                    opt_param.grad.copy_(averaged_grad, non_blocking=True)

        self.grad_averager.notify_used_averaged_gradients()

    def _load_local_gradients_into_optimizer(self):
        """Fallback to using local gradients in the optimizer (instead of averaged gradients)"""
        logger.log(self.status_loglevel, f"Proceeding with local gradients")
        self.grad_averager.load_accumulators_into_averager_()
        # note: we load gradients into grad_averager even though there is only one peer because of two reasons:
        # - if offload_optimizer, then we must load gradients onto the CPU gradient buffers used by the optimizer
        # - if not offload_optimizer, we must un-scale gradients (divide them by the number of accumulation steps)
        self._load_averaged_gradients_into_optimizer_()

    def zero_grad(self, set_to_none: bool = False):
        """Reset gradients from model. If reuse_grad_buffers=True, this will raise an error."""
        if self.use_gradient_averaging and self.grad_averager.reuse_grad_buffers:
            raise ValueError(
                f"When running {self.__class__.__name__} with reuse_grad_buffers=True, user should never "
                f"call zero_grad manually. Gradients will be refreshed internally"
            )
        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    pass
                elif set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()

    def _should_load_state_from_peers(self) -> bool:
        """
        If true, peer will discard local progress and attempt to download state from peers.
        This method allows peer to continue training in two cases:
         - peer is on the same epoch as other collaborators - keep training normally
         - peer was on the same epoch and accumulated some grads, but some collaborators
             have just transitioned to the next epoch - this peer should also transition.

        :note: The latter case occurs due to the lack of network synchrony: the first peer that
        detects enough samples will transition to the next step and start counting samples anew.
        Some other peers may take time before they check with DHT and observe that
          - the global epoch is technically one epoch ahead of the current one and
          - the remaining (non-transitioned) peers no longer have target_batch_size between them
        If this is the case, peer should transition to the next epoch and does *not* need to re-load state.
        """
        if self._should_check_synchronization_on_update and self.tracker.fetched_global_progress_this_epoch.is_set():
            self._should_check_synchronization_on_update = False
            return self.local_epoch != self.tracker.global_epoch  # require exact synchronization once per step
        return self.local_epoch < self.tracker.global_epoch - 1  # catch up if a peer just switched to next epoch

    def is_synchronized_with_peers(self) -> bool:
        """Checks whether the current peer is up-to-date with others in terms of the epoch (step) number."""
        return self.local_epoch >= self.tracker.global_epoch - 1

    def load_state_from_peers(self, **kwargs):
        """
        Attempt to load the newest collaboration state from other peers within the same run_id.

        If successful, this will update parameters, optimizer state, local epoch and learning rate schedule in-place.
        """
        # note: we tag along for the next all-reduce because the run may have already started and cancelling it
        # will cause peers to restart matchmaking and may  stall the entire collaboration for a few seconds.
        if self.scheduled_grads is not None and not self.scheduled_grads.done():
            self._tag_along_with_zero_weight(self.scheduled_grads)
            self.scheduled_grads = None
        self.state_averager.step(wait_for_delayed_updates=True)

        with self.tracker.pause_updates():
            while True:
                try:
                    self.state_averager.load_state_from_peers(timeout=self.load_state_timeout, **kwargs)
                    if self.grad_averager is not None:
                        self.grad_averager.load_state_from_peers(timeout=self.load_state_timeout, **kwargs)
                    break
                except KeyboardInterrupt:
                    raise
                except BaseException as e:
                    logger.exception(f"Failed to load state from peers: {e}, retrying ...")
                    continue

            if self.tracker.global_epoch - 1 <= self.local_epoch < self.tracker.global_epoch:
                logger.log(self.status_loglevel, f"Catching up with collaboration step {self.tracker.global_epoch}")
                self.state_averager.local_epoch = self.tracker.global_epoch

            self.tracker.report_local_progress(local_epoch=self.local_epoch, samples_accumulated=0)

            if not self.client_mode:
                self.state_averager.state_sharing_priority = self.local_epoch

            if self.use_gradient_averaging:
                self.grad_averager.reset_accumulated_grads_()
                if not self.client_mode:
                    self.grad_averager.state_sharing_priority = self.local_epoch

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
            f"{self.__class__.__name__} does not support calling add_param_group after creation. "
            f"Please provide all parameter groups at init"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(prefix={self.run_id}, epoch={self.local_epoch})"

    def _tag_along_with_zero_weight(self, control: StepControl):
        """Wait for a running averaging round to finish with zero weight."""
        if not control.triggered:
            control.weight = 0
            control.allow_allreduce()
        if not control.done():
            try:
                control.result(self.averaging_timeout)
            except BaseException as e:
                logger.exception(e)
                if not control.done():
                    control.cancel()

    def shutdown(self):
        logger.log(self.status_loglevel, "Sending goodbye to peers...")
        self.tracker.shutdown(self.shutdown_timeout)
        self.state_averager.step(wait_for_delayed_updates=True)
        for scheduled_round in self.scheduled_grads, self.scheduled_state:
            if scheduled_round is not None:
                if scheduled_round.stage == AveragingStage.LOOKING_FOR_GROUP:
                    scheduled_round.cancel()
                else:
                    self._tag_along_with_zero_weight(scheduled_round)

        logger.log(self.status_loglevel, "Shutting down averagers...")
        self.state_averager.shutdown()
        if self.use_gradient_averaging:
            self.grad_averager.shutdown()
        logger.log(self.status_loglevel, f"{self.__class__.__name__} is shut down")

    def __del__(self):
        if self._parent_pid == os.getpid() and self.is_alive():
            self.shutdown()
