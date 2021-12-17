from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Dict, Iterator, Optional

import numpy as np
import torch
from pydantic import BaseModel, StrictBool, StrictFloat, confloat, conint

from hivemind.dht import DHT
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import BytesWithPublicKey, SchemaValidator
from hivemind.optim.base import DecentralizedOptimizerBase
from hivemind.optim.grad_scaler import HivemindGradScaler
from hivemind.optim.training_averager import TrainingAverager
from hivemind.utils import get_dht_time, get_logger
from hivemind.utils.performance_ema import PerformanceEMA

logger = get_logger(__name__)
LRSchedulerBase = getattr(torch.optim.lr_scheduler, "_LRScheduler", None)


@dataclass(frozen=False)
class CollaborationState:
    optimizer_step: int
    samples_accumulated: int
    target_batch_size: int
    num_peers: int
    num_clients: int
    eta_next_step: float
    next_fetch_time: float

    @property
    def ready_for_step(self):
        return self.samples_accumulated >= self.target_batch_size or get_dht_time() >= self.eta_next_step

    def register_step(self, local_step: int):
        self.optimizer_step = max(local_step, self.optimizer_step)
        self.samples_accumulated = 0
        self.eta_next_step = float("inf")


class TrainingState(BaseModel):
    peer_id: bytes
    step: conint(ge=0, strict=True)
    samples_accumulated: conint(ge=0, strict=True)
    samples_per_second: confloat(ge=0.0, strict=True)
    time: StrictFloat
    client_mode: StrictBool


class TrainingProgressSchema(BaseModel):
    progress: Dict[BytesWithPublicKey, Optional[TrainingState]]


class CollaborativeOptimizer(DecentralizedOptimizerBase):
    """
    An optimizer that performs model updates after collaboratively accumulating a target (large) batch size across peers.

    These optimizers use DHT to track how much progress did the collaboration make towards target batch size.
    Once enough samples were accumulated, optimizers will compute a weighted average of their statistics.

    :note: **For new projects, please use hivemind.Optimizer**. CollaborativeOptimizer is an older version of that.
      Currently, hivemind.Optimizer supports all the features of CollaborativeOptimizer and many advanced ones.
      CollaborativeOptimizer will still be supported for a while, but it will be deprecated in v1.1.0.

    :note: This optimizer behaves unlike regular pytorch optimizers in two ways:

      * calling .step will periodically zero-out gradients w.r.t. model parameters after each step
      * it may take multiple .step calls without updating model parameters, waiting for peers to accumulate enough samples


    :param opt: a standard pytorch optimizer, preferably a large-batch one such as LAMB, LARS, etc.
    :param dht: a running hivemind.DHT daemon connected to other peers
    :param prefix: a common prefix for all metadata stored by CollaborativeOptimizer in the DHT
    :param target_batch_size: perform optimizer step after all peers collectively accumulate this many samples
    :param batch_size_per_step: before each call to .step, user should accumulate gradients over this many samples
    :param min_refresh_period: wait for at least this many seconds before fetching new collaboration state
    :param max_refresh_period: wait for at most this many seconds before fetching new collaboration state
    :param default_refresh_period: if no peers are detected, attempt to fetch collaboration state this often (seconds)
    :param expected_drift_peers: assume that this many new peers can join between steps
    :param expected_drift_rate: assumes that this fraction of current collaboration can join/leave between steps
    :note: The expected collaboration drift parameters are used to adjust the frequency with which this optimizer will
      refresh the collaboration-wide statistics (to avoid missing the moment when to run the next step)
    :param bandwidth: peer's network bandwidth for the purpose of load balancing (recommended: internet speed in mbps)
    :param step_tolerance: a peer can temporarily be delayed by this many steps without being deemed out of sync
    :param performance_ema_alpha: smoothing value used to estimate this peer's performance (training samples per second)
    :param averaging_expiration: peer's requests for averaging will be valid for this many seconds
    :param metadata_expiration: peer's metadata (e.g. samples processed) is stored onto DHT for this many seconds
    :param averaging_timeout: if an averaging step hangs for this long, it will be cancelled.
    :param load_state_timeout: wait for at most this many seconds before giving up on load_state_from_peers
    :param scheduler: if specified, use this scheduler to update optimizer learning rate
    :param reuse_grad_buffers: if True, use model's .grad buffers for gradient accumulation.
      This is more memory efficient, but it requires that the user does *NOT* call model/opt zero_grad at all
    :param accumulate_grads_on: if specified, accumulate gradients on this device. By default, this will use the same
     device as model parameters. One can specify a different device (e.g. 'cpu' vs 'cuda') to save device memory at
     the cost of extra time per step. If reuse_gradient_accumulators is True, this parameter has no effect.
    :param client_mode: if True, runs training without incoming connections, in a firewall-compatible mode
    :param kwargs: additional parameters forwarded to DecentralizedAverager
    :note: If you are using CollaborativeOptimizer with lr_scheduler, it is recommended to pass this scheduler
      explicitly into this class. Otherwise, scheduler may not be synchronized between peers.
    """

    def __init__(
        self,
        opt: torch.optim.Optimizer,
        *,
        dht: DHT,
        prefix: str,
        target_batch_size: int,
        batch_size_per_step: Optional[int] = None,
        scheduler: Optional[LRSchedulerBase] = None,
        min_refresh_period: float = 0.5,
        max_refresh_period: float = 30,
        default_refresh_period: float = 3,
        expected_drift_peers: float = 3,
        expected_drift_rate: float = 0.2,
        performance_ema_alpha: float = 0.1,
        metadata_expiration: float = 60.0,
        averaging_timeout: Optional[float] = None,
        load_state_timeout: float = 600.0,
        step_tolerance: int = 1,
        reuse_grad_buffers: bool = False,
        accumulate_grads_on: Optional[torch.device] = None,
        client_mode: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(opt, dht)

        signature_validator = RSASignatureValidator()
        self._local_public_key = signature_validator.local_public_key
        dht.add_validators([SchemaValidator(TrainingProgressSchema, prefix=prefix), signature_validator])

        if reuse_grad_buffers and accumulate_grads_on is not None:
            logger.warning("Setting 'accumulate_grads_on' has no effect if reuse_grad_buffers=True")
        self.prefix, self.scheduler = prefix, scheduler
        self.target_batch_size, self.batch_size_per_step = target_batch_size, batch_size_per_step
        self.min_refresh_period, self.max_refresh_period, self.default_refresh_period = (
            min_refresh_period,
            max_refresh_period,
            default_refresh_period,
        )
        self.expected_drift_peers, self.expected_drift_rate = expected_drift_peers, expected_drift_rate
        self.averaging_timeout = averaging_timeout
        self.load_state_timeout = load_state_timeout
        self.metadata_expiration = metadata_expiration
        self._grads, self.reuse_grad_buffers, self.accumulate_grads_on = None, reuse_grad_buffers, accumulate_grads_on
        self.client_mode, self.step_tolerance = client_mode, step_tolerance
        self.status_loglevel = logging.INFO if verbose else logging.DEBUG
        self.averager = self._make_averager(**kwargs)

        self._step_supports_amp_scaling = self.reuse_grad_buffers  # enable custom execution with torch GradScaler

        self.training_progress_key = f"{self.prefix}_progress"
        self.local_samples_accumulated = 0  # a number of local samples accumulated since last optimizer update
        self.local_updates_accumulated = 0  # a number of calls to step() since last optimizer update
        self.performance_ema = PerformanceEMA(alpha=performance_ema_alpha)
        self.last_step_time = None

        self.collaboration_state = self._fetch_state()
        self.lock_collaboration_state, self.collaboration_state_updated = Lock(), Event()
        self.lock_local_progress, self.should_report_progress = Lock(), Event()
        self.progress_reporter = Thread(target=self.report_training_progress, daemon=True, name=f"{self}.reporter")
        self.progress_reporter.start()
        self.collaboration_state_updater = Thread(
            target=self.check_collaboration_state_periodically, daemon=True, name=f"{self}.collaboration_state_updater"
        )
        self.collaboration_state_updater.start()

    def _make_averager(self, **kwargs):
        return TrainingAverager(
            self.opt,
            dht=self.dht,
            average_parameters=True,
            average_gradients=True,
            prefix=f"{self.prefix}_averaging",
            allreduce_timeout=self.averaging_timeout,
            client_mode=self.client_mode,
            **kwargs,
        )

    @property
    def local_step(self) -> int:
        return self.averager.local_step

    @property
    def is_synchronized(self) -> bool:
        return self.local_step >= self.collaboration_state.optimizer_step

    @property
    def is_within_tolerance(self) -> bool:
        return self.local_step >= self.collaboration_state.optimizer_step - self.step_tolerance

    def is_alive(self) -> bool:
        return self.averager.is_alive()

    def load_state_from_peers(self, **kwargs):
        """Attempt to fetch the newest collaboration state from other peers"""
        with self.lock_collaboration_state:
            while True:
                try:
                    self.averager.load_state_from_peers(timeout=self.load_state_timeout, **kwargs)
                    break
                except KeyboardInterrupt:
                    raise
                except BaseException as e:
                    logger.exception(f"Failed to load state from peers: {e}, retrying ...")
                    continue

            self.local_samples_accumulated = self.local_updates_accumulated = 0
            self.reset_accumulated_grads_()
            self.update_scheduler()

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict["state"]["collaborative_step"] = self.local_step
        return state_dict

    def load_state_dict(self, state_dict: dict):
        if "collaborative_step" in state_dict["state"]:
            self.averager.local_step = state_dict["state"].pop("collaborative_step")
        return super().load_state_dict(state_dict)

    def step(self, batch_size: Optional[int] = None, grad_scaler: Optional[HivemindGradScaler] = None, **kwargs):
        """
        Report accumulating gradients w.r.t. batch_size additional samples, optionally update model parameters

        :param batch_size: optional override for batch_size_per_step from init
        :param grad_scaler: if amp is enabled, this **must** be a hivemind-aware gradient scaler
        :note: this .step is different from normal pytorch optimizers in several key ways. See __init__ for details.
        """
        if grad_scaler is not None and not isinstance(grad_scaler, HivemindGradScaler):
            raise ValueError("CollaborativeOptimizer requires a hivemind-aware gradient scaler (HivemindGradScaler)")
        if self.batch_size_per_step is None:
            if batch_size is None:
                raise ValueError("Please either set batch_size_per_step parameter at init or when calling .step")
            logger.log(self.status_loglevel, f"Setting default batch_size_per_step to {batch_size}")
            self.batch_size_per_step = batch_size
        batch_size = batch_size if batch_size is not None else self.batch_size_per_step

        if not self.is_synchronized and not self.is_within_tolerance:
            logger.log(self.status_loglevel, "Peer is out of sync")
            self.load_state_from_peers()
            return
        elif not self.is_synchronized and self.is_within_tolerance:
            self.averager.local_step = self.collaboration_state.optimizer_step
            logger.log(self.status_loglevel, f"Catching up with collaboration step {self.local_step}")

        if grad_scaler is not None and not grad_scaler.are_grads_finite(self):
            logger.log(self.status_loglevel, "Encountered incorrect value in fp16 grads, resetting local gradients")
            self.local_samples_accumulated = self.local_steps_accumulated = 0
            self.reset_accumulated_grads_()
            self.should_report_progress.set()
            return

        if self.last_step_time is not None and get_dht_time() - self.last_step_time > self.metadata_expiration:
            logger.warning(
                f"Training step took {get_dht_time() - self.last_step_time}, "
                f"but metadata expired in {self.metadata_expiration} s."
            )

        self.accumulate_grads_(batch_size)

        with self.lock_local_progress:
            self.local_samples_accumulated += batch_size
            self.local_updates_accumulated += 1
            self.performance_ema.update(task_size=batch_size)
            self.should_report_progress.set()

        if not self.collaboration_state.ready_for_step:
            return

        logger.log(self.status_loglevel, f"Beginning global optimizer step #{self.collaboration_state.optimizer_step}")
        with self.performance_ema.pause(), self.lock_collaboration_state:
            self.collaboration_state = self._fetch_state()
            self.collaboration_state_updated.set()

            # divide accumulators by local steps to recover the true average grad w.r.t. local_samples_accumulated
            self.apply_accumulated_grads_(scale_by=1.0 / self.local_updates_accumulated)
            if grad_scaler is not None:
                with grad_scaler.running_global_step():
                    assert grad_scaler.unscale_(self)

            current_step, group_info = self.averager.local_step, None

            if self.collaboration_state.num_peers > 1:
                mean_samples_per_worker = self.target_batch_size / self.collaboration_state.num_peers
                weight = self.local_samples_accumulated / mean_samples_per_worker
                try:
                    group_info = self.averager.step(
                        weight=weight, gather=current_step, timeout=self.averaging_timeout, **kwargs
                    )
                    if group_info:
                        logger.log(self.status_loglevel, f"Averaged tensors successfully with {len(group_info)} peers")

                        # update our current step if we averaged with another peer that was at a more recent step
                        for peer, peer_step in group_info.items():
                            if isinstance(peer_step, int):
                                current_step = max(current_step, peer_step)
                            else:
                                logger.warning(f"Peer {peer} sent malformed data about current step: {peer_step}")

                except BaseException as e:
                    logger.log(self.status_loglevel, f"Skipped averaging: averaging round failed with {repr(e)}")

            else:
                logger.log(
                    self.status_loglevel,
                    f"Skipped averaging: collaboration consists of " f"{self.collaboration_state.num_peers} peer(s)",
                )

            if grad_scaler is not None:
                with grad_scaler.running_global_step():
                    assert grad_scaler.step(self)
            else:
                self.opt.step()

            self.reset_accumulated_grads_()
            self.local_samples_accumulated = self.local_updates_accumulated = 0
            self.collaboration_state.register_step(current_step + 1)
            self.averager.local_step = current_step + 1
            self.collaboration_state_updated.set()
            self.update_scheduler()

            if grad_scaler is not None:
                with grad_scaler.running_global_step():
                    assert grad_scaler.update()

            if not self.averager.client_mode:
                self.averager.state_sharing_priority = self.local_step

        logger.log(self.status_loglevel, f"Optimizer step: done!")

        return group_info

    def step_aux(self, **kwargs):
        """
        Find and assist other peers in averaging without sending local gradients.

        :note: this .step is different from normal pytorch optimizers in several key ways. See __init__ for details.
        """

        if not self.collaboration_state.ready_for_step:
            return

        logger.log(self.status_loglevel, f"Beginning global optimizer step #{self.collaboration_state.optimizer_step}")
        self.collaboration_state = self._fetch_state()
        self.collaboration_state_updated.set()

        with self.lock_collaboration_state:
            current_step, group_info = self.averager.local_step, None

            try:
                group_info = self.averager.step(timeout=self.averaging_timeout, gather=current_step, **kwargs)
                if group_info:
                    logger.log(self.status_loglevel, f"Averaged tensors successfully with {len(group_info)} peers")

                    # update our current step if we averaged with another peer that was at a more recent step
                    for peer, peer_step in group_info.items():
                        if isinstance(peer_step, int):
                            current_step = max(current_step, peer_step)
                        else:
                            logger.warning(f"Peer {peer} sent malformed data about current step: {peer_step}")
            except BaseException as e:
                logger.log(self.status_loglevel, f"Skipped averaging: averaging round failed with {repr(e)}")

            self.collaboration_state.register_step(current_step + 1)
            self.averager.local_step = current_step + 1
            self.collaboration_state_updated.set()

        logger.log(self.status_loglevel, f"Optimizer step: done!")

        return group_info

    def _grad_buffers(self) -> Iterator[torch.Tensor]:
        """pytorch-internal gradient buffers"""
        for param_group in self.opt.param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    yield torch.zeros_like(param)
                else:
                    yield param.grad

    @torch.no_grad()
    def accumulated_grads(self) -> Iterator[torch.Tensor]:
        """local gradient accumulators"""
        if self.reuse_grad_buffers:
            yield from self._grad_buffers()
            return

        if self._grads is None:
            self._grads = [torch.zeros_like(grad, device=self.accumulate_grads_on) for grad in self._grad_buffers()]
        yield from self._grads

    @torch.no_grad()
    def accumulate_grads_(self, batch_size: int):
        """add current gradients to grad accumulators (if any)"""
        if self.reuse_grad_buffers:
            # user is responsible for accumulating gradients in .grad buffers
            assert batch_size == self.batch_size_per_step, "Custom batch size is not supported if reuse_grad_buffers"
        else:
            alpha = float(batch_size) / self.batch_size_per_step
            for grad_buf, grad_acc in zip(self._grad_buffers(), self.accumulated_grads()):
                grad_acc.add_(grad_buf.to(grad_acc.device), alpha=alpha)

    @torch.no_grad()
    def apply_accumulated_grads_(self, scale_by: Optional[float] = None):
        if not self.reuse_grad_buffers:
            for grad_buf, grad_acc in zip(self._grad_buffers(), self.accumulated_grads()):
                grad_buf.copy_(grad_acc.to(grad_buf.device), non_blocking=True)
        if scale_by is not None:
            for grad_buf in self._grad_buffers():
                grad_buf.mul_(scale_by)

    @torch.no_grad()
    def reset_accumulated_grads_(self):
        for grad_buf in self.accumulated_grads():
            grad_buf.zero_()

    def report_training_progress(self):
        """Periodically publish metadata and the current number of samples accumulated towards the next step"""
        while self.is_alive():
            self.should_report_progress.wait()
            self.should_report_progress.clear()
            with self.lock_local_progress:
                current_time = get_dht_time()
                local_state_info = TrainingState(
                    peer_id=self.averager.peer_id.to_bytes(),
                    step=self.local_step,
                    samples_accumulated=self.local_samples_accumulated,
                    samples_per_second=self.performance_ema.samples_per_second,
                    time=current_time,
                    client_mode=self.averager.client_mode,
                )

            self.dht.store(
                key=self.training_progress_key,
                subkey=self._local_public_key,
                value=local_state_info.dict(),
                expiration_time=current_time + self.metadata_expiration,
                return_future=True,
            )

    def check_collaboration_state_periodically(self):
        """
        Periodically check the training progress from all peers. Trigger update after target_batch_size total samples
        """
        while self.is_alive():
            time_to_next_update = max(0.0, self.collaboration_state.next_fetch_time - get_dht_time())
            if self.collaboration_state_updated.wait(time_to_next_update):
                self.collaboration_state_updated.clear()
                continue  # if state was updated externally, reset timer

            with self.lock_collaboration_state:
                self.collaboration_state = self._fetch_state()

    def _fetch_state(self) -> CollaborationState:
        """Read performance statistics reported by peers, estimate progress towards next batch"""
        response, _expiration = self.dht.get(self.training_progress_key, latest=True) or (None, -float("inf"))
        current_time = get_dht_time()

        if not isinstance(response, dict) or len(response) == 0:
            logger.log(self.status_loglevel, f"Found no active peers: {response}")
            samples_left_to_target_batch_size = max(0, self.target_batch_size - self.local_samples_accumulated)
            local_eta_next_step = samples_left_to_target_batch_size / self.performance_ema.samples_per_second

            return CollaborationState(
                self.local_step,
                self.local_samples_accumulated,
                self.target_batch_size,
                num_peers=0,
                num_clients=0,
                eta_next_step=current_time + local_eta_next_step,
                next_fetch_time=current_time + self.default_refresh_period,
            )

        valid_peer_states = [
            TrainingState.parse_obj(peer_state.value)
            for peer_state in response.values()
            if peer_state.value is not None
        ]

        num_peers = len(valid_peer_states)
        num_clients = sum(state.client_mode for state in valid_peer_states)
        global_optimizer_step = self.local_step
        for state in valid_peer_states:
            if not state.client_mode:
                global_optimizer_step = max(global_optimizer_step, state.step)

        total_samples_accumulated = estimated_current_samples = total_samples_per_second = 0

        for state in valid_peer_states:
            total_samples_per_second += state.samples_per_second
            if state.step == global_optimizer_step:
                total_samples_accumulated += state.samples_accumulated
                estimated_current_samples += (
                    state.samples_accumulated + max(0, current_time - state.time) * state.samples_per_second
                )
            # note: we deliberately count only valid peers for samples_accumulated, but all peers for performance;
            # the rationale behind this is that outdated peers will synchronize and begin contributing shortly.

        estimated_samples_remaining = self.target_batch_size - estimated_current_samples
        estimated_time_to_next_step = max(0, estimated_samples_remaining) / total_samples_per_second

        expected_max_peers = max(num_peers + self.expected_drift_peers, num_peers * (1 + self.expected_drift_rate))
        time_to_next_fetch = float(
            np.clip(
                a=estimated_time_to_next_step * num_peers / expected_max_peers,
                a_min=self.min_refresh_period,
                a_max=self.max_refresh_period,
            )
        )
        logger.log(
            self.status_loglevel,
            f"{self.prefix} accumulated {total_samples_accumulated} samples from "
            f"{num_peers} peers for step #{global_optimizer_step}. "
            f"ETA {estimated_time_to_next_step:.2f} sec (refresh in {time_to_next_fetch:.2f} sec)",
        )
        return CollaborationState(
            global_optimizer_step,
            total_samples_accumulated,
            target_batch_size=self.target_batch_size,
            num_peers=num_peers,
            num_clients=num_clients,
            eta_next_step=current_time + estimated_time_to_next_step,
            next_fetch_time=current_time + time_to_next_fetch,
        )

    def zero_grad(self, *args, **kwargs):
        if self.reuse_grad_buffers:
            raise ValueError(
                f"When running {self.__class__.__name__} with reuse_grad_buffers=True, user should never "
                f"call zero_grad manually. Gradients will be refreshed internally."
            )
        return self.opt.zero_grad(*args, **kwargs)

    def update_scheduler(self):
        if self.scheduler:
            while self.scheduler._step_count < self.local_step:
                self.scheduler.step()

    def shutdown(self):
        logger.debug("Shutting down averager...")
        self.averager.shutdown()
        logger.debug("Sending goodbye to peers...")
        self.dht.store(
            self.training_progress_key,
            subkey=self._local_public_key,
            value=None,
            expiration_time=get_dht_time() + self.metadata_expiration,
        )
        logger.debug(f"{self.__class__.__name__} is shut down")

    def __del__(self):
        self.shutdown()
