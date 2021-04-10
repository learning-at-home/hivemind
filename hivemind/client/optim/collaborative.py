from __future__ import annotations
import warnings
from dataclasses import dataclass
from threading import Thread, Lock, Event
from typing import Optional

import torch
import numpy as np

from hivemind.dht import DHT
from hivemind.client.optim.base import DecentralizedOptimizerBase
from hivemind.client.averaging.training import TrainingAverager
from hivemind.utils import get_logger, get_dht_time, run_in_background, ValueWithExpiration
from hivemind.client.optim.performance_ema import PerformanceEMA

logger = get_logger(__name__)
LRSchedulerBase = getattr(torch.optim.lr_scheduler, '_LRScheduler', None)

#TODO make sure that the weighted averager will NOT advertize for averaging if it is out of sync!
# warn if expiration is too infrequent (measure mean time between steps)
#TODO make sure we can recover from zero_grad!
#TODO enforce minimum period on reporting progress


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

    def register_step(self):
        self.optimizer_step += 1
        self.samples_accumulated = 0
        self.eta_next_step = float('inf')


class CollaborativeOptimizer(DecentralizedOptimizerBase):
    """
    An optimizer that performs model updates after collaboratively accumulating a target (large) batch size across peers

    These optimizers use DHT to track how much progress did the collaboration make towards target batch size.
    Once enough samples were accumulated, optimizers will compute a weighted average of their statistics.

    :note: calling .step() with this optimizer does not always trigger a model update

    :param opt: a standard pytorch optimizer, preferably a large-batch one such as LAMB, LARS, etc.
    :param dht: a running hivemind.DHT daemon connected to other peers
    :param prefix: a common prefix for all metadata stored by CollaborativeOptimizer in the DHT
    :param target_batch_size: perform optimizer step after all peers collectively accumulate this many samples
    :param batch_size_per_step: before each call to .step, user should accumulate gradients over this many samples
    :param target_group_size: maximum group size for DecentralizedAverager's all-reduce
    :param min_refresh_period: wait for at least this many seconds before fetching new collaboration state
    :param max_refresh_period: wait for at most this many seconds before fetching new collaboration state
    :param default_refresh_period: if no peers are detected, attempt to fetch collaboration state this often (seconds)
    :param expected_drift_peers: assume that this many new peers can join between steps
    :param expected_drift_rate: assumes that this fraction of current collaboration can join/leave between steps
    :note: the expected collaboration drift parameters are used to adjust the frequency with which this optimizer will
      refresh the collaboration-wide statistics (to avoid missing the moment when to run the next step)
    :param bandwidth: peer's network bandwidth for the purpose of load balancing (recommended: internet speed in mbps)
    :param performance_ema_alpha: smoothing value used to estimate this peer's performance (training samples per second)
    :param averaging_expiration: peer's requests for averaging will be valid for this many seconds
    :param metadata_expiration: peer's metadata (e.g. samples processed) is stored onto DHT for this many seconds
    :param averaging_timeout: if an averaging step hangs for this long, it will be cancelled.
    :param scheduler: if specified, use this scheduler to update optimizer learning rate
    :note: if you are using CollaborativeOptimizer with a lr_scheduler, it is recommended to pass this scheduler
      explicitly into this class. Otherwise, scheduler may not be synchronized between peers.
    """

    def __init__(self, opt: torch.optim.Optimizer, *, dht: DHT, prefix: str, target_group_size: int,
                 target_batch_size: int, batch_size_per_step: int, scheduler: Optional[LRSchedulerBase] = None,
                 min_refresh_period: float = 0.5, max_refresh_period: float = 30, default_refresh_period: float = 3,
                 expected_drift_peers: float = 3, expected_drift_rate: float = 0.2, performance_ema_alpha: float = 0.1,
                 averaging_expiration: float = 30, metadata_expiration: float = 30.0,
                 averaging_timeout: Optional[float] = None, **kwargs):
        super().__init__(opt, dht)
        self.prefix, self.scheduler, self.averaging_timeout = prefix, scheduler, averaging_timeout
        self.target_batch_size, self.batch_size_per_step = target_batch_size, batch_size_per_step
        self.min_refresh_period, self.max_refresh_period, self.default_refresh_period =\
            min_refresh_period, max_refresh_period, default_refresh_period
        self.expected_drift_peers, self.expected_drift_rate = expected_drift_peers, expected_drift_rate
        self.averaging_expiration, self.metadata_expiration = averaging_expiration, metadata_expiration
        self.averager = TrainingAverager(opt, average_parameters=True, average_gradients=True, dht=dht,
                                         prefix=f"{self.prefix}_averaging", target_group_size=target_group_size,
                                         allreduce_timeout=self.averaging_timeout, **kwargs)

        # Each step contains {gradient_accumulation_steps} of forward and backward passes
        self.performance_ema = PerformanceEMA(alpha=performance_ema_alpha)

        self.training_progress_key = f"{self.prefix}_progress"
        self.local_samples_accumulated = 0  # a number of local samples accumulated since last optimizer update
        self.local_steps_accumulated = 0  # a number of calls to step() since last optimizer update

        self.collaboration_state = self.fetch_collaboration_state()
        self.lock_collaboration_state, self.state_updated = Lock(), Event()
        self.collaboration_state_updater = Thread(target=self.check_collaboration_state_periodically, daemon=True,
                                                  name=f"{self}.collaboration_state_updater")
        self.collaboration_state_updater.start()

    @property
    def local_step(self) -> int:
        return self.averager.local_step

    @property
    def is_synchronized(self) -> bool:
        return self.local_step >= self.collaboration_state.optimizer_step

    def is_alive(self) -> bool:
        return self.averager.is_alive()

    def load_state_from_peers(self, **kwargs):
        """ Attempt to fetch the newest collaboration state from other peers """
        with self.lock_collaboration_state:
            self.averager.load_state_from_peers(**kwargs)
            self.local_samples_accumulated = self.local_steps_accumulated = 0
            self.update_scheduler()
            self.opt.zero_grad()

    def step(self, batch_size: Optional[int] = None, **kwargs):
        batch_size = self.batch_size_per_step if batch_size is None else batch_size
        if not self.is_synchronized:
            self.load_state_from_peers()
            return

        self.local_samples_accumulated += batch_size
        self.local_steps_accumulated += 1
        self.performance_ema.update(num_processed=self.batch_size_per_step)
        run_in_background(self.report_training_progress)
        if not self.collaboration_state.ready_for_step:
            return

        logger.info("Averaging parameters and gradients with peers...")
        self.collaboration_state = self.fetch_collaboration_state()
        self.state_updated.set()

        if not self.is_synchronized:
            self.load_state_from_peers()
            return

        with self.performance_ema.pause(), self.lock_collaboration_state:
            if self.collaboration_state.num_peers > 1:
                mean_samples_per_worker = self.target_batch_size / self.collaboration_state.num_peers
                weight = self.local_samples_accumulated / mean_samples_per_worker
                self.averager.step(weight=weight, timeout=self.averaging_timeout, **kwargs)
            else:
                logger.info(
                    f"Skipped averaging: collaboration consists of {self.collaboration_state.num_peers} peer(s).")
                self.averager.local_step += 1

            output = self.opt.step()
            self.opt.zero_grad()
            self.local_samples_accumulated = self.local_steps_accumulated = 0
            self.collaboration_state.register_step()
            self.state_updated.set()
            self.update_scheduler()
            logger.info(f"Optimizer step: done!")
            return output

    def report_training_progress(self):
        """ Declare this trainer's current step and the number of batches accumulated towards the next step """
        current_time = get_dht_time()
        local_state_info = [self.local_step, self.local_samples_accumulated, self.performance_ema.samples_per_second,
                            current_time, not self.averager.listen]
        assert self.is_valid_peer_state(local_state_info), local_state_info
        self.dht.store(self.training_progress_key, subkey=self.averager.endpoint, value=local_state_info,
                       expiration_time=current_time + self.metadata_expiration, return_future=True)

    def check_collaboration_state_periodically(self):
        """
        Periodically check the training progress from all peers. Trigger update after target_batch_size total samples
        """
        while self.is_alive:
            time_to_next_update = max(0.0, self.collaboration_state.next_fetch_time - get_dht_time())
            if self.state_updated.wait(time_to_next_update):
                self.state_updated.clear()
                continue  # if state was updated externally, reset timer

            with self.lock_collaboration_state:
                self.collaboration_state = self.fetch_collaboration_state()

    def fetch_collaboration_state(self) -> CollaborationState:
        """ Read performance statistics reported by peers, estimate progress towards next batch """
        response, _expiration = self.dht.get(self.training_progress_key, latest=True) or (None, -float('inf'))
        current_time = get_dht_time()

        if not isinstance(response, dict) or len(response) == 0:
            logger.info(f"Found no active peers: {response}")
            local_eta_next_step = max(0, self.target_batch_size - self.local_steps_accumulated
                                      ) / self.performance_ema.samples_per_second
            return CollaborationState(self.local_step, self.local_samples_accumulated, self.target_batch_size,
                                      num_peers=0, num_clients=0, eta_next_step=current_time + local_eta_next_step,
                                      next_fetch_time=current_time + self.default_refresh_period)

        valid_peer_states = [peer_state.value for peer_state in response.values()
                             if isinstance(peer_state, ValueWithExpiration)
                             and self.is_valid_peer_state(peer_state.value)]

        num_peers = len(valid_peer_states)
        num_clients = sum(is_client for *_, is_client in valid_peer_states)

        global_optimizer_step = self.local_step
        for opt_step, samples_accumulated, samples_per_second, timestep, is_client in valid_peer_states:
            if not is_client:
                global_optimizer_step = max(global_optimizer_step, opt_step)

        total_samples_accumulated = estimated_curent_samples = total_samples_per_second = 0

        for opt_step, samples_accumulated, samples_per_second, timestep, is_client in valid_peer_states:
            total_samples_per_second += samples_per_second
            if opt_step == global_optimizer_step:
                total_samples_accumulated += samples_accumulated
                estimated_curent_samples += samples_accumulated + max(0, current_time - timestep) * samples_per_second
            # note: we deliberately count only valid peers for samples_accumulated, but all peers for performance;
            # the rationale behind this is that outdated peers will synchronize and begin contributing shortly.

        estimated_samples_remaining = self.target_batch_size - estimated_curent_samples
        estimated_time_to_next_step = max(0, estimated_samples_remaining) / total_samples_per_second

        expected_max_peers = max(num_peers + self.expected_drift_peers, num_peers * (1 + self.expected_drift_rate))
        time_to_next_fetch = float(np.clip(a=estimated_time_to_next_step * num_peers / expected_max_peers,
                                           a_min=self.min_refresh_period, a_max=self.max_refresh_period))
        logger.info(f"Collaboration accumulated {total_samples_accumulated} samples from {num_peers} peers; "
                    f"ETA {estimated_time_to_next_step:.2f} seconds (refresh in {time_to_next_fetch:.2f}s.)")
        return CollaborationState(
            global_optimizer_step, total_samples_accumulated, target_batch_size=self.target_batch_size,
            num_peers=num_peers, num_clients=num_clients, eta_next_step=current_time + estimated_time_to_next_step,
            next_fetch_time=current_time + time_to_next_fetch)

    def zero_grad(self, *args, **kwargs):
        warnings.warn("CollaborativeOptimizer.zero_grad is a no-op and doesn't need to be called")

    @staticmethod
    def is_valid_peer_state(state):
        return isinstance(state, (list, tuple)) and len(state) == 5 \
               and all(map(isinstance, state, (int, int, float, float, bool)))

    def update_scheduler(self):
        if self.scheduler:
            while self.scheduler._step_count < self.local_step:
                self.scheduler.step()

    def shutdown(self):
        logger.info("Shutting down averager...")
        self.averager.shutdown()
        logger.info("Sending goodbye to peers...")
        self.dht.store(self.training_progress_key, subkey=self.averager.endpoint, value=None,
                       expiration_time=get_dht_time() + self.metadata_expiration)
        logger.info(f"{self.__class__.__name__} is shut down.")

    def __del__(self):
        self.shutdown()

