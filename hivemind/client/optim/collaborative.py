from __future__ import annotationsns
import time
from threading import Thread, Lock, Event
from typing import Optional

import torch

from hivemind import run_in_background
from hivemind.dht import DHT
from hivemind.client.averaging import DecentralizedAverager
from hivemind.client.optim.base import DecentralizedOptimizerBase
from hivemind.utils import get_logger, get_dht_time
from hivemind.client.optim.performance_ema import PerformanceEMA

logger = get_logger(__name__)


@dataclass(frozen=False)
class CollaborationState:
    optimizer_step: int
    samples_accumulated: int
    target_batch_size: int
    num_peers: int
    eta_next_step: float
    next_fetch_time: float

    @property
    def should_perform_step(self):
        return self.samples_accumulated >= self.target_batch_size or hivemind.get_dht_time() >= self.eta_next_step

    def register_step(self):
        self.optimizer_step += 1
        self.samples_accumulated = 0
        self.eta_next_step = float('inf')


class CollaborativeOptimizer(DecentralizedOptimizerBase):
    """
    An optimizer that performs model updates after collaboratively accumulating a target (large) batch size across all peers.
    TODO: explain that this thing "watching when to perform average"

    :param opt: a standard pytorch optimizer, preferably a large-batch one such as LAMB, LARS, etc.
    :param dht: a running hivemind.DHT daemon connected to other peers
    :param prefix: a common prefix for all metadata stored by CollaborativeOptimizer in the DHT
    :param target_batch_size: perform optimizer step after all peers collectively accumulate this many samples
    :param batch_size_per_step: before each call to CollaborativeOptimizer.step, user should accumulate gradients over this many samples
    :param target_group_size: maximum group size for DecentralizedAverager's all-reduce
    :param min_refresh_period: wait for at least this many seconds before fetching new collaboration state
    :param max_refresh_period: wait for at most this many seconds before fetching new collaboration state
    :param default_refresh_period: if no peers are detected, attempt to fetch collaboration state this often (seconds)
    :param expected_collaboration_drift_peers: assume that this many new peers can join between steps
    :param expected_collaboration_drift_rate: assumes that this fraction of current collaboration can join/leave between steps
    :note: the expected collaboration drift variables are used to adjust the frequency with which this peer checks for updates
    :param bandwidth: peer's network bandwidth for the purpose of load balancing in DecentralizedAverager (recommended: internet speed in mbps)
    :param performance_ema_alpha: smoothing value used to estimate this peer's performance (training samples per second)
    :param timeout: if an averaging step hangs for this long, it will be cancelled.

    #TODO inherit client mode from DHT,
    #TODO allreduce timeout

    """

    def __init__(self, opt: torch.optim.Optimizer, dht: DHT, prefix: str,
                 target_batch_size: int, batch_size_per_step: int, target_group_size: int,
                 min_refresh_period: float = 0.5, max_refresh_period: float = 30, default_refresh_period: float = 3,
                 expected_collaboration_drift_peers: float = 3, expected_collaboration_drift_rate=0.2,
                 bandwidth: float = 1000.0, performance_ema_alpha: float = 0.1,
                 averaging_step_timeout: Optional[float] = None, verbose: bool = False, **kwargs):
        super().__init__(opt, dht)

        with torch.no_grad():
            averaged_tensors = tuple(p.cpu().float().clone().requires_grad_(False)
                                     for group in self.param_groups for p in group['params'])
        self.averager = ParameterAndGradientAverager(opt, dht=dht) #TODO args)

        self.batch_size_per_step = batch_size_per_step
        self.target_batch_size = target_batch_size
        self.averaging_step_timeout = averaging_step_timeout

        self.opt = opt

        # Each step contains {gradient_accumulation_steps} of forward and backward passes
        self.performance_ema = PerformanceEMA(alpha=performance_ema_alpha)

        self.local_samples_accumulated = 0  # a number of local samples accumulated since last optimizer update
        self.local_steps_accumulated = 0  # a number of calls to step() since last optimizer update

        self.collaboration_state = self.fetch_collaboration_state()
        self.is_synchronized = False

        run_in_background(self.check_collaboration_state_periodically)

    def load_state_from_peers(self):
        # TODO: reset local samples accumulated
        try:
            raise NotImlementedError("")
            self.local_samples_accumulated = 0
            self.local_steps_accumulated = 0
        finally:
            self.is_synchronized = True

    def step(self, *args, **kwargs):
        # This is OK
        if not self.is_synchronized:
            logger.info(f"TODO")
            self.load_state_from_peers()
            return

        self.local_samples_accumulated += self.batch_size_per_step
        self.local_steps_accumulated += 1
        self.performance_ema.update(num_processed=self.batch_size_per_step)
        run_in_background(self.report_training_progress)

        if not self.collaboration_state.should_perform_step:
            return

        self.averager_step()
        self.opt.step()
        self.opt.zero_grad()

    def averager_step(self):
        logger.info("Averaging parameters and gradients with peers...")
        collaboration = self.fetch_collaboration_state()
        if collaboration.num_peers <= 1:
            logger.info(f"Skipping averaging: collaboration consists of {collaboration.num_peers} peers.")
        return

        mean_samples_per_worker = collaboration.samples_accumulated / collaboration.num_peers

        weight = self.local_samples_accumulated / mean_samples_per_worker

        # TODO take parameters from optimizers
        #
        #  local_tensors = [tensor for tensor in .parameters()]

        local_tensors.extend([tensor.grad for tensor in self.opt.parameters()])

        # TODO: extract this and move to ParameterAndGradientAverager

        self.local_samples_accumulated = self.local_steps_accumulated = 0
        self.collaboration_state.register_step()
        logger.info(f"Optimizer step: done!")


    def check_collaboration_state_periodically(self):
        """
        Periodically check the training progress from all peers. Trigger update after target_batch_size total samples
        """
        while self.is_alive:
            with self.lock:
                self.collaboration_state = self.fetch_collaboration_state()
            time.sleep(max(0, self.collaboration_state.next_fetch_time - hivemind.get_dht_time()))


    def fetch_collaboration_state(self) -> CollaborationState:
        """ Read performance statistics reported by peers, estimate progress towards next batch """
        target_batch_size = self.collaboration_args.target_batch_size
        response, _expiration = self.dht.get(self.training_progess_key, latest=True) or (None, -float('inf'))
        current_time = hivemind.get_dht_time()

        if not isinstance(response, dict) or len(response) == 0:
            logger.warning(f"Found no active peers: {response}")
            local_eta_next_step = max(0,
                                      target_batch_size - self.local_steps_accumulated) / self.performance_ema.samples_per_second
            return CollaborationState(self.local_step, self.local_samples_accumulated, target_batch_size, 0,
                                      eta_next_step=current_time + local_eta_next_step,
                                      next_fetch_time=current_time + self.collaboration_args.default_refresh_period)

        valid_peer_states = [peer_state.value for peer_state in response.values()
                             if isinstance(peer_state, ValueWithExpiration)
                             and self.is_valid_peer_state(peer_state.value)]
        global_optimizer_step = max(self.local_step, max(step for step, *_ in valid_peer_states))

        num_peers = len(valid_peer_states)
        total_samples_accumulated = estimated_curent_samples = total_samples_per_second = 0

        for opt_step, samples_accumulated, samples_per_second, timestep in valid_peer_states:
            total_samples_per_second += samples_per_second
            if opt_step == global_optimizer_step:
                total_samples_accumulated += samples_accumulated
                estimated_curent_samples += samples_accumulated + max(0, current_time - timestep) * samples_per_second
            # note: we deliberately count only valid peers for samples_accumulated, but all peers for performance;
            # the rationale behind this is that outdated peers will synchronize and begin contributing shortly.

        estimated_time_to_next_step = max(0, target_batch_size - estimated_curent_samples) / total_samples_per_second

        expected_max_peers = max(num_peers + self.collaboration_args.expected_collaboration_drift_peers,
                                 num_peers * (1 + self.collaboration_args.expected_collaboration_drift_rate))
        time_to_next_fetch = float(np.clip(a=estimated_time_to_next_step * num_peers / expected_max_peers,
                                           a_min=self.collaboration_args.min_refresh_period,
                                           a_max=self.collaboration_args.max_refresh_period))
        logger.info(f"Collaboration accumulated {total_samples_accumulated} samples from {num_peers} peers; "
                    f"ETA {estimated_time_to_next_step:.2f} seconds (refresh in {time_to_next_fetch:.2f}s.)")
        return CollaborationState(global_optimizer_step, total_samples_accumulated, target_batch_size=target_batch_size,
                                  num_peers=num_peers, eta_next_step=current_time + estimated_time_to_next_step,
                                  next_fetch_time=current_time + time_to_next_fetch)

    def zero_grad(self, *args, **kwargs):
        warnings.warn("CollaborativeOptimizer.zero_grad is a no-op and doesn't need to be called")


    def __del__(self):
        raise NotImplementedError()


    def shutdown(self):
        raise NotImplementedError()
