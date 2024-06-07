import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from pydantic import BaseModel, StrictBool, StrictFloat, confloat, conint

from hivemind.dht import DHT
from hivemind.dht.schema import BytesWithPublicKey, RSASignatureValidator, SchemaValidator
from hivemind.utils import DHTExpiration, ValueWithExpiration, enter_asynchronously, get_dht_time, get_logger
from hivemind.utils.crypto import RSAPrivateKey
from hivemind.utils.performance_ema import PerformanceEMA

logger = get_logger(__name__)


@dataclass(frozen=False)
class GlobalTrainingProgress:
    epoch: int
    samples_accumulated: int
    target_batch_size: int
    num_peers: int
    num_clients: int
    eta_next_epoch: float
    next_fetch_time: float


class LocalTrainingProgress(BaseModel):
    peer_id: bytes
    epoch: conint(ge=0, strict=True)
    samples_accumulated: conint(ge=0, strict=True)
    samples_per_second: confloat(ge=0.0, strict=True)
    time: StrictFloat
    client_mode: StrictBool


class TrainingProgressSchema(BaseModel):
    progress: Dict[BytesWithPublicKey, Optional[LocalTrainingProgress]]


class ProgressTracker(threading.Thread):
    """
    Auxiliary class that keeps track of local & global training progress, measured in epochs.
    An epoch can be incremented after collaboration accumulates a said number of gradients (target_batch_size).
    Similarly to pytorch LR scheduler, epoch can be incremented on a single optimizer update or many local updates.

    :param min_refresh_period: wait for at least this many seconds before fetching new collaboration state
    :param max_refresh_period: wait for at most this many seconds before fetching new collaboration state
    :param default_refresh_period: if no peers are detected, attempt to fetch collaboration state this often (seconds)
    :param expected_drift_peers: assume that this many new peers can join between epochs
    :param expected_drift_rate: assumes that this fraction of current collaboration can join/leave between epochs
    :note: The expected collaboration drift parameters are used to adjust the frequency with which this optimizer will
      refresh the collaboration-wide statistics (to avoid missing the moment when peers transition to the next epoch)
    :param performance_ema_alpha: smoothing value used to estimate this peer's performance (samples per second)
    :param metadata_expiration: peer's metadata (e.g. samples processed) is stored onto DHT for this many seconds

    Example:

    >>> tracker = ProgressTracker(hivemind.DHT(...), prefix="my_experiment_with_several_peers", target_batch_size=100)
    >>> local_epoch, local_samples = 0, 0
    >>> while True:
    >>>     accumulate_gradients(batch_size=32)
    >>>     local_samples += 32
    >>>     tracker.report_local_progress(local_epoch, local_samples)
    >>>     if local_epoch < tracker.global_progress.epoch:
    >>>         download_state_from_peers()  # if peer is out of sync, synchronize it with the swarm
    >>>     if tracker.accumulated_enough_samples:
    >>>         with tracker.pause_updates():
    >>>             aggregate_gradients_with_peers()
    >>>             update_model_parameters()
    >>>             local_epoch = tracker.update_epoch(local_epoch + 1)
    >>>             local_samples = 0
    """

    def __init__(
        self,
        dht: DHT,
        prefix: str,
        target_batch_size: int,
        *,
        client_mode: Optional[bool] = None,
        min_refresh_period: float = 0.5,
        max_refresh_period: float = 10,
        default_refresh_period: float = 3,
        expected_drift_peers: float = 3,
        expected_drift_rate: float = 0.2,
        performance_ema_alpha: float = 0.1,
        metadata_expiration: float = 60.0,
        status_loglevel: int = logging.DEBUG,
        private_key: Optional[RSAPrivateKey] = None,
        daemon: bool = True,
        start: bool,
    ):
        client_mode = client_mode if client_mode is not None else dht.client_mode
        self.dht, self.prefix, self.client_mode = dht, prefix, client_mode
        self.training_progress_key = f"{self.prefix}_progress"
        self.target_batch_size = target_batch_size
        self.min_refresh_period, self.max_refresh_period = min_refresh_period, max_refresh_period
        self.default_refresh_period = default_refresh_period
        self.expected_drift_peers, self.expected_drift_rate = expected_drift_peers, expected_drift_rate
        self.status_loglevel = status_loglevel
        self.performance_ema = PerformanceEMA(alpha=performance_ema_alpha)
        self.metadata_expiration = metadata_expiration

        signature_validator = RSASignatureValidator(private_key)
        self._local_public_key = signature_validator.local_public_key
        dht.add_validators([SchemaValidator(TrainingProgressSchema, prefix=prefix), signature_validator])

        # report the collaboration progress periodically or in background
        self.local_progress = self._get_local_progress(local_epoch=0, samples_accumulated=0)
        metadata, _expiration = self.dht.get(self.training_progress_key, latest=True) or (None, -float("inf"))
        self.global_progress = self._parse_swarm_progress_data(metadata)
        self.lock_global_progress, self.global_state_updated = threading.Lock(), threading.Event()
        self.should_report_progress, self.fetched_global_progress_this_epoch = threading.Event(), threading.Event()
        self.shutdown_triggered, self.shutdown_complete = threading.Event(), threading.Event()
        super().__init__(name=f"{self.__class__.__name__}({self.prefix})", daemon=daemon)
        if start:
            self.start()

    @property
    def global_epoch(self) -> int:
        return self.global_progress.epoch

    @property
    def ready_to_update_epoch(self) -> bool:
        """Whether or not this peer can increment epoch right away."""
        return (
            self.global_epoch > self.local_progress.epoch
            or self.global_progress.samples_accumulated >= self.target_batch_size
            or get_dht_time() >= self.global_progress.eta_next_epoch
        )

    @property
    def estimated_next_update_time(self) -> DHTExpiration:
        """Estimate (absolute) time when this peer should increment epoch"""
        if self.ready_to_update_epoch:
            return get_dht_time()
        return self.global_progress.eta_next_epoch

    def _get_local_progress(self, local_epoch: int, samples_accumulated: int):
        return LocalTrainingProgress(
            peer_id=self.dht.peer_id.to_bytes(),
            epoch=local_epoch,
            samples_accumulated=samples_accumulated,
            samples_per_second=self.performance_ema.samples_per_second,
            time=get_dht_time(),
            client_mode=self.client_mode,
        )

    def report_local_progress(self, local_epoch: int, samples_accumulated: int, update_global_samples: bool = True):
        """Update the number of locally accumulated samples and notify to other peers about this."""
        extra_samples = samples_accumulated - self.local_progress.samples_accumulated
        if update_global_samples and local_epoch == self.local_progress.epoch == self.global_progress.epoch:
            self.global_progress.samples_accumulated += extra_samples
            # note: the above line can decrease the number of samples, e.g. if forced to reset due to overflow

        if extra_samples > 0:
            self.performance_ema.update(task_size=extra_samples)
            logger.debug(f"Updated performance EMA: {self.performance_ema.samples_per_second:.5f}")
        else:
            logger.debug("Resetting performance timestamp to current time (progress was reset)")
            self.performance_ema.reset_timer()

        self.local_progress = self._get_local_progress(local_epoch, samples_accumulated)
        self.should_report_progress.set()

    @contextlib.contextmanager
    def pause_updates(self):
        """Temporarily stop progress tracker from updating global training state"""
        with self.lock_global_progress, self.performance_ema.pause():
            yield

    def update_epoch(self, new_epoch: Optional[int] = None) -> int:
        """Update the local epoch, reset the number of sample accumulated, reset local progress, return new epoch"""
        assert self.lock_global_progress.locked(), "ProgressTracker must be paused when incrementing epoch"
        if new_epoch is None:
            new_epoch = self.local_progress.epoch + 1
        if new_epoch > self.global_progress.epoch:
            self.global_progress.epoch = new_epoch
            self.global_progress.samples_accumulated = 0
            self.global_progress.eta_next_epoch = float("inf")
        self.report_local_progress(new_epoch, samples_accumulated=0)
        self.fetched_global_progress_this_epoch.clear()
        return new_epoch

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(asyncio.gather(self._progress_reporter(), self._progress_fetcher()))
        self.shutdown_complete.set()

    async def _progress_reporter(self):
        """Periodically publish metadata and the current number of samples accumulated towards the next epoch"""
        last_report_time = -float("inf")
        last_report_epoch = -float("inf")
        store_task = None
        try:
            while not self.shutdown_triggered.is_set():
                wait_timeout = max(0.0, last_report_time - get_dht_time() + self.metadata_expiration / 2)
                logger.debug(f"Will report progress again in {wait_timeout} seconds or on user command")
                await asyncio.get_event_loop().run_in_executor(None, self.should_report_progress.wait, wait_timeout)
                if self.should_report_progress.is_set():
                    logger.debug(f"Progress update triggered by report_local_progress")
                    self.should_report_progress.clear()
                else:
                    logger.debug(f"Progress update triggered by metadata_expiration")

                local_progress = self.local_progress
                last_report_time = get_dht_time()
                if local_progress.samples_accumulated > 0:
                    last_report_epoch = self.global_epoch

                if last_report_epoch >= self.global_epoch - 1:
                    # report progress if peer is synchronized and actively reporting samples. Do not report aux peers.
                    store_task = asyncio.create_task(
                        asyncio.wait_for(
                            self.dht.store(
                                key=self.training_progress_key,
                                subkey=self._local_public_key,
                                value=local_progress.dict(),
                                expiration_time=last_report_time + self.metadata_expiration,
                                return_future=True,
                            ),
                            timeout=self.metadata_expiration,
                        )
                    )
        finally:
            logger.log(self.status_loglevel, f"No longer reporting progress for {self.prefix}")
            if store_task is not None:
                store_task.cancel()

    async def _progress_fetcher(self):
        """
        Periodically check the training progress from all peers. Trigger update after target_batch_size total samples
        """
        loop = asyncio.get_event_loop()
        shutdown_checker = asyncio.create_task(
            asyncio.wait_for(loop.run_in_executor(None, self.shutdown_triggered.wait), None)
        )

        async def _fetch_progress_unless_shutdown_triggered():
            """Fetch progress, avoid deadlocks if DHT was shut down before this get finished."""
            getter = asyncio.create_task(
                asyncio.wait_for(self.dht.get(self.training_progress_key, latest=True, return_future=True), None)
            )
            await asyncio.wait({getter, shutdown_checker}, return_when=asyncio.FIRST_COMPLETED)
            if self.shutdown_triggered.is_set():
                return
            return await getter

        try:
            while not self.shutdown_triggered.is_set():
                time_to_next_update = max(0.0, self.global_progress.next_fetch_time - get_dht_time())
                state_updated_externally = await loop.run_in_executor(
                    None, self.global_state_updated.wait, time_to_next_update
                )
                if state_updated_externally:
                    self.global_state_updated.clear()
                    continue

                async with enter_asynchronously(self.lock_global_progress):
                    maybe_metadata = await _fetch_progress_unless_shutdown_triggered()
                    if self.shutdown_triggered.is_set():
                        break
                    metadata = maybe_metadata.value if isinstance(maybe_metadata, ValueWithExpiration) else None
                    self.global_progress = self._parse_swarm_progress_data(metadata)
                    self.fetched_global_progress_this_epoch.set()

        finally:
            logger.log(self.status_loglevel, f"No longer fetching {self.training_progress_key}")

    def _parse_swarm_progress_data(self, metadata: TrainingProgressSchema) -> GlobalTrainingProgress:
        """Read performance statistics reported by peers, estimate progress towards next batch"""
        current_time = get_dht_time()

        if not isinstance(metadata, dict) or len(metadata) == 0:
            logger.log(self.status_loglevel, f"Found no active peers: {metadata}")
            samples_remaining_to_next_epoch = max(0, self.target_batch_size - self.local_progress.samples_accumulated)
            local_eta_next_epoch = samples_remaining_to_next_epoch / self.performance_ema.samples_per_second

            return GlobalTrainingProgress(
                self.local_progress.epoch,
                self.local_progress.samples_accumulated,
                self.target_batch_size,
                num_peers=0,
                num_clients=0,
                eta_next_epoch=current_time + local_eta_next_epoch,
                next_fetch_time=current_time + self.default_refresh_period,
            )

        valid_peer_entries = [
            LocalTrainingProgress.parse_obj(peer_state.value)
            for peer_state in metadata.values()
            if peer_state.value is not None
        ]

        num_peers = len(valid_peer_entries)
        num_clients = sum(peer.client_mode for peer in valid_peer_entries)

        global_epoch = self.local_progress.epoch
        for peer in valid_peer_entries:
            if not peer.client_mode:
                global_epoch = max(global_epoch, peer.epoch)

        total_samples_accumulated = estimated_current_samples = 0
        total_samples_per_second = self.performance_ema.eps

        for peer in valid_peer_entries:
            total_samples_per_second += peer.samples_per_second
            if peer.epoch == global_epoch:
                total_samples_accumulated += peer.samples_accumulated
                estimated_current_samples += (
                    peer.samples_accumulated + max(0.0, current_time - peer.time) * peer.samples_per_second
                )
            # note: we deliberately count only valid peers for samples_accumulated, but all peers for performance;
            # the rationale behind this is that outdated peers will synchronize and begin contributing shortly.

        estimated_samples_remaining = self.target_batch_size - estimated_current_samples
        estimated_time_to_next_epoch = max(0, estimated_samples_remaining) / total_samples_per_second

        expected_max_peers = max(num_peers + self.expected_drift_peers, num_peers * (1 + self.expected_drift_rate))
        time_to_next_fetch = float(
            np.clip(
                a=estimated_time_to_next_epoch * num_peers / expected_max_peers,
                a_min=self.min_refresh_period,
                a_max=self.max_refresh_period,
            )
        )
        logger.log(
            self.status_loglevel,
            f"{self.prefix} accumulated {total_samples_accumulated} samples for epoch #{global_epoch} from "
            f"{num_peers} peers. ETA {estimated_time_to_next_epoch:.2f} sec (refresh in {time_to_next_fetch:.2f} sec)",
        )
        return GlobalTrainingProgress(
            global_epoch,
            total_samples_accumulated,
            target_batch_size=self.target_batch_size,
            num_peers=num_peers,
            num_clients=num_clients,
            eta_next_epoch=current_time + estimated_time_to_next_epoch,
            next_fetch_time=current_time + time_to_next_fetch,
        )

    def shutdown(self, timeout: Optional[float] = None):
        """Permanently disable all tracking activity"""
        self.shutdown_triggered.set()
        self.should_report_progress.set()
        self.global_state_updated.set()
        self.shutdown_complete.wait(timeout)
        self.dht.store(
            self.training_progress_key,
            subkey=self._local_public_key,
            value=None,
            expiration_time=get_dht_time() + self.metadata_expiration,
            return_future=True,
        )

    def __del__(self):
        if self.is_alive():
            self.shutdown()
