import os
import struct
from enum import Enum
from typing import Optional

import numpy as np
import torch

from hivemind.utils import DHTExpiration, MPFuture, get_dht_time, get_logger

logger = get_logger(__name__)


class AveragingStage(Enum):
    IDLE = 0  # still initializing
    LOOKING_FOR_GROUP = 1  # running decentralized matchmaking, can't run allreduce yet
    AWAITING_TRIGGER = 2  # waiting for user to set the trigger that allows running allreduce
    RUNNING_ALLREDUCE = 3  # exchanging tensors with groupmates
    FINISHED = 4  # either done or failed with exception


class StepControl(MPFuture):
    """
    An auxiliary data structure that allows user to control stages and track progress in a single averaging step

    :param scheduled_time: estimated time when averaging should begin. Will be used for scheduling
    :param deadline: if averaging is still in progress at this time, it should be stopped due to TimeoutError
    :param allow_retries: if True, allow running matchmaking and all-reduce again if previous attempt fails
    :param weight: averaging weight, can be changed afterwards
    :param data_for_gather: send this data to all peers in the next group and gather it from groupmates
    """

    # indices for the shared buffer
    _SCHEDULED_TIME, _WEIGHT, _STAGE, _BEGAN_ALLREDUCE = slice(0, 8), slice(8, 16), 16, 17

    def __init__(
        self,
        scheduled_time: DHTExpiration,
        deadline: float,
        allow_retries: bool,
        weight: float,
        data_for_gather: bytes,
    ):
        super().__init__()
        self._data_for_gather, self._deadline, self._allow_retries = data_for_gather, deadline, allow_retries
        self._trigger: Optional[MPFuture] = None
        self._cancel: Optional[MPFuture] = None

        # Buffer contents:
        # scheduled_time (double) | weight (double) | stage (AveragingStage, 1 byte) | began_allreduce: (bool, 1 byte)
        self._shared_buffer = torch.zeros([18], dtype=torch.uint8).share_memory_()
        self.stage = AveragingStage.IDLE
        self.scheduled_time = scheduled_time
        self.weight = weight
        self.began_allreduce = False

    def attach(self, trigger: MPFuture, cancel: MPFuture):
        assert self._trigger is None and self._cancel is None, "Futures are already attached"
        self._trigger, self._cancel = trigger, cancel

    def allow_allreduce(self):
        """Allow averager to begin all-reduce when it finds a group. Meant to be triggered by user."""
        assert self._trigger is not None, "StepControl does not have an attached trigger"
        if self._trigger.done():
            logger.warning("Trigger is already set")
        else:
            self._trigger.set_result(None)

    async def wait_for_trigger(self):
        assert self._trigger is not None, "StepControl does not have an attached trigger"
        await self._trigger

    @property
    def triggered(self) -> bool:
        assert self._trigger is not None, "StepControl does not have an attached trigger"
        return self._trigger.done()

    @property
    def scheduled_time(self) -> DHTExpiration:
        return struct.unpack("d", self._shared_buffer[StepControl._SCHEDULED_TIME].numpy().data)[0]

    @scheduled_time.setter
    def scheduled_time(self, scheduled_time):
        if self.began_allreduce:
            logger.warning("Changing scheduled time has no effect after all-reduce has already started")
        if scheduled_time >= self.deadline:
            logger.warning("Changing scheduled time to after deadline, averaging will likely fail due to timeout")
        struct.pack_into("d", self._shared_buffer[StepControl._SCHEDULED_TIME].numpy().data, 0, float(scheduled_time))

    @property
    def weight(self) -> float:
        return struct.unpack("d", self._shared_buffer[StepControl._WEIGHT].numpy().data)[0]

    @weight.setter
    def weight(self, weight: float):
        assert weight >= 0 and np.isfinite(weight)
        if self.began_allreduce:
            logger.warning("Changing weights has no effect after all-reduce has already started")
        struct.pack_into("d", self._shared_buffer[StepControl._WEIGHT].numpy().data, 0, float(weight))

    @property
    def stage(self) -> AveragingStage:
        return AveragingStage(self._shared_buffer[StepControl._STAGE].item())

    @stage.setter
    def stage(self, stage: AveragingStage):
        if stage == AveragingStage.RUNNING_ALLREDUCE:
            self.began_allreduce = True
        self._shared_buffer[StepControl._STAGE] = stage.value

    @property
    def began_allreduce(self) -> bool:
        return bool(self._shared_buffer[StepControl._BEGAN_ALLREDUCE].item())

    @began_allreduce.setter
    def began_allreduce(self, value: bool):
        self._shared_buffer[StepControl._BEGAN_ALLREDUCE] = int(value)

    @property
    def data_for_gather(self) -> bytes:
        return self._data_for_gather

    @property
    def deadline(self) -> DHTExpiration:
        return self._deadline

    def get_timeout(self) -> Optional[DHTExpiration]:
        return max(0.0, self.deadline - get_dht_time())

    @property
    def allow_retries(self) -> bool:
        return self._allow_retries

    def __getstate__(self):
        return dict(
            super().__getstate__(),
            _trigger=self._trigger,
            _cancel=self._cancel,
            _shared_buffer=self._shared_buffer,
            immutable_params=(self._data_for_gather, self._deadline, self._allow_retries),
        )

    def __setstate__(self, state):
        super().__setstate__(state)
        self._trigger, self._cancel, self._shared_buffer = state["_trigger"], state["_cancel"], state["_shared_buffer"]
        self._data_for_gather, self._deadline, self._allow_retries = state["immutable_params"]

    def __del__(self):
        if os.getpid() == self._origin_pid and not self.triggered:
            logger.warning(
                "Deleted an averaging StepControl, but the step was not triggered. This may cause other "
                "peers to fail an averaging round via TimeoutError."
            )
        super().__del__()

    def cancel(self) -> bool:
        if self._trigger is not None:
            self._trigger.cancel()
        if self._cancel is not None:
            self._cancel.set_result(None)
        return super().cancel()

    async def wait_for_cancel(self):
        """Await for step to be cancelled by the user. Should be called from insider the averager."""
        await self._cancel
