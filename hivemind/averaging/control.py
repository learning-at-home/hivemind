import struct
from enum import Enum
from typing import Optional

import numpy as np
import torch

from hivemind.utils import MPFuture, DHTExpiration


class AveragingStage(Enum):
    IDLE = 0               # still initializing
    LOOKING_FOR_GROUP = 1  # running decentralized matchmaking, can't run allreduce yet
    AWAITING_TRIGGER = 2   # waiting for user to set the trigger that allows running allreduce
    RUNNING_ALLREDUCE = 3  # exchanging tensors with groupmates
    FINISHED = 4           # either done or failed with exception


class StepControl(MPFuture):
    """
    An auxiliary data structure that allows user to control stages and track progress in a single averaging step
    TODO description
    :param gather_binary: optionally send this data to all peers in the next group and gather it from groupmates
    :param timeout: maximum time that may be spent looking for group (does not include allreduce itself)
    :returns: an assembled group if successful, None if failed; does NOT perform the actual averaging


    """
    def __init__(self, scheduled_time: DHTExpiration, weight: float, wait_for_trigger: bool,
                 gather_binary: bytes, timeout: Optional[float], allow_retries: bool):
        super().__init__()
        self._gather_binary, self._timeout, self._allow_retries = gather_binary, timeout, allow_retries
        self._trigger: Optional[MPFuture] = None
        if not wait_for_trigger:
            self.allow_allreduce()
        self._metadata = torch.zeros([18], dtype=torch.uint8).share_memory_()
        self.stage = AveragingStage.IDLE
        self.scheduled_time = scheduled_time
        self.weight = weight
        self.can_modify = True

    def _attach_trigger(self, trigger: MPFuture):
        assert self._trigger is None
        self._trigger = trigger

    def allow_allreduce(self):
        """Allows averager to begin allreduce when it finds a group."""
        self._trigger.set_result(None)

    async def wait_for_trigger(self):
        await self._trigger

    @property
    def scheduled_time(self) -> DHTExpiration:
        return struct.unpack('d', self._metadata[0:8].numpy().data)[0]

    @scheduled_time.setter
    def scheduled_time(self, scheduled_time):
        assert self.can_modify, "cannot change scheduling after all-reduce has already started"
        #TODO check that scheduled time is still within timeout
        struct.pack_into('d', self._metadata[0:8].numpy().data, 0, float(scheduled_time))

    @property
    def weight(self) -> float:
        return struct.unpack('d', self._metadata[8:16].numpy().data)[0]

    @weight.setter
    def weight(self, weight: float):
        assert self.can_modify, "cannot change weights after all-reduce has already started"
        assert weight >= 0 and np.isfinite(weight)
        struct.pack_into('d', self._metadata[8:16].numpy().data, 0, float(weight))

    @property
    def stage(self) -> AveragingStage:
        return AveragingStage(self._metadata[16].item())

    @stage.setter
    def stage(self, stage: AveragingStage):
        if stage == AveragingStage.RUNNING_ALLREDUCE:
            self.can_modify = False
        self._metadata[16] = stage.value

    @property
    def can_modify(self) -> bool:
        return bool(self._metadata[17].item())

    @can_modify.setter
    def can_modify(self, value: bool):
        self._metadata[17] = int(value)

    @property
    def gather_binary(self) -> bytes:
        return self._gather_binary

    @property
    def timeout(self) -> DHTExpiration:
        return self.timeout

    @property
    def allow_retries(self) -> bool:
        return self._allow_retries

    def cancel(self) -> bool:
        self._trigger.cancel()
        return self.cancel()
