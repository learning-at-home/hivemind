import asyncio
import re
import random
from typing import Optional, List, Tuple

import numpy as np

from hivemind.dht import DHT
from hivemind.client.averaging.allreduce import AllReduceRunner
from hivemind.utils import get_logger, Endpoint, DHTExpiration

GroupKey = str
GROUP_PATTERN = re.compile('^(([^.])+)[.]0b[01]*$')  # e.g. bert_exp4_averaging.0b01001101
logger = get_logger(__name__)


def is_valid_group(maybe_group: str) -> bool:
    """ A group identifier must contain group type, followed by one or more .-separated indices, and any ?metadata"""
    return bool(GROUP_PATTERN.fullmatch(maybe_group))


class GroupKeyManager:
    """
    Utility class that declares and retrieves averaging-related keys
    """
    RESERVED_KEY_FOR_NBITS = '::NBITS'

    def __init__(self, dht: DHT, endpoint: Endpoint, prefix: str, initial_group_bits: Optional[str],
                 target_group_size: int, insufficient_size: Optional[int] = None, excessive_size: Optional[int] = None,
                 nbits_expiration: float = 600):
        assert initial_group_bits is None or all(bit in '01' for bit in initial_group_bits)
        self.dht, self.endpoint, self.prefix, self.group_bits = dht, endpoint, prefix, initial_group_bits or ''
        self.target_group_size = target_group_size
        self.insufficient_size = insufficient_size or max(1, target_group_size // 2)
        self.excessive_size = excessive_size or target_group_size * 3
        self.nbits_expiration = nbits_expiration

    @property
    def current_key(self) -> GroupKey:
        return f"{self.prefix}.0b{self.group_bits}"

    async def update_key_on_success(self, allreduce_group: AllReduceRunner, is_leader: bool = True):
        """ this function is triggered every time an averager finds an allreduce group """
        rng = random.Random(allreduce_group.group_key_seed)
        index = allreduce_group.ordered_group_endpoints.index(self.endpoint)
        generalized_index = rng.sample(range(self.target_group_size), allreduce_group.group_size)[index]
        nbits = int(np.ceil(np.log2(self.target_group_size)))
        new_bits = bin(generalized_index)[2:].rjust(nbits, '0')
        self.group_bits = (self.group_bits + new_bits)[-len(self.group_bits):]
        logger.debug(f"{self.endpoint} - updated group key to {self.group_bits}")

        if is_leader and self.insufficient_size < allreduce_group.group_size < self.excessive_size:
            asyncio.create_task(self.declare_nbits(self.prefix, len(self.group_bits), self.nbits_expiration))

    async def update_key_on_group_not_found(self):
        """ this function is triggered whenever averager fails to assemble group within timeout """
        self.group_bits = self.group_bits[1:]

    async def update_key_on_overcrowded(self):
        """ this function is triggered if averager encounters an overcrowded group """
        self.group_bits = random.choice('01') + self.group_bits

    async def publish_current_key(self, expiration_time: float, looking_for_group: bool = True) -> bool:
        """ A shortcut function for declare_group_key with averager's current active key and endpoint """
        return await self.declare_averager(self.current_key, self.endpoint, expiration_time, looking_for_group)

    async def declare_averager(self, group_key: GroupKey, endpoint: Endpoint, expiration_time: float,
                               looking_for_group: bool = True) -> bool:
        """
        Add (or remove) the averager to a given allreduce bucket

        :param group_key: allreduce group key, e.g. my_averager.0b011011101
        :param endpoint: averager public endpoint for incoming requests
        :param expiration_time: intent to run allreduce before this timestamp
        :param looking_for_group: by default (True), declare the averager as "looking for group" in a given group;
          If False, this will instead mark that the averager as no longer looking for group, (e.g. it already finished)
        :return: True if declared, False if declaration was rejected by DHT peers
        :note: when leaving (i.e. is_active=False), please specify the same expiration_time as when entering the group
        :note: setting is_active=False does *not* guarantee that others will immediately stop to query you.
        """
        expiration_time = expiration_time if looking_for_group else float(np.nextafter(expiration_time, float('inf')))
        return await self.dht.store(key=group_key, subkey=endpoint, value=looking_for_group,
                                    expiration_time=expiration_time, return_future=True)

    async def get_averagers(self, group_key: GroupKey, only_active: bool) -> List[Tuple[Endpoint, DHTExpiration]]:
        """
        Find and return averagers that were declared with a given all-reduce key

        :param group_key: finds averagers that have the this group key, e.g. my_averager.0b011011101
        :param only_active: if True, return only active averagers that are looking for group (i.e. with value = True)
            if False, return all averagers under a given group_key regardless of value
        :return: endpoints and expirations of every matching averager
        """
        assert is_valid_group(group_key), f"Group key {group_key} is invalid, must follow {GROUP_PATTERN}"
        result = await self.dht.get(group_key, latest=True, return_future=True)
        if result is None:
            logger.debug(f"Allreduce group not found: {group_key}, creating new group.")
            return []
        averagers = [(endpoint, entry.expiration_time) for endpoint, entry in result.value.items()
                     if not only_active or entry.value is True]
        return averagers

    async def declare_nbits(self, group_key: GroupKey, nbits: int, expiration_time: DHTExpiration) -> bool:
        """ notify other peers that they can run averaging at this depth """
        return await self.dht.store(key=group_key, subkey=self.RESERVED_KEY_FOR_NBITS, value=nbits,
                                    expiration_time=expiration_time, return_future=True)

    async def get_nbits(self, group_key: GroupKey) -> int:
        """ notify other peers that they can run averaging at this depth. If not found, return 0. """
        result = await self.dht.get(key=group_key, return_future=True)
        return result.get(self.RESERVED_KEY_FOR_NBITS, 0) if isinstance(result, dict) else 0