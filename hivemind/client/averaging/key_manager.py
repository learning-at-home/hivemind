import re
import random
from typing import Optional, List, Tuple

import numpy as np

from hivemind.dht import DHT
from hivemind.client.averaging.allreduce import AllReduceRunner
from hivemind.utils import get_logger, Endpoint, DHTExpiration, get_dht_time

GroupKey = str
GROUP_PATTERN = re.compile('^(([^.])+)[.]0b[01]*$')  # e.g. bert_exp4_averaging.0b01001101
logger = get_logger(__name__)
SUCCESS_KEY = 'abyrvalg'


def is_valid_group(maybe_group: str) -> bool:
    """ A group identifier must contain group type, followed by one or more .-separated indices, and any ?metadata"""
    return bool(GROUP_PATTERN.fullmatch(maybe_group))


class GroupKeyManager:
    """
    Utility class that declares and fetches averaging-related keys using a DHT
    """
    RESERVED_KEY_FOR_NBITS = '::NBITS'

    def __init__(self, dht: DHT, endpoint: Endpoint, prefix: str, initial_group_bits: Optional[str],
                 target_group_size: int, insufficient_size: Optional[int] = None, excessive_size: Optional[int] = None,
                 nbits_expiration: float = 60, nbits_rewrite_grace_period: float = 15):
        assert initial_group_bits is None or all(bit in '01' for bit in initial_group_bits)
        if initial_group_bits is None:
            initial_group_nbits = 0  # TODO
            initial_group_bits = ''.join(random.choice('01') for _ in range(initial_group_nbits))
        self.dht, self.endpoint, self.prefix, self.group_bits = dht, endpoint, prefix, initial_group_bits
        self.target_group_size = target_group_size
        self.insufficient_size = insufficient_size or max(1, target_group_size // 2)
        self.excessive_size = excessive_size or round(target_group_size * 1.1)
        self.nbits_expiration, self.nbits_grace_period = nbits_expiration, nbits_rewrite_grace_period
        self.suggested_nbits: Optional[int] = None
        self.num_active_averagers = 0
        self.success_upper = 0
        self.success_lower = 0
        self.small_group = 0
        self.crowded = 0

    @property
    def current_key(self) -> GroupKey:
        return f"{self.prefix}.0b{self.group_bits}"

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
        if result is None or not isinstance(result.value, dict):
            logger.debug(f"Allreduce group not found: {group_key}, creating new group.")
            return []
        averagers = [(key, entry.expiration_time) for key, entry in result.value.items()
                     if key != self.RESERVED_KEY_FOR_NBITS and (not only_active or entry.value is True)]
        self.num_active_averagers = len([key for key, entry in result.value.items() if entry.value is True])

        return averagers

    async def update_key_on_group_assembled(self, allreduce_group: AllReduceRunner, is_leader: bool = True):
        """ this function is triggered every time an averager finds an allreduce group """

        # IMPORTANT LOGIC OF MOSHPIT SGD
        rng = random.Random(allreduce_group.group_key_seed)
        index = allreduce_group.ordered_group_endpoints.index(self.endpoint)
        generalized_index = rng.sample(range(self.target_group_size), allreduce_group.group_size)[index]
        nbits = int(np.ceil(np.log2(self.target_group_size)))
        new_bits = bin(generalized_index)[2:].rjust(nbits, '0')
        self.group_bits = (self.group_bits + new_bits)[-len(self.group_bits):] if self.group_bits else ''
        logger.debug(f"{self.endpoint} - updated group key to {self.group_bits}")
        # /IMPORTANT LOGIC OF MOSHPIT SGD

        c = await self.dht.get(key=SUCCESS_KEY, latest=True, return_future=True)
        if c is not None:
            d = c.value
            if len(self.group_bits) - 1 in d:
                self.success_lower += 1
            else:
                self.success_lower = 0
            if len(self.group_bits) + 1 in d:
                self.success_upper += 1
            else:
                self.success_upper = 0

        else:
            self.success_upper = 0
            self.success_lower = 0

        if is_leader and self.target_group_size//2 <= self.num_active_averagers <= self.target_group_size:
            await self.dht.store(key=SUCCESS_KEY, subkey=len(self.group_bits), value=True,
                                 expiration_time=get_dht_time() + 1, return_future=True, )
        if self.num_active_averagers > self.target_group_size:
            self.crowded += 1
        else:
            self.crowded = 0

        if self.num_active_averagers <= self.target_group_size//2:
            self.small_group += 1
            if self.small_group > 1 and len(self.group_bits):
                self.group_bits = self.group_bits[1:]
                return
        else:
            self.small_group = 0

        if self.success_upper > 2 or self.crowded > 1:
            self.group_bits = random.choice('01') + self.group_bits
            return


    async def update_key_on_not_enough_peers(self):
        """ this function is triggered whenever averager fails to assemble group within timeout """
