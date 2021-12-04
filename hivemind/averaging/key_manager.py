import random
import re
from typing import List, Optional, Tuple

import numpy as np

from hivemind.averaging.group_info import GroupInfo
from hivemind.dht import DHT
from hivemind.p2p import PeerID
from hivemind.utils import DHTExpiration, ValueWithExpiration, get_dht_time, get_logger

GroupKey = str
GROUP_PATTERN = re.compile("^(([^.])+)[.]0b[01]*$")  # e.g. bert_exp4_averaging.0b01001101
DEFAULT_NUM_BUCKETS = 256
logger = get_logger(__name__)


def is_valid_group(maybe_group: str) -> bool:
    """A group identifier must contain group type, followed by one or more .-separated indices, and any ?metadata"""
    return bool(GROUP_PATTERN.fullmatch(maybe_group))


class GroupKeyManager:
    """
    Utility class that declares and fetches averaging-related keys using a DHT
    """

    def __init__(
        self,
        dht: DHT,
        prefix: str,
        initial_group_bits: str,
        target_group_size: Optional[int],
    ):
        assert all(bit in "01" for bit in initial_group_bits)
        if target_group_size is not None and not is_power_of_two(target_group_size):
            logger.warning("It is recommended to set target_group_size to a power of 2")

        self.dht, self.prefix, self.group_bits = dht, prefix, initial_group_bits
        self.target_group_size = target_group_size
        self.peer_id = dht.peer_id

    @property
    def current_key(self) -> GroupKey:
        return f"{self.prefix}.0b{self.group_bits}"

    async def declare_averager(
        self, group_key: GroupKey, peer_id: PeerID, expiration_time: float, looking_for_group: bool = True
    ) -> bool:
        """
        Add (or remove) the averager to a given allreduce bucket

        :param group_key: allreduce group key, e.g. my_averager.0b011011101
        :param peer_id: averager public peer_id for incoming requests
        :param expiration_time: intent to run allreduce before this timestamp
        :param looking_for_group: by default (True), declare the averager as "looking for group" in a given group;
          If False, this will instead mark that the averager as no longer looking for group, (e.g. it already finished)
        :return: True if declared, False if declaration was rejected by DHT peers
        :note: when leaving (i.e. is_active=False), please specify the same expiration_time as when entering the group
        :note: setting is_active=False does *not* guarantee that others will immediately stop to query you.
        """
        expiration_time = expiration_time if looking_for_group else float(np.nextafter(expiration_time, float("inf")))
        return await self.dht.store(
            key=group_key,
            subkey=peer_id.to_bytes(),
            value=looking_for_group,
            expiration_time=expiration_time,
            return_future=True,
        )

    async def get_averagers(self, group_key: GroupKey, only_active: bool) -> List[Tuple[PeerID, DHTExpiration]]:
        """
        Find and return averagers that were declared with a given all-reduce key

        :param group_key: finds averagers that have the this group key, e.g. my_averager.0b011011101
        :param only_active: if True, return only active averagers that are looking for group (i.e. with value = True)
            if False, return all averagers under a given group_key regardless of value
        :return: peer_ids and expirations of every matching averager
        """
        assert is_valid_group(group_key), f"Group key {group_key} is invalid, must follow {GROUP_PATTERN}"
        result = await self.dht.get(group_key, latest=True, return_future=True)
        if result is None or not isinstance(result.value, dict):
            logger.debug(f"Allreduce group not found: {group_key}, creating new group")
            return []
        averagers = []
        for key, looking_for_group in result.value.items():
            try:
                if only_active and not looking_for_group.value:
                    continue
                averagers.append((PeerID(key), looking_for_group.expiration_time))
            except Exception as e:
                logger.warning(f"Could not parse group key {key} ({looking_for_group}, exc={e})")
        return averagers

    async def update_key_on_group_assembled(self, group_info: GroupInfo, is_leader: bool = True):
        """this function is triggered every time an averager finds an allreduce group"""
        rng = random.Random(group_info.group_id)
        index = group_info.peer_ids.index(self.peer_id)
        num_buckets = self.target_group_size
        if num_buckets is None:
            num_buckets = next_power_of_two(group_info.group_size)
        generalized_index = rng.sample(range(num_buckets), group_info.group_size)[index]
        nbits = int(np.ceil(np.log2(num_buckets)))
        new_bits = bin(generalized_index)[2:].rjust(nbits, "0")
        self.group_bits = (self.group_bits + new_bits)[-len(self.group_bits) :] if self.group_bits else ""
        logger.debug(f"{self.peer_id} - updated group key to {self.group_bits}")

    async def update_key_on_not_enough_peers(self):
        """this function is triggered whenever averager fails to assemble group within timeout"""
        pass  # to be implemented in subclasses


def is_power_of_two(n):
    """Check whether n is a power of 2"""
    return (n != 0) and (n & (n - 1) == 0)


def next_power_of_two(n):
    """Round n up to the nearest power of 2"""
    return 1 if n == 0 else 2 ** (n - 1).bit_length()
