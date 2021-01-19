import re
from typing import List, Tuple

from hivemind.dht import DHT
from hivemind.utils import Endpoint, DHTExpiration, get_logger
from numpy import nextafter

GroupKey = str
GROUP_PATTERN = re.compile('^(([^.])+)[.]0b[01]+$')  # e.g. bert_exp4_averaging.0b01001101
logger = get_logger(__name__)


def is_valid_group(maybe_group: str) -> bool:
    """ A group identifier must contain group type, followed by one or more .-separated indices, and any ?metadata"""
    return bool(GROUP_PATTERN.fullmatch(maybe_group))


class DHTHandler:
    """ Utility class that implements basic DHT interactions required by DecentralizedAverager """
    def __init__(self, dht: DHT):
        self.dht = dht

    async def declare_averager(self, group_key: GroupKey, endpoint: Endpoint, expiration_time: float, *,
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
        assert is_valid_group(group_key), f"Group key {group_key} is invalid, must follow {GROUP_PATTERN}"
        expiration_time = expiration_time if looking_for_group else float(nextafter(expiration_time, float('inf')))
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
