import random
from typing import Optional

from hivemind.client.averaging.allreduce import AllReduceRunner
from hivemind.client.averaging.dht_handler import DHTHandler, GroupKey


class GroupKeyManager:
    """
    Utility class that keeps track of the current averaging key
    """
    def __init__(self, dht_handler: DHTHandler, prefix: str, initial_group_bits: Optional[str], target_group_size: int,
                 insufficient_size: Optional[int] = None, excessive_size: Optional[int] = None):
        self.dht_handler, self.prefix, self.target_group_size = dht_handler, prefix, target_group_size
        self.insufficient_size = insufficient_size or max(1, target_group_size // 2)
        self.excessive_size = excessive_size or target_group_size * 3
        self.group_bits = initial_group_bits or ''
        assert all(bit in '01' for bit in self.group_bits)

    @property
    def current_key(self) -> GroupKey:
        return f"{self.prefix}.0b{self.group_bits}"

    def update_key_on_success(self, allreduce_group: AllReduceRunner, is_leader=True):
        pass #TODO

    def update_key_on_failure(self):
        pass #TODO
