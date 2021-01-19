from typing import Optional

from hivemind.client.averaging.allreduce import AllReduceRunner
from hivemind.client.averaging.dht_handler import DHTHandler, GroupKey


class GroupKeyManager:
    """ Utility class that keeps track of the current averaging key """
    def __init__(self, dht_handler: DHTHandler, prefix: str, initial_group_bits: Optional[str] = None, lookahead: int = 5):
        assert initial_group_bits is None or all(bit in '01' for bit in initial_group_bits)
        self.dht_handler, self.prefix, self.lookahead = dht_handler, prefix, lookahead
        self.group_bits = initial_group_bits if initial_group_bits is not None else self.infer_group_key()

    @property
    def current_key(self) -> GroupKey:
        return f"{self.prefix}.0b{self.group_bits}"

    def infer_group_key(self):
        """ Infer the current depth from the existing DHT keys """
        raise NotImplementedError()

    def update_key_on_success(self, allreduce_group: AllReduceRunner):
        pass #TODO

    def update_key_on_failure(self):
        pass #TODO

    async def update_dht_on_success(self):
        pass #TODO
