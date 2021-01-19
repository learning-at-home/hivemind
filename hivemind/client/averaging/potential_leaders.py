import asyncio
import contextlib
from math import isfinite
from typing import Optional, Set, Tuple

import hivemind
from hivemind.client.averaging.key_manager import GroupKey, GroupKeyManager
from hivemind.utils import Endpoint, DHTExpiration, TimedStorage, get_dht_time, get_logger

logger = get_logger(__name__)


class PotentialLeaders:
    """ An utility class that searches for averagers that could become our leaders """

    def __init__(self, endpoint: Endpoint, averaging_expiration: DHTExpiration, target_group_size: Optional[int]):
        self.endpoint, self.averaging_expiration = endpoint, averaging_expiration
        self.target_group_size = target_group_size
        self.running, self.update_triggered, self.update_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.declared_expiration, self.lock_search, self.lock_declare = asyncio.Event(), asyncio.Lock(), asyncio.Lock()
        self.leader_queue = TimedStorage[Endpoint, DHTExpiration]()
        self.past_attempts: Set[Tuple[Endpoint, DHTExpiration]] = set()
        self.declared_expiration_time = float('inf')
        self.declared_group_key: Optional[GroupKey] = None
        self.max_assured_time = float('-inf')
        self.search_end_time = float('inf')

    @contextlib.asynccontextmanager
    async def begin_search(self, key_manager: GroupKeyManager, timeout: Optional[float]):
        async with self.lock_search:
            self.running.set()
            self.search_end_time = get_dht_time() + timeout if timeout is not None else float('inf')
            update_queue_task = asyncio.create_task(self._update_queue_periodically(key_manager))
            declare_averager_task = asyncio.create_task(self._declare_averager_periodically(key_manager))
            try:
                yield self
            finally:
                if not update_queue_task.done():
                    update_queue_task.cancel()
                if not declare_averager_task.done():
                    declare_averager_task.cancel()
                for field in (self.past_attempts, self.leader_queue, self.running,
                              self.update_finished, self.update_triggered, self.declared_expiration):
                    field.clear()
                self.max_assured_time = float('-inf')
                self.search_end_time = float('inf')

    @contextlib.asynccontextmanager
    async def pause_search(self):
        was_running = self.running.is_set()
        try:
            self.running.clear()
            yield
        finally:
            if was_running:
                self.running.set()
            else:
                self.running.clear()

    async def pop_next_leader(self) -> Endpoint:
        """ Remove and return the next most suitable leader or throw an exception if reached timeout """
        assert self.running.is_set(), "Not running search at the moment"
        while True:
            maybe_next_leader, entry = self.leader_queue.top()

            if maybe_next_leader is None or self.max_assured_time <= entry.expiration_time <= self.search_end_time:
                self.update_triggered.set()

            if maybe_next_leader is None or entry.expiration_time >= self.declared_expiration_time:
                await asyncio.wait({self.update_finished.wait(), self.declared_expiration.wait()},
                                   return_when=asyncio.FIRST_COMPLETED)
                self.declared_expiration.clear()
                if self.update_finished.is_set():
                    self.update_finished.clear()
                    continue
                else:
                    raise asyncio.TimeoutError("pop_next_leader was invalidated: re-declared averager in background")

            del self.leader_queue[maybe_next_leader]
            self.past_attempts.add((maybe_next_leader, entry.expiration_time))
            return maybe_next_leader

    @property
    def request_expiration_time(self) -> float:
        """ this averager's current expiration time - used to send join requests to leaders """
        if isfinite(self.declared_expiration_time):
            return self.declared_expiration_time
        else:
            return min(get_dht_time() + self.averaging_expiration, self.search_end_time)

    async def _update_queue_periodically(self, key_manager: GroupKeyManager):
        DISCREPANCY = hivemind.utils.timed_storage.MAX_DHT_TIME_DISCREPANCY_SECONDS
        while get_dht_time() < self.search_end_time:
            new_peers = await key_manager.get_averagers(key_manager.current_key, only_active=True)
            self.max_assured_time = max(self.max_assured_time, get_dht_time() + self.averaging_expiration - DISCREPANCY)

            self.leader_queue.clear()
            for peer, peer_expiration_time in new_peers:
                if peer == self.endpoint or (peer, peer_expiration_time) in self.past_attempts:
                    continue
                self.leader_queue.store(peer, peer_expiration_time, peer_expiration_time)
                self.max_assured_time = max(self.max_assured_time, peer_expiration_time - DISCREPANCY)

            self.update_finished.set()

            await asyncio.wait(
                {self.running.wait(), self.update_triggered.wait()}, return_when=asyncio.ALL_COMPLETED,
                timeout=self.search_end_time - get_dht_time() if isfinite(self.search_end_time) else None)
            self.update_triggered.clear()

    async def _declare_averager_periodically(self, key_manager: GroupKeyManager):
        async with self.lock_declare:
            try:
                while True:
                    await self.running.wait()

                    new_expiration_time = min(get_dht_time() + self.averaging_expiration, self.search_end_time)
                    self.declared_group_key = group_key = key_manager.current_key
                    self.declared_expiration_time = new_expiration_time
                    self.declared_expiration.set()
                    await key_manager.publish_current_key(looking_for_group=True, expiration_time=new_expiration_time)
                    await asyncio.sleep(self.declared_expiration_time - get_dht_time())
            except Exception as e:  # note: we catch exceptions here because otherwise they are never printed
                logger.error(f"{self.endpoint} - caught {type(e)}: {e}")
            finally:
                if self.declared_group_key is not None:
                    prev_declared_key, prev_expiration_time = self.declared_group_key, self.declared_expiration_time
                    self.declared_group_key, self.declared_expiration_time = None, float('inf')
                    self.leader_queue, self.max_assured_time = TimedStorage[Endpoint, DHTExpiration](), float('-inf')
                    await key_manager.declare_group_key(prev_declared_key, self.endpoint, prev_expiration_time,
                                                        looking_for_group=False)
