import heapq
import random
import threading
import time
from contextlib import contextmanager
from operator import itemgetter
from typing import Dict, List, Tuple

import grpc
from grpc._channel import _InactiveRpcError

from hivemind.dht import DHT
from hivemind.moe.client.expert import RemoteExpert
from hivemind.moe.server.expert_uid import ExpertPrefix, ExpertUID
from hivemind.optim.performance_ema import PerformanceEMA
from hivemind.utils import Endpoint, TimedStorage, DHTExpiration, ValueWithExpiration, get_dht_time, get_logger

logger = get_logger(__name__)

QueueItem = Tuple[bool, float, float, ExpertUID]


class ExpertBalancer:
    def __init__(
        self, dht: DHT, key: ExpertPrefix, update_period: float = 30.0, sleep_timeout: float = 5.0, **kwargs
    ):
        self.dht, self.key = dht, key
        self.ema_kwargs = kwargs
        self.experts = TimedStorage[ExpertUID, Endpoint]()
        self.blacklist = TimedStorage[ExpertUID, type(None)]()
        self.throughputs: Dict[ExpertUID, PerformanceEMA] = {}
        # had_forward_pass, throughput, random, uid
        self.queue: List[QueueItem] = []
        self.uid_to_queue: Dict[ExpertUID, QueueItem] = {}
        self.lock = threading.Lock()
        self.is_alive = threading.Event()
        self.is_alive.set()
        self.update_trigger, self.update_finished = threading.Event(), threading.Event()
        self.update_period, self.last_update = update_period, get_dht_time()
        self.sleep_timeout = sleep_timeout
        self.update_thread = threading.Thread(target=self.update_experts_in_background, daemon=True)
        self.update_thread.start()

    def update_experts_in_background(self):
        while self.is_alive.is_set():
            time_to_next_update = max(0.0, self.last_update + self.update_period - get_dht_time())
            try:
                self.update_trigger.wait(timeout=time_to_next_update)
                # update triggered by main thread
            except TimeoutError:
                pass  # update triggered by refresh_period

            self.update_trigger.clear()
            response = self.dht.get(self.key, latest=True)
            if isinstance(response, ValueWithExpiration) and isinstance(response.value, dict):
                for index, expert_info in response.value.items():
                    try:
                        (uid, endpoint), expiration_time = expert_info

                        maybe_banned = self.blacklist.get(uid)
                        if maybe_banned is None or expiration_time > maybe_banned.expiration_time:
                            self._add_expert(uid, endpoint, expiration_time)
                        else:
                            logger.debug(f"Not adding expert {uid} (blacklisted).")
                    except Exception as e:
                        logger.warning(f"Skipping malformed expert info {expert_info} (exc={e})")
            else:
                logger.warning(
                    f"Could not refresh experts, dht info key contains {response}, "
                    f"will retry in {time_to_next_update}s"
                )
            if len(self.queue) == 0:
                logger.warning("Update routine finished, but still no experts available.")
                time.sleep(self.sleep_timeout)

            self.last_update = get_dht_time()
            self.update_finished.set()

    def _add_expert(self, uid: ExpertUID, endpoint: Endpoint, expiration_time: DHTExpiration):
        with self.lock:
            self.experts.store(uid, endpoint, expiration_time)
            if uid not in self.uid_to_queue:
                logger.debug(f"Adding new expert: {uid}, expiration time = {expiration_time:.3f}.")
                self.throughputs[uid] = PerformanceEMA(**self.ema_kwargs, paused=True)
                heap_entry = (False, 0.0, random.random(), uid)
                heapq.heappush(self.queue, heap_entry)
                self.uid_to_queue[uid] = heap_entry
            else:
                logger.debug(f"Refreshing existing expert: {uid}, new expiration time = {expiration_time:.3f}.")

    def _ban_expert(self, uid: ExpertUID):
        with self.lock:
            maybe_expert = self.experts.get(uid)
            expiration_time = maybe_expert.expiration_time if maybe_expert else get_dht_time()
            self.blacklist.store(uid, None, expiration_time)
            self.uid_to_queue.pop(uid, None)
            self.throughputs.pop(uid, None)
            del self.experts[uid]
            logger.debug(f"Banned expert {uid} with expiration time = {expiration_time:.2f}.")

    @contextmanager
    def use_another_expert(self, task_size: float) -> RemoteExpert:
        while True:
            if len(self.queue) == 0:
                self.update_finished.clear()
                self.update_trigger.set()
                self.update_finished.wait()
                continue

            with self.lock:
                logger.debug(f"Getting a new expert, queue state: {self.queue}")
                *_, uid = heap_entry = heapq.heappop(self.queue)
                maybe_endpoint = self.experts.get(uid)
                if maybe_endpoint is None:
                    # remove expired expert from queue
                    self.uid_to_queue.pop(uid, None)
                    self.throughputs.pop(uid, None)
                if self.uid_to_queue.get(uid) != heap_entry:
                    logger.debug(f"Skipping expert {uid} "
                                 f"(uid_to_queue={self.uid_to_queue.get(uid)}, entry={heap_entry})")
                    continue  # skip uids that are banned or expired
                break
        try:
            with self.throughputs[uid].update_threadsafe(task_size):
                logger.debug(f"Using expert {uid}, throughput = {self.throughputs[uid].samples_per_second}.")
                yield RemoteExpert(uid, maybe_endpoint.value)
                logger.debug(f"Finished using expert {uid}.")

            new_throughput = 1 / self.throughputs[uid].samples_per_second
            new_heap_entry = (True, new_throughput, random.random(), uid)
            with self.lock:
                heapq.heappush(self.queue, new_heap_entry)
                self.uid_to_queue[uid] = new_heap_entry
        except _InactiveRpcError as error:
            if error.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                # response was too slow, choose the next expert and mark this one as slow
                with self.lock:
                    new_throughput = max(self.queue, key=itemgetter(1))[1]
                    new_heap_entry = (True, new_throughput, random.random(), uid)
                    heapq.heappush(self.queue, new_heap_entry)
                    self.uid_to_queue[uid] = new_heap_entry

                raise
            else:
                self._ban_expert(uid)
                raise
        except BaseException:
            self._ban_expert(uid)
            raise

    def shutdown(self):
        self.is_alive.clear()
        self.update_finished.clear()
        self.update_trigger.set()
        self.update_finished.wait()
