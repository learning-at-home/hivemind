import argparse
import asyncio
import random
import time
import uuid
from typing import Tuple

import numpy as np
from tqdm import trange

import hivemind
from hivemind.utils.limits import increase_file_limit
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


class NodeKiller:
    """Auxiliary class that kills dht nodes over a pre-defined schedule"""

    def __init__(self, shutdown_peers: list, shutdown_timestamps: list):
        self.shutdown_peers = set(shutdown_peers)
        self.shutdown_timestamps = shutdown_timestamps
        self.current_iter = 0
        self.timestamp_iter = 0
        self.lock = asyncio.Lock()

    async def check_and_kill(self):
        async with self.lock:
            if (
                self.shutdown_timestamps != None
                and self.timestamp_iter < len(self.shutdown_timestamps)
                and self.current_iter == self.shutdown_timestamps[self.timestamp_iter]
            ):
                shutdown_peer = random.sample(self.shutdown_peers, 1)[0]
                shutdown_peer.shutdown()
                self.shutdown_peers.remove(shutdown_peer)
                self.timestamp_iter += 1
            self.current_iter += 1


async def store_and_get_task(
    peers: list,
    total_num_rounds: int,
    num_store_peers: int,
    num_get_peers: int,
    wait_after_iteration: float,
    delay: float,
    expiration: float,
    latest: bool,
    node_killer: NodeKiller,
) -> Tuple[list, list, list, list, int, int]:
    """Iteratively choose random peers to store data onto the dht, then retrieve with another random subset of peers"""

    total_stores = total_gets = 0
    successful_stores = []
    successful_gets = []
    store_times = []
    get_times = []

    for _ in range(total_num_rounds):
        key = uuid.uuid4().hex

        store_start = time.perf_counter()
        store_peers = random.sample(peers, min(num_store_peers, len(peers)))
        store_subkeys = [uuid.uuid4().hex for _ in store_peers]
        store_values = {subkey: uuid.uuid4().hex for subkey in store_subkeys}
        store_tasks = [
            peer.store(
                key,
                subkey=subkey,
                value=store_values[subkey],
                expiration_time=hivemind.get_dht_time() + expiration,
                return_future=True,
            )
            for peer, subkey in zip(store_peers, store_subkeys)
        ]
        store_result = await asyncio.gather(*store_tasks)
        await node_killer.check_and_kill()

        store_times.append(time.perf_counter() - store_start)

        total_stores += len(store_result)
        successful_stores_per_iter = sum(store_result)
        successful_stores.append(successful_stores_per_iter)
        await asyncio.sleep(delay)

        get_start = time.perf_counter()
        get_peers = random.sample(peers, min(num_get_peers, len(peers)))
        get_tasks = [peer.get(key, latest, return_future=True) for peer in get_peers]
        get_result = await asyncio.gather(*get_tasks)
        get_times.append(time.perf_counter() - get_start)

        successful_gets_per_iter = 0

        total_gets += len(get_result)
        for result in get_result:
            if result != None:
                attendees, expiration = result
                if len(attendees.keys()) == successful_stores_per_iter:
                    get_ok = True
                    for key in attendees:
                        if attendees[key][0] != store_values[key]:
                            get_ok = False
                            break
                    successful_gets_per_iter += get_ok

        successful_gets.append(successful_gets_per_iter)
        await asyncio.sleep(wait_after_iteration)

    return store_times, get_times, successful_stores, successful_gets, total_stores, total_gets


async def benchmark_dht(
    num_peers: int,
    initial_peers: int,
    random_seed: int,
    num_threads: int,
    total_num_rounds: int,
    num_store_peers: int,
    num_get_peers: int,
    wait_after_iteration: float,
    delay: float,
    wait_timeout: float,
    expiration: float,
    latest: bool,
    failure_rate: float,
):
    random.seed(random_seed)

    logger.info("Creating peers...")
    peers = []
    for _ in trange(num_peers):
        neighbors = sum(
            [peer.get_visible_maddrs() for peer in random.sample(peers, min(initial_peers, len(peers)))], []
        )
        peer = hivemind.DHT(initial_peers=neighbors, start=True, wait_timeout=wait_timeout)
        peers.append(peer)

    benchmark_started = time.perf_counter()
    logger.info("Creating store and get tasks...")
    shutdown_peers = random.sample(peers, min(int(failure_rate * num_peers), num_peers))
    assert len(shutdown_peers) != len(peers)
    remaining_peers = list(set(peers) - set(shutdown_peers))
    shutdown_timestamps = random.sample(
        range(0, num_threads * total_num_rounds), min(len(shutdown_peers), num_threads * total_num_rounds)
    )
    shutdown_timestamps.sort()
    node_killer = NodeKiller(shutdown_peers, shutdown_timestamps)
    task_list = [
        asyncio.create_task(
            store_and_get_task(
                remaining_peers,
                total_num_rounds,
                num_store_peers,
                num_get_peers,
                wait_after_iteration,
                delay,
                expiration,
                latest,
                node_killer,
            )
        )
        for _ in trange(num_threads)
    ]

    store_and_get_result = await asyncio.gather(*task_list)
    benchmark_total_time = time.perf_counter() - benchmark_started
    total_store_times = []
    total_get_times = []
    total_successful_stores = []
    total_successful_gets = []
    total_stores = total_gets = 0
    for result in store_and_get_result:
        store_times, get_times, successful_stores, successful_gets, stores, gets = result

        total_store_times.extend(store_times)
        total_get_times.extend(get_times)
        total_successful_stores.extend(successful_stores)
        total_successful_gets.extend(successful_gets)
        total_stores += stores
        total_gets += gets

    alive_peers = [peer.is_alive() for peer in peers]
    logger.info(
        f"Store wall time (sec.): mean({np.mean(total_store_times):.3f}) "
        + f"std({np.std(total_store_times, ddof=1):.3f}) max({np.max(total_store_times):.3f})"
    )
    logger.info(
        f"Get wall time (sec.): mean({np.mean(total_get_times):.3f}) "
        + f"std({np.std(total_get_times, ddof=1):.3f}) max({np.max(total_get_times):.3f})"
    )
    logger.info(f"Average store time per worker: {sum(total_store_times) / num_threads:.3f} sec.")
    logger.info(f"Average get time per worker: {sum(total_get_times) / num_threads:.3f} sec.")
    logger.info(f"Total benchmark time: {benchmark_total_time:.5f} sec.")
    logger.info(
        "Store success rate: "
        + f"{sum(total_successful_stores) / total_stores * 100:.1f}% ({sum(total_successful_stores)}/{total_stores})"
    )
    logger.info(
        "Get success rate: "
        + f"{sum(total_successful_gets) / total_gets * 100:.1f}% ({sum(total_successful_gets)}/{total_gets})"
    )
    logger.info(f"Node survival rate: {len(alive_peers) / len(peers) * 100:.3f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_peers", type=int, default=16, required=False)
    parser.add_argument("--initial_peers", type=int, default=4, required=False)
    parser.add_argument("--random_seed", type=int, default=30, required=False)
    parser.add_argument("--num_threads", type=int, default=10, required=False)
    parser.add_argument("--total_num_rounds", type=int, default=16, required=False)
    parser.add_argument("--num_store_peers", type=int, default=8, required=False)
    parser.add_argument("--num_get_peers", type=int, default=8, required=False)
    parser.add_argument("--wait_after_iteration", type=float, default=0, required=False)
    parser.add_argument("--delay", type=float, default=0, required=False)
    parser.add_argument("--wait_timeout", type=float, default=5, required=False)
    parser.add_argument("--expiration", type=float, default=300, required=False)
    parser.add_argument("--latest", type=bool, default=True, required=False)
    parser.add_argument("--failure_rate", type=float, default=0.1, required=False)
    parser.add_argument("--increase_file_limit", action="store_true")
    args = vars(parser.parse_args())

    if args.pop("increase_file_limit", False):
        increase_file_limit()

    asyncio.run(benchmark_dht(**args))
