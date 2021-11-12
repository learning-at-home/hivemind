import argparse
import asyncio
import random
import time
import uuid

import numpy as np
from tqdm import trange

import hivemind
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


async def store_task(peer, key, value, expiration):
    subkey = uuid.uuid4().hex

    store_ok = await peer.store(
        key, subkey=subkey, value=value, expiration_time=hivemind.get_dht_time() + expiration, return_future=True
    )

    return store_ok, subkey, value


async def store_and_get_task(
    peers: list,
    total_num_rounds: int,
    num_store_peers: int,
    num_get_peers: int,
    wait_after_iteration: float,
    delay: float,
    expiration: float,
    latest: bool,
    failure_rate: float,
):
    total_stores = total_gets = 0
    successful_stores = []
    successful_gets = []
    store_times = []
    get_times = []

    for _ in range(total_num_rounds):
        key = uuid.uuid4().hex
        value = uuid.uuid4().hex

        store_start = time.perf_counter()
        store_peers = random.sample(peers, min(num_store_peers, len(peers)))
        store_tasks = [store_task(peer, key, value, expiration) for peer in store_peers]
        store_result, _ = await asyncio.wait(store_tasks, return_when=asyncio.ALL_COMPLETED)
        store_times.append(time.perf_counter() - store_start)

        successful_stores_per_iter = 0
        store_attendees = dict()

        total_stores += len(store_result)
        for result in store_result:
            store_ok, attendee, value = result.result()
            successful_stores_per_iter += store_ok
            store_attendees[attendee] = value

        successful_stores.append(successful_stores_per_iter)
        await asyncio.sleep(delay)

        get_start = time.perf_counter()
        get_peers = random.sample(peers, min(num_get_peers, len(peers)))
        get_tasks = [peer.get(key, latest, return_future=True) for peer in get_peers]
        get_result, _ = await asyncio.wait(get_tasks, return_when=asyncio.ALL_COMPLETED)
        get_times.append(time.perf_counter() - get_start)

        successful_gets_per_iter = 0

        total_gets += len(get_result)
        for result in get_result:
            attendees, expiration = result.result()
            if attendees.keys() == store_attendees.keys():
                get_ok = True
                for key in attendees:
                    if attendees[key][0] != store_attendees[key]:
                        get_ok = False
                        break
                successful_gets_per_iter += get_ok

        successful_gets.append(successful_gets_per_iter)
        await asyncio.sleep(wait_after_iteration)

    logger.info(
        "Store wall time: "
        + f"mean({np.mean(store_times):.3f}) std({np.std(store_times, ddof=1):.3f}) max({np.max(store_times):.3f}) "
        + "Store success rate: "
        + f"{sum(successful_stores) / total_stores * 100:.1f}% ({sum(successful_stores)}/{total_stores})"
    )
    logger.info(
        "Get wall time: "
        + f"mean({np.mean(get_times):.3f}) std({np.std(get_times, ddof=1):.3f}) max({np.max(get_times):.3f}) "
        + "Get success rate: "
        + f"{sum(successful_gets) / total_gets * 100:.1f}% ({sum(successful_gets)}/{total_gets})"
    )

    return sum(store_times), sum(get_times), sum(successful_gets), total_gets


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
    task_list = [
        asyncio.create_task(
            store_and_get_task(
                peers,
                total_num_rounds,
                num_store_peers,
                num_get_peers,
                wait_after_iteration,
                delay,
                expiration,
                latest,
                failure_rate,
            )
        )
        for _ in trange(num_threads)
    ]

    store_and_get_result = await asyncio.gather(*task_list)
    benchmark_total_time = time.perf_counter() - benchmark_started
    total_store_time = total_get_time = 0
    total_successful_gets = total_gets = 0
    for result in store_and_get_result:
        store_time, get_time, successful_gets, gets = result

        total_store_time += store_time
        total_get_time += get_time
        total_successful_gets += successful_gets
        total_gets += gets

    alive_peers = [peer.is_alive() for peer in peers]
    logger.info(
        f"Average store time per worker: {total_store_time / num_threads} "
        + f"Average get time per worker: {total_get_time / num_threads}"
    )
    logger.info(
        f"Total get succcess rate: {total_successful_gets / total_gets * 100:.1f}% "
        + f"({total_successful_gets}/{total_gets})"
    )
    logger.info(f"Total benchmark time: {benchmark_total_time:.5f} sec.")
    logger.info(f"Node survival rate: {len(alive_peers) / len(peers) * 100:.3f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_peers", type=int, default=16, required=False)
    parser.add_argument("--initial_peers", type=int, default=4, required=False)
    parser.add_argument("--random_seed", type=int, default=30, required=False)
    parser.add_argument("--num_threads", type=int, default=100, required=False)
    parser.add_argument("--total_num_rounds", type=int, default=16, required=False)
    parser.add_argument("--num_store_peers", type=int, default=8, required=False)
    parser.add_argument("--num_get_peers", type=int, default=8, required=False)
    parser.add_argument("--wait_after_iteration", type=float, default=0, required=False)
    parser.add_argument("--delay", type=float, default=0, required=False)
    parser.add_argument("--wait_timeout", type=float, default=5, required=False)
    parser.add_argument("--expiration", type=float, default=300, required=False)
    parser.add_argument("--latest", type=bool, default=True, required=False)
    parser.add_argument("--failure_rate", type=float, default=0.1, required=False)

    args = vars(parser.parse_args())

    asyncio.run(benchmark_dht(**args))
