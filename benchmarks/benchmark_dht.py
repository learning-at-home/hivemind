import argparse
import random
import time
import uuid

import asyncio

from tqdm import trange

import hivemind
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


async def store_task(peer, key, value, expiration):
    subkey = uuid.uuid4().hex
    
    store_ok = await peer.store(
        key, subkey=subkey, value=value, 
        expiration_time=hivemind.get_dht_time() + expiration, return_future=True
    )

    return store_ok


async def get_task(peer, key):
    latest = bool(random.getrandbits(1))
    value, expiration = await peer.get(key, latest=latest, return_future=True)

    return value, expiration


async def corouting_task(
    peers: list,
    total_num_rounds: int,
    num_store_peers: int,
    num_get_peers: int,
    wait_after_iteration: float,
    wait_before_read: float,
    expiration: float,
):
    successful_stores = total_stores = total_store_time = 0
    successful_gets = total_get_time = 0
    
    for _ in range(total_num_rounds):
        key = uuid.uuid4().hex
        
        store_start = time.perf_counter()
        store_peers = random.sample(peers, min(num_store_peers, len(peers)))
        store_tasks = [
            store_task(peer, key, 'yes', expiration) for peer in store_peers
        ]
        store_result, _ = await asyncio.wait(
            store_tasks, return_when=asyncio.ALL_COMPLETED
        )
        total_store_time += time.perf_counter() - store_start

        total_stores += len(store_result)
        successful_stores += sum(map(bool, [
            result.result() for result in store_result
        ]))
        await asyncio.sleep(wait_before_read)
        
        get_start = time.perf_counter()
        get_peers = random.sample(peers, min(num_get_peers, len(peers)))
        get_tasks = [
            get_task(peer, key) for peer in get_peers
        ]
        get_result, _ = await asyncio.wait(
            get_tasks, return_when=asyncio.ALL_COMPLETED
        )
        for result in get_result:
            if result.result()[0]:
                successful_gets += 1
        
        total_get_time += time.perf_counter() - get_start

        await asyncio.sleep(wait_after_iteration)

    logger.info(
        f"Store success rate: {successful_stores / total_stores * 100:.1f}% \
        ({successful_stores} / {total_stores})"
    )
    logger.info(f"Mean store time: {total_store_time / total_stores:.5}, \
        Total: {total_store_time:.5}")
    logger.info(
        f"Get success rate: \
        {successful_gets / num_get_peers / min(num_get_peers, len(peers)) * 100:.1f} \
        ({successful_gets} / {num_get_peers * min(num_get_peers, len(peers))})"
    )
    logger.info(
        f"Mean get time: \
        {total_get_time / num_get_peers / total_num_rounds:.5f}, \
        Total: {total_get_time:.5f}"
    )
            

def benchmark_dht(
    num_peers: int,
    initial_peers: int,
    random_seed: int,
    num_threads: int,
    total_num_rounds: int,
    num_store_peers: int,
    num_get_peers: int,
    wait_after_iteration: float,
    wait_before_read: float,
    wait_timeout: float,
    expiration: float,
):
    random.seed(random_seed)
    
    logger.info("Creating peers...")
    peers = []
    for _ in trange(num_peers):
        neighbors = sum(
            [peer.get_visible_maddrs() for peer in random.sample(
                peers, min(initial_peers, len(peers))
            )], []
        )
        peer = hivemind.DHT(initial_peers=neighbors, start=True, 
                wait_timeout=wait_timeout)
        peers.append(peer)

    logger.info("Creating coroutines...")
    loop = asyncio.get_event_loop()

    task_list = [
        loop.create_task(
            corouting_task(peers, total_num_rounds, num_store_peers, 
                num_get_peers, wait_after_iteration, wait_before_read,
                expiration)
        ) for _ in range(num_threads)
    ]

    loop.run_until_complete(asyncio.wait(task_list))
    loop.close()

    alive_peers = [peer.is_alive() for peer in peers]
    logger.info(
        f"Node survival rate: {len(alive_peers) / len(peers) * 100:.3f}%"
    )

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_peers", type=int, default=32, required=False)
    parser.add_argument("--initial_peers", type=int, default=1, required=False)
    parser.add_argument("--random_seed", type=int, default=random.randint(1, 1000))
    parser.add_argument("--num_threads", type=int, default=16, required=False)
    parser.add_argument("--total_num_rounds", type=int, default=8, required=False)
    parser.add_argument("--num_store_peers", type=int, default=8, required=False)
    parser.add_argument("--num_get_peers", type=int, default=8, required=False)
    parser.add_argument("--wait_after_iteration", type=float, default=0, required=False)
    parser.add_argument("--wait_before_read", type=float, default=0, required=False)
    parser.add_argument("--wait_timeout", type=float, default=5, required=False)
    parser.add_argument("--expiration", type=float, default=300, required=False)

    args = vars(parser.parse_args())

    benchmark_dht(**args)