import argparse
import random
import time

from tqdm import trange

import hivemind
import hivemind.client.expert_uid
import hivemind.client.dht_ops
from hivemind.utils.threading import increase_file_limit

logger = hivemind.get_logger(__name__)


def random_endpoint() -> hivemind.Endpoint:
    return f"{random.randint(0, 256)}.{random.randint(0, 256)}.{random.randint(0, 256)}." \
           f"{random.randint(0, 256)}:{random.randint(0, 65535)}"


def benchmark_dht(num_peers: int, initial_peers: int, num_experts: int, expert_batch_size: int, random_seed: int,
                  wait_after_request: float, wait_before_read: float, wait_timeout: float, expiration: float):
    random.seed(random_seed)

    print("Creating peers...")
    peers = []
    for _ in trange(num_peers):
        neighbors = [f'0.0.0.0:{node.port}' for node in random.sample(peers, min(initial_peers, len(peers)))]
        peer = hivemind.DHT(initial_peers=neighbors, start=True, wait_timeout=wait_timeout,
                            expiration=expiration, listen_on=f'0.0.0.0:*')
        peers.append(peer)

    store_peer, get_peer = peers[-2:]

    expert_uids = list(set(f"expert.{random.randint(0, 999)}.{random.randint(0, 999)}.{random.randint(0, 999)}"
                           for _ in range(num_experts)))
    print(f"Sampled {len(expert_uids)} unique ids (after deduplication)")
    random.shuffle(expert_uids)

    print(f"Storing experts to dht in batches of {expert_batch_size}...")
    successful_stores = total_stores = total_store_time = 0
    benchmark_started = time.perf_counter()
    endpoints = []

    for start in trange(0, num_experts, expert_batch_size):
        store_start = time.perf_counter()
        endpoints.append(random_endpoint())
        store_ok = hivemind.declare_experts(store_peer, expert_uids[start: start + expert_batch_size], endpoints[-1])
        successes = store_ok.values()
        total_store_time += time.perf_counter() - store_start

        total_stores += len(successes)
        successful_stores += sum(successes)
        time.sleep(wait_after_request)

    print(f"Store success rate: {successful_stores / total_stores * 100:.1f}% ({successful_stores} / {total_stores})")
    print(f"Mean store time: {total_store_time / total_stores:.5}, Total: {total_store_time:.5}")
    time.sleep(wait_before_read)

    if time.perf_counter() - benchmark_started > expiration:
        logger.warning("All keys expired before benchmark started getting them. Consider increasing expiration_time")

    successful_gets = total_get_time = 0

    for start in trange(0, len(expert_uids), expert_batch_size):
        get_start = time.perf_counter()
        get_result = hivemind.get_experts(get_peer, expert_uids[start: start + expert_batch_size])
        total_get_time += time.perf_counter() - get_start

        for i, expert in enumerate(get_result):
            if expert is not None and expert.uid == expert_uids[start + i] \
                    and expert.endpoint == endpoints[start // expert_batch_size]:
                successful_gets += 1

    if time.perf_counter() - benchmark_started > expiration:
        logger.warning("keys expired midway during get requests. If that isn't desired, increase expiration_time param")

    print(f"Get success rate: {successful_gets / len(expert_uids) * 100:.1f} ({successful_gets} / {len(expert_uids)})")
    print(f"Mean get time: {total_get_time / len(expert_uids):.5f}, Total: {total_get_time:.5f}")

    alive_peers = [peer.is_alive() for peer in peers]
    print(f"Node survival rate: {len(alive_peers) / len(peers) * 100:.3f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_peers', type=int, default=32, required=False)
    parser.add_argument('--initial_peers', type=int, default=1, required=False)
    parser.add_argument('--num_experts', type=int, default=256, required=False)
    parser.add_argument('--expert_batch_size', type=int, default=32, required=False)
    parser.add_argument('--expiration', type=float, default=300, required=False)
    parser.add_argument('--wait_after_request', type=float, default=0, required=False)
    parser.add_argument('--wait_before_read', type=float, default=0, required=False)
    parser.add_argument('--wait_timeout', type=float, default=5, required=False)
    parser.add_argument('--random_seed', type=int, default=random.randint(1, 1000))
    parser.add_argument('--increase_file_limit', action="store_true")
    args = vars(parser.parse_args())

    if args.pop('increase_file_limit', False):
        increase_file_limit()

    benchmark_dht(**args)
