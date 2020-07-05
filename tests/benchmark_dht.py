import time
import argparse
import random
from typing import Tuple
from warnings import warn
import hivemind
from tqdm import trange

from test_utils import increase_file_limit


def random_endpoint() -> Tuple[str, int]:
    return (f"{random.randint(0, 256)}.{random.randint(0, 256)}."
            f"{random.randint(0, 256)}.{random.randint(0, 256)}", random.randint(0, 65535))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_peers', type=int, default=32, required=False)
    parser.add_argument('--initial_peers', type=int, default=1, required=False)
    parser.add_argument('--num_experts', type=int, default=256, required=False)
    parser.add_argument('--expert_batch_size', type=int, default=32, required=False)
    parser.add_argument('--expiration_time', type=float, default=300, required=False)
    parser.add_argument('--wait_after_request', type=float, default=0, required=False)
    parser.add_argument('--wait_before_read', type=float, default=0, required=False)
    parser.add_argument('--wait_timeout', type=float, default=5, required=False)
    parser.add_argument('--random_seed', type=int, default=random.randint(1, 1000))
    parser.add_argument('--increase_file_limit', action="store_true")
    args = parser.parse_args()

    if args.increase_file_limit:
        increase_file_limit()

    random.seed(args.random_seed)
    hivemind.DHT.EXPIRATION = args.expiration_time

    print("Creating peers...")
    peers = []
    for _ in trange(args.num_peers):
        neighbors = [f'0.0.0.0:{node.port}' for node in random.sample(peers, min(args.initial_peers, len(peers)))]
        peer = hivemind.DHT(*neighbors, start=True, wait_timeout=args.wait_timeout, listen_on=f'0.0.0.0:*')
        peers.append(peer)

    store_peer, get_peer = peers[-2:]

    expert_uids = list(set(f"expert.{random.randint(0, 999)}.{random.randint(0, 999)}.{random.randint(0, 999)}"
                           for _ in range(args.num_experts)))
    print(f"Sampled {len(expert_uids)} unique ids (after deduplication)")
    random.shuffle(expert_uids)

    print(f"Storing peers to dht in batches of {args.expert_batch_size}...")
    successful_stores = total_stores = total_store_time = 0
    benchmark_started = time.perf_counter()
    endpoints = []

    for start in trange(0, args.num_experts, args.expert_batch_size):
        store_start = time.perf_counter()
        endpoints.append(random_endpoint())
        success_list = store_peer.declare_experts(expert_uids[start: start + args.expert_batch_size], *endpoints[-1])
        total_store_time += time.perf_counter() - store_start

        total_stores += len(success_list)
        successful_stores += sum(success_list)
        time.sleep(args.wait_after_request)

    print(f"Store success rate: {successful_stores / total_stores * 100:.1f}% ({successful_stores} / {total_stores})")
    print(f"Mean store time: {total_store_time / total_stores:.5}, Total: {total_store_time:.5}")
    time.sleep(args.wait_before_read)

    if time.perf_counter() - benchmark_started > args.expiration_time:
        warn("Warning: all keys expired before benchmark started getting them. Consider increasing expiration_time")

    successful_gets = total_get_time = 0

    for start in trange(0, len(expert_uids), args.expert_batch_size):
        get_start = time.perf_counter()
        get_result = get_peer.get_experts(expert_uids[start: start + args.expert_batch_size])
        total_get_time += time.perf_counter() - get_start

        for i, expert in enumerate(get_result):
            if expert is not None and expert.uid == expert_uids[start + i] \
                    and (expert.host, expert.port) == endpoints[start // args.expert_batch_size]:
                successful_gets += 1

    if time.perf_counter() - benchmark_started > args.expiration_time:
        warn("Warning: keys expired midway during get requests. If that is not desired, increase expiration_time param")

    print(f"Get success rate: {successful_gets / len(expert_uids) * 100:.1f} ({successful_gets} / {len(expert_uids)})")
    print(f"Mean get time: {total_get_time / len(expert_uids):.5f}, Total: {total_get_time:.5f}")

    alive_peers = [peer.is_alive() for peer in peers]
    print(f"Node survival rate: {len(alive_peers) / len(peers) * 100:.3f}%")
