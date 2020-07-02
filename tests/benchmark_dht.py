import time
import argparse
import random
import resource
from typing import Tuple
from warnings import warn

from dask.dataframe.multi import required

import hivemind


def random_endpoint() -> Tuple[str, int]:
    return (f"{random.randint(0, 256)}.{random.randint(0, 256)}."
            f"{random.randint(0, 256)}.{random.randint(0, 256)}", random.randint(0, 65535))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=20, required=False)
    parser.add_argument('--num_neighbors', type=int, default=20, required=False)
    parser.add_argument('--num_experts', type=int, default=20, required=False)
    parser.add_argument('--experts_batch_size', type=int, default=20, required=False)
    parser.add_argument('--expiration_time', type=float, default=300, required=False)
    parser.add_argument('--wait_after_request', type=float, default=0, required=False)
    parser.add_argument('--wait_before_read', type=float, default=0, required=False)
    parser.add_argument('--wait_timeout', type=float, default=5, required=False)
    parser.add_argument('--random_seed', type=int, default=random.randint(1, 1000))
    parser.add_argument('--increase_file_limit', action="store_true")
    args = parser.parse_args()

    if args.increase_file_limit:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 2 ** 15), max(hard, 2 ** 15)))

    num_nodes = args.num_nodes
    num_neighbors = args.num_neighbors
    num_experts = args.num_experts
    experts_batch_size = args.experts_batch_size
    expiration_time = args.expiration_time
    wait_after_request = args.wait_after_request
    wait_before_read = args.wait_before_read
    wait_timeout = args.wait_timeout
    random.seed(args.random_seed)

    hivemind.DHT.EXPIRATION = expiration_time

    peers = [hivemind.DHT(start=True, wait_timeout=wait_timeout, listen_on=f'0.0.0.0:*')]
    for i in range(num_nodes - 1):
        neighbors_i = [f'0.0.0.0:{node.port}' for node in random.sample(peers, min(num_neighbors, len(peers)))]
        peer = hivemind.DHT(*neighbors_i, start=True, wait_timeout=wait_timeout, listen_on=f'0.0.0.0:*')
        peers.append(peer)

    store_peer, get_peer = peers[-2:]

    expert_uids = list(set(f"expert.{random.randint(0, 999)}.{random.randint(0, 999)}.{random.randint(0, 999)}"
                           for _ in range(num_experts)))
    print(f"Sampled {len(expert_uids)} unique ids (after deduplication)")
    random.shuffle(expert_uids)

    successful_stores = total_stores = total_store_time = 0
    benchmark_started = time.perf_counter()
    batch_endpoints = []

    for start in range(0, num_experts, experts_batch_size):
        store_start = time.perf_counter()
        batch_endpoints.append(random_endpoint())
        success_list = store_peer.declare_experts(expert_uids[start: start + experts_batch_size], *batch_endpoints[-1])
        total_store_time += time.perf_counter() - store_start

        total_stores += len(success_list)
        successful_stores += sum(success_list)
        time.sleep(wait_after_request)

    print(f"store success rate: {successful_stores / total_stores} ({successful_stores} / {total_stores})")
    print(f"mean store time: {total_store_time / total_stores}, Total: {total_store_time}")
    time.sleep(wait_before_read)

    if time.perf_counter() - benchmark_started > args.expiration_time:
        warn("Warning: all keys expired before benchmark started getting them. Consider increasing expiration_time")

    successful_gets = total_get_time = 0

    for start in range(0, num_experts, experts_batch_size):
        get_start = time.perf_counter()
        get_result = get_peer.get_experts(expert_uids[start: start + experts_batch_size])
        total_get_time += time.perf_counter() - get_start

        for i, expert in enumerate(get_result):
            if expert is not None and expert.uid == expert_uids[start+i] \
                    and [expert.host, expert.port] == batch_endpoints[start//experts_batch_size]:
                successful_gets += 1

    if time.perf_counter() - benchmark_started > args.expiration_time:
        warn("Warning: keys expired midway during get requests. If that is not desired, increase expiration_time param")

    print(f"Get success rate: {successful_gets / len(expert_uids)} ({successful_gets} / {len(expert_uids)})")
    print(f"Mean get time: {total_get_time / len(expert_uids)}, Total: {total_get_time}")

    alive_peers = [peer.is_alive() for peer in peers]
    print(f"Node survival rate: {len(alive_peers) / len(peers) * 100}%")
