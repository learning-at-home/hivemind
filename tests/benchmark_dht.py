import time
import argparse
import random
import resource
from typing import Tuple

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
    parser.add_argument('--sleep_after_request', type=float, default=0, required=False)
    parser.add_argument('--expiration_time', type=float, default=10, required=False)
    parser.add_argument('--wait_before_read', type=float, default=1, required=False)
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
    request_period = args.request_period
    expiration_time = args.expiration_time
    wait_before_read = args.wait_before_read
    random_seed = args.random_seed

    hivemind.DHT.EXPIRATION = 3600
    ip = "0.0.0.0"

    peers = [hivemind.DHT(start=True, wait_timeout=30, listen_on=f'0.0.0.0:*')]
    for i in range(num_nodes):
        neighbors_i = [f'{ip}:{node.port}' for node in random.sample(peers, min(num_neighbors, len(peers)))]
        peer = hivemind.DHT(*neighbors_i, listen_on=f'0.0.0.0:*', start=True)
        peers.append(peer)

    store_peer = hivemind.DHT(start=True, wait_timeout=30, listen_on=f'0.0.0.0:*')
    get_peer = hivemind.DHT(start=True, wait_timeout=30, listen_on=f'0.0.0.0:*')

    random.seed(random_seed)
    expert_uids = list(set(f"expert.{random.randint(0, 999)}.{random.randint(0, 999)}.{random.randint(0, 999)}"
                           for _ in range(num_experts)))
    print(f"Sampled {len(expert_uids)} unique ids (after deduplication)")
    random.shuffle(expert_uids)

    success_store = 0
    all_store = 0
    time_store = 0
    success_get = 0
    all_get = 0
    time_get = 0
    batch_endpoints = []

    for start in range(0, num_experts, experts_batch_size):
        store_start = time.time()
        batch_endpoints.append(random_endpoint())
        success_list = store_peer.declare_experts(expert_uids[start: start + experts_batch_size], *batch_endpoints[-1])
        time_store += time.time() - store_start

        all_store += len(success_list)
        success_store += sum(success_list)

        time.sleep(request_period)

    get_time = time.time()
    get_result = get_peer.get_experts(expert_uids[:, 0])
    get_time = time.time() - get_time

    for i in range(num_experts):
        expert = get_result[i]
        if expert is not None and [expert.uid, expert.host, expert.port] == experts[i]:
            success_get += 1
    all_get = num_experts

    print("store success rate: ", success_store / all_store)
    print("mean store time: ", time_store / all_store)
    print("get success rate: ", success_get / all_get)
    print("mean get time: ", time_get / all_get)
    print("death rate: ", (num_nodes - alive_nodes_count) / num_nodes)