import argparse
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 2 ** 15), max(hard, 2 ** 15)))
import hivemind, random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=20, required=False)
    parser.add_argument('--num_neighbors', type=int, default=20, required=False)
    parser.add_argument('--num_experts', type=int, default=20, required=False)
    parser.add_argument('--experts_batch_size', type=int, default=20, required=False)
    parser.add_argument('--request_period', type=float, default=2, required=False)
    parser.add_argument('--expiration_time', type=float, default=10, required=False)
    parser.add_argument('--wait_before_read', type=float, default=1, required=False)
    parser.add_argument('--random_seed', type=int, default=random.randint(1, 1000))
    args = parser.parse_args()

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

    peers = [hivemind.DHT(start=True, wait_timeout=30, listen_on=f'0.0.0.0:{15000}')]
    store_peer = hivemind.DHT(start=True, wait_timeout=30, listen_on=f'0.0.0.0:{15000+num_nodes+2}')
    get_peer = hivemind.DHT(start=True, wait_timeout=30, listen_on=f'0.0.0.0:{15000+num_nodes+3}')
    for i in range(num_nodes):
        neighbors_i = [f'{ip}:{node.port}' for node in random.sample(peers, min(num_neighbors, len(peers)))]
        peer = hivemind.DHT(*neighbors_i, listen_on=f'0.0.0.0:{15000 + i + 1}', start=True)
        peers.append(peer)

    random.seed(random_seed)
    expert_uids = list(set(f"expert.{random.randint(0, 999)}.{random.randint(0, 999)}.{random.randint(0, 999)}" for _ in range(num_experts)))
    expert_uids = [[i, *random_addres_port()] for i in expert_uids]
    random.shuffle(expert_uids)

    success_store = 0
    all_store = 0
    time_store = 0
    success_get = 0
    all_get = 0
    time_get = 0

    for i in range(num_experts//experts_batch_size):
        store_start = time.time()
        succes_list = store_peer.declare_experts(*expert_uids[i*experts_batch_size:(i+1)*experts_batch_size])
        time_store += time.time() - store_start

        all_store += len(succes_list)
        success_store += sum(succes_list)

        time.sleep(request_period)

    get_time = time.time()
    get_result = get_peer.get_experts(expert_uids[:,0])
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