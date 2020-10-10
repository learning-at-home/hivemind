import time
import random

import hivemind
from hivemind import LOCALHOST
from tqdm.auto import trange
import numpy as np
from test_utils import increase_file_limit

# TODO THIS FILE SHOULDN'T MAKE IT TO MASTER
# this is a performance test for beam search debugging;
# we will need to re-write it as an option for benchmark_dht.py before merging into master

if __name__ == '__main__':
    increase_file_limit()
    dht_size = 256
    total_experts = 4096
    batch_size = 65
    initial_peers = 3
    beam_size = 4
    parallel_rpc = 256
    grid_dims = [256, 256, 256]
    num_beam_searches = 10_000

    print("Spawning dht peers...")

    dht = []
    for i in trange(dht_size):
        neighbors_i = [f'{LOCALHOST}:{node.port}' for node in random.sample(dht, min(initial_peers, len(dht)))]
        dht.append(hivemind.DHT(start=True, expiration=999999, initial_peers=neighbors_i, parallel_rpc=parallel_rpc))

    print("Declaring experts...")
    real_experts = sorted({
        'expert.' + '.'.join([str(random.randint(0, dim)) for dim in grid_dims])
        for _ in range(total_experts)
    })
    for batch_start in trange(0, len(real_experts), batch_size):
        random.choice(dht).declare_experts(
            real_experts[batch_start: batch_start + batch_size], wait=True,
            endpoint=f"host{batch_start // batch_size}:{random.randint(0, 65536)}")

    print("Creating you...")
    neighbors_i = [f'{LOCALHOST}:{node.port}' for node in random.sample(dht, min(initial_peers, len(dht)))]
    you = hivemind.DHT(start=True, expiration=999999, initial_peers=neighbors_i, parallel_rpc=parallel_rpc)
    time.sleep(1)

    print("Running beam search...")
    for i in trange(num_beam_searches):
        batch_scores = [np.random.randn(batch_size, dim) for dim in grid_dims]
        topk_experts = you.batch_find_best_experts('expert', batch_scores, beam_size)
