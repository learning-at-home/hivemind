import asyncio
import random
import time
import torch
import hivemind
from hivemind.client.averaging.allreduce import AllReduceProtocol, split_into_parts, restore_from_parts
from hivemind.utils import LOCALHOST, Endpoint
import threading
import logging
import hivemind
hivemind.utils.increase_file_limit(4096, 4096)

dht_root = hivemind.DHT(listen_on=f'{LOCALHOST}:*', start=True)
trigger = threading.Condition()

num_peers = 128

peer_tensors = [(torch.randn(123), torch.full((3,), fill_value=float(random.randint(-5, 5))))
                for _ in range(num_peers)]

peer_average_estimates = [None for _ in range(num_peers)]

stuff = {dht_root}


def run_averager(index):
    dht = hivemind.DHT(listen_on=f'{LOCALHOST}:*', initial_peers=[f"{LOCALHOST}:{dht_root.port}"],
                       start=True)
    averager = hivemind.DecentralizedAverager(
        peer_tensors[i], dht, prefix='my_tensor', initial_group_bits='0110',
        listen_on=f"{LOCALHOST}:{4000 + index}",
        target_group_size=32, averaging_expiration=10, start=True)

    print(end=f'<<{index}\n', flush=True)

    for _ in range(100):
        success = averager.step(timeout=90) is not None
        print(end=('+' if success else '-'), flush=True)
    stuff.add(dht), stuff.add(averager)
    print(end=f'>>{index}\n', flush=True)



threads = []
for i in range(num_peers):
    thread = threading.Thread(target=run_averager, args=[i])
    threads.append(thread)
    thread.start()
    time.sleep(0.1)

print()

t = time.time()
for thread in threads:
    thread.join()
print()
print(f"test run took {time.time() - t:.3f} seconds")
for peer in stuff:
    peer.terminate()