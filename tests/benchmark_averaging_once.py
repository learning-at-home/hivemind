import random
import time

import torch
import hivemind
from hivemind.utils import LOCALHOST
import threading

dht_root = hivemind.DHT(listen_on=f'{LOCALHOST}:*', start=True)
trigger = threading.Condition()

num_peers = 50

peer_tensors = [(torch.randn(123), torch.full((3,), fill_value=float(random.randint(-5, 5))))
                for _ in range(num_peers)]

peer_average_estimates = [None for _ in range(num_peers)]

stuff = set()


def run_averager(index):
    dht = hivemind.DHT(listen_on=f'{LOCALHOST}:*', initial_peers=[f"{LOCALHOST}:{dht_root.port}"],
                       start=True)
    averager = hivemind.DecentralizedAverager(
        peer_tensors[i], dht, prefix='my_tensor', initial_group_bits='0110',
        listen_on=f"{LOCALHOST}:{3000 + index}",
        target_group_size=8, averaging_expiration=10, start=True)

    print(end=f'<{index}', flush=True)
    with trigger:
        trigger.wait()

    print(end=f'>{index}', flush=True)
    averager.step()
    stuff.add(dht), stuff.add(averager)


threads = []
for i in range(num_peers):
    thread = threading.Thread(target=run_averager, args=[i])
    threads.append(thread)
    thread.start()

print()
time.sleep(5)

t = time.time()

print('trigger now')
with trigger:
    trigger.notify_all()
print()

for thread in threads:
    thread.join()
print(f"test run took {time.time() - t:.3f} seconds")