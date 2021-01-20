import math
import time
import threading
import argparse

import torch
import hivemind
from hivemind.utils import LOCALHOST, increase_file_limit
from hivemind.proto import runtime_pb2


def sample_tensors(hid_size, num_layers):
    tensors = []
    for i in range(num_layers):
        tensors.append(torch.randn(hid_size, 3 * hid_size))
        tensors.append(torch.randn(3 * hid_size))
        tensors.append(torch.randn(3 * hid_size))
        tensors.append(torch.randn(hid_size, hid_size))
        tensors.append(torch.ones(hid_size))
        tensors.append(torch.zeros(hid_size))
        tensors.append(torch.randn(hid_size, 4 * hid_size))
        tensors.append(torch.randn(4 * hid_size))
        tensors.append(torch.ones(4 * hid_size))
        tensors.append(torch.randn(2, hid_size, hid_size, 2))
        tensors.append(torch.randn(hid_size))
        tensors.append(torch.randn(hid_size))
        tensors.append(torch.randn(hid_size))
    return tuple(tensors)


def benchmark_averaging(num_peers: int, target_group_size: int, num_rounds: int,
                        averaging_expiration: float, request_timeout: float, round_timeout: float,
                        hid_size: int, num_layers: int, spawn_dtime: float):
    dht_root = hivemind.DHT(listen_on=f'{LOCALHOST}:*', start=True)
    num_groups = 2 ** int(round(math.log2(num_peers / target_group_size)))
    nbits = int(round(math.log2(num_groups)))
    peer_tensors = [sample_tensors(hid_size, num_layers)
                    for _ in range(num_peers)]
    processes = {dht_root}

    def run_averager(index):
        dht = hivemind.DHT(listen_on=f'{LOCALHOST}:*',
                           initial_peers=[f"{LOCALHOST}:{dht_root.port}"],
                           start=True)
        initial_bits = bin(index % num_groups)[2:].rjust(nbits, '0')
        averager = hivemind.DecentralizedAverager(
            peer_tensors[i], dht, prefix='my_tensor', initial_group_bits=initial_bits, listen_on=f"{LOCALHOST}:*",
            compression_type=runtime_pb2.CompressionType.FLOAT16, target_group_size=target_group_size,
            averaging_expiration=averaging_expiration, request_timeout=request_timeout, start=True)
        processes.update({dht, averager})

        print(end=f'<started {index}>\n', flush=True)
        for _ in range(num_rounds):
            success = averager.step(timeout=round_timeout)
            print(end=('+' if success else '-'), flush=True)
        print(end=f'<finished {index}>\n', flush=True)

    threads = []
    for i in range(num_peers):
        thread = threading.Thread(target=run_averager, args=[i])
        threads.append(thread)
        thread.start()
        time.sleep(spawn_dtime)

    t = time.time()
    for thread in threads:
        thread.join()

    print(f"\ntest run took {time.time() - t:.3f} seconds")

    for process in processes:
        process.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_peers', type=int, default=16, required=False)
    parser.add_argument('--target_group_size', type=int, default=4, required=False)
    parser.add_argument('--num_rounds', type=int, default=5, required=False)
    parser.add_argument('--hid_size', type=int, default=256, required=False)
    parser.add_argument('--num_layers', type=int, default=3, required=False)
    parser.add_argument('--averaging_expiration', type=float, default=15, required=False)
    parser.add_argument('--round_timeout', type=float, default=30, required=False)
    parser.add_argument('--request_timeout', type=float, default=3, required=False)
    parser.add_argument('--spawn_dtime', type=float, default=0.1, required=False)
    parser.add_argument('--increase_file_limit', action="store_true")
    args = vars(parser.parse_args())

    if args.pop('increase_file_limit', False):
        increase_file_limit()

    benchmark_averaging(**args)
