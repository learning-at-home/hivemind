import argparse
import math
import threading
import time

import torch

import hivemind
from hivemind.compression import Float16Compression
from hivemind.utils.limits import increase_file_limit
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


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


def benchmark_averaging(
    num_peers: int,
    target_group_size: int,
    num_rounds: int,
    min_matchmaking_time: float,
    request_timeout: float,
    round_timeout: float,
    hid_size: int,
    num_layers: int,
    spawn_dtime: float,
):
    dht_root = hivemind.DHT(start=True)
    initial_peers = dht_root.get_visible_maddrs()

    num_groups = 2 ** int(round(math.log2(num_peers / target_group_size)))
    nbits = int(round(math.log2(num_groups)))
    peer_tensors = [sample_tensors(hid_size, num_layers) for _ in range(num_peers)]
    processes = {dht_root}
    lock_stats = threading.Lock()
    successful_steps = total_steps = 0

    def run_averager(index):
        nonlocal successful_steps, total_steps, lock_stats
        dht = hivemind.DHT(initial_peers=initial_peers, start=True)
        initial_bits = bin(index % num_groups)[2:].rjust(nbits, "0")
        averager = hivemind.averaging.DecentralizedAverager(
            peer_tensors[index],
            dht,
            prefix="my_tensor",
            initial_group_bits=initial_bits,
            compression=Float16Compression(),
            target_group_size=target_group_size,
            min_matchmaking_time=min_matchmaking_time,
            request_timeout=request_timeout,
            start=True,
        )
        processes.update({dht, averager})

        logger.info(
            f"Averager {index}: started with peer id {averager.peer_id}, group_bits: {averager.get_group_bits()}"
        )
        for step in range(num_rounds):
            try:
                success = averager.step(timeout=round_timeout) is not None
            except:
                success = False
            with lock_stats:
                successful_steps += int(success)
                total_steps += 1
            logger.info(f"Averager {index}: {'finished' if success else 'failed'} step #{step}")
        logger.info(f"Averager {index}: done.")

    threads = []
    for i in range(num_peers):
        thread = threading.Thread(target=run_averager, args=[i])
        threads.append(thread)
        thread.start()
        time.sleep(spawn_dtime)

    t = time.time()
    for thread in threads:
        thread.join()

    logger.info(f"Benchmark finished in {time.time() - t:.3f} seconds.")
    logger.info(f"Success rate: {successful_steps / total_steps} ({successful_steps} out of {total_steps} attempts)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_peers", type=int, default=16, required=False)
    parser.add_argument("--target_group_size", type=int, default=4, required=False)
    parser.add_argument("--num_rounds", type=int, default=5, required=False)
    parser.add_argument("--hid_size", type=int, default=256, required=False)
    parser.add_argument("--num_layers", type=int, default=3, required=False)
    parser.add_argument("--min_matchmaking_time", type=float, default=5, required=False)
    parser.add_argument("--round_timeout", type=float, default=15, required=False)
    parser.add_argument("--request_timeout", type=float, default=1, required=False)
    parser.add_argument("--spawn_dtime", type=float, default=0.1, required=False)
    parser.add_argument("--increase_file_limit", action="store_true")
    args = vars(parser.parse_args())

    if args.pop("increase_file_limit", False):
        increase_file_limit()

    benchmark_averaging(**args)
