import argparse
import multiprocessing as mp
import random
import sys
import time

import torch

from hivemind.dht import DHT
from hivemind.moe.client.expert import RemoteExpert
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.moe.expert_uid import ExpertInfo
from hivemind.moe.server import ModuleBackend, Server
from hivemind.moe.server.layers import name_to_block
from hivemind.p2p import P2P
from hivemind.utils.limits import increase_file_limit
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.utils.tensor_descr import BatchTensorDescriptor

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


def print_device_info(device=None):
    """Prints device stats. Code from https://stackoverflow.com/a/53374933/12891528"""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # Additional Info when using cuda
    if device.type == "cuda":
        logger.info(torch.cuda.get_device_name(0))
        logger.info(f"Memory Usage:")
        logger.info(f"Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB")
        logger.info(f"Cached:   {round(torch.cuda.memory_cached(0) / 1024 ** 3, 1)} GB")


def client_process(
    can_start,
    benchmarking_failed,
    server_maddrs,
    server_peer_id,
    num_experts,
    batch_size,
    hid_dim,
    num_batches,
    backprop=True,
) -> None:
    torch.set_num_threads(1)
    can_start.wait()

    p2p = RemoteExpertWorker.run_coroutine(P2P.create(initial_peers=server_maddrs))
    experts = [
        RemoteExpert(expert_info=ExpertInfo(uid=f"expert.{i}", peer_id=server_peer_id), p2p=p2p)
        for i in range(num_experts)
    ]

    try:
        dummy_batch = torch.randn(batch_size, hid_dim)
        for _ in range(num_batches):
            expert = random.choice(experts)
            out = expert(dummy_batch)
            if backprop:
                out.sum().backward()
    except BaseException as e:
        benchmarking_failed.set()
        raise e


def benchmark_throughput(
    num_experts=16,
    num_handlers=None,
    num_clients=128,
    num_batches_per_client=16,
    expert_cls="ffn",
    hid_dim=1024,
    batch_size=2048,
    max_batch_size=None,
    backprop=True,
    device=None,
):
    assert (
        not hasattr(torch.cuda, "is_initialized")
        or not torch.cuda.is_initialized()
        or torch.device(device) == torch.device("cpu")
    )
    assert expert_cls in name_to_block
    max_batch_size = max_batch_size or batch_size * 4
    num_handlers = max(1, num_handlers or num_clients // 2)
    benchmarking_failed = mp.Event()
    can_start = mp.Event()
    timestamps = dict(started=time.perf_counter())

    try:
        server_dht = DHT(start=True)
        clients = [
            mp.Process(
                target=client_process,
                name=f"client_process-{i}",
                args=(
                    can_start,
                    benchmarking_failed,
                    server_dht.get_visible_maddrs(),
                    server_dht.peer_id,
                    num_experts,
                    batch_size,
                    hid_dim,
                    num_batches_per_client,
                    backprop,
                ),
                daemon=True,
            )
            for i in range(num_clients)
        ]

        for client in clients:
            client.start()

        timestamps["launched_clients"] = timestamps["began_launching_server"] = time.perf_counter()

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        module_backends = {}
        for i in range(num_experts):
            expert = torch.jit.script(name_to_block[expert_cls](hid_dim))
            module_backends[f"expert.{i}"] = ModuleBackend(
                name=f"expert.{i}",
                module=expert,
                optimizer=torch.optim.Adam(expert.parameters()),
                args_schema=(BatchTensorDescriptor(hid_dim),),
                outputs_schema=BatchTensorDescriptor(hid_dim),
                max_batch_size=max_batch_size,
            )
        timestamps["created_experts"] = time.perf_counter()

        server = Server(
            dht=server_dht,
            module_backends=module_backends,
            num_connection_handlers=num_handlers,
            device=device,
        )
        server.start()
        server.ready.wait()

        timestamps["server_ready"] = time.perf_counter()
        can_start.set()

        for client in clients:
            client.join()

        timestamps["clients_finished"] = time.perf_counter()

    except BaseException as e:
        benchmarking_failed.set()
        raise e
    finally:
        for client in clients:
            if client.is_alive():
                client.terminate()
        server.shutdown()
        timestamps["server_shutdown_finished"] = time.perf_counter()
        server.join()

    sys.stdout.flush()
    sys.stderr.flush()
    time_between = (
        lambda key1, key2: abs(timestamps[key2] - timestamps[key1])
        if (key1 in timestamps and key2 in timestamps)
        else float("nan")
    )
    total_examples = batch_size * num_clients * num_batches_per_client

    logger.info("Benchmark finished, status:" + ["Success", "Failure"][benchmarking_failed.is_set()])
    logger.info(
        f"Server parameters: num_experts={num_experts}, num_handlers={num_handlers}, "
        f"max_batch_size={max_batch_size}, expert_cls={expert_cls}, hid_dim={hid_dim}, device={device}"
    )
    logger.info(
        f"Client parameters: num_clients={num_clients}, num_batches_per_client={num_batches_per_client}, "
        f"batch_size={batch_size}, backprop={backprop}"
    )

    logger.info("Results: ")
    logger.info(
        f"\tServer startup took {time_between('began_launching_server', 'server_ready') :.3f} s. "
        f"({time_between('began_launching_server', 'created_experts') :.3f} s. experts + "
        f"{time_between('created_experts', 'server_ready') :.3f} s. networking)"
    )
    logger.info(f"\tProcessed {total_examples} examples in {time_between('server_ready', 'clients_finished') :.3f}")
    logger.info(
        f"\tThroughput for {'forward + backward' if backprop else 'forward'} passes: "
        f"{total_examples / time_between('server_ready', 'clients_finished') :.3f} samples / s."
    )
    logger.info(f"\tBenchmarking took {time_between('started', 'server_shutdown_finished') :.3f} s.")
    if benchmarking_failed.is_set():
        logger.info("Note: benchmark code failed, timing/memory results only indicate time till failure!")
    print_device_info(device)
    sys.stdout.flush()
    sys.stderr.flush()

    assert not benchmarking_failed.is_set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="default", required=False)
    parser.add_argument("--num_batches_per_client", type=int, default=16, required=False)
    args = parser.parse_args()

    if args.preset in ("default", "ffn_forward_backward"):
        benchmark_throughput()
    elif args.preset == "ffn_forward":
        benchmark_throughput(backprop=False, num_batches_per_client=args.num_batches_per_client)
    elif args.preset == "ffn_small_batch":
        benchmark_throughput(
            backprop=False,
            num_experts=4,
            batch_size=32,
            max_batch_size=8192,
            num_batches_per_client=args.num_batches_per_client,
        )
    elif args.preset == "ffn_small_batch_512clients":
        benchmark_throughput(
            backprop=True,
            num_experts=1,
            batch_size=1,
            max_batch_size=8192,
            num_clients=512,
            num_batches_per_client=args.num_batches_per_client,
        )
    elif args.preset == "ffn_small_batch_512clients_32handlers":
        benchmark_throughput(
            backprop=True,
            num_experts=1,
            batch_size=1,
            max_batch_size=8192,
            num_handlers=32,
            num_clients=512,
            num_batches_per_client=args.num_batches_per_client,
        )
    elif args.preset == "ffn_massive":
        increase_file_limit()
        benchmark_throughput(
            backprop=False,
            num_clients=512,
            batch_size=512,
            max_batch_size=8192,
            num_batches_per_client=args.num_batches_per_client,
        )
    elif args.preset == "minimalistic":
        benchmark_throughput(
            num_experts=1, num_clients=1, num_handlers=1, num_batches_per_client=args.num_batches_per_client
        )
    elif args.preset == "nop":
        benchmark_throughput(expert_cls="nop", backprop=False, num_batches_per_client=args.num_batches_per_client)
    else:
        raise ValueError(f"No such benchmark preset: {args.preset}")
