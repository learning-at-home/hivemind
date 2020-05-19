import argparse
import multiprocessing as mp
import random
import resource
import sys
import time

import torch
from test_utils import layers, print_device_info

import hivemind
from hivemind import find_open_port


def client_process(can_start, benchmarking_failed, port, num_experts, batch_size, hid_dim, num_batches, backprop=True):
    torch.set_num_threads(1)
    can_start.wait()
    experts = [hivemind.RemoteExpert(f"expert{i}", port=port) for i in range(num_experts)]

    try:
        dummy_batch = torch.randn(batch_size, hid_dim)
        for batch_i in range(num_batches):
            expert = random.choice(experts)
            out = expert(dummy_batch)
            if backprop:
                out.sum().backward()
    except BaseException as e:
        benchmarking_failed.set()
        raise e


def benchmark_throughput(num_experts=16, num_handlers=None, num_clients=128, num_batches_per_client=16,
                         expert_cls='ffn', hid_dim=1024, batch_size=2048, max_batch_size=None, backprop=True,
                         device=None, port=None):
    assert not hasattr(torch.cuda, 'is_initialized') or not torch.cuda.is_initialized() \
           or torch.device(device) == torch.device('cpu')
    assert expert_cls in layers.name_to_block
    port = port or find_open_port()
    max_batch_size = max_batch_size or batch_size * 4
    num_handlers = max(1, num_handlers or num_clients // 32)
    benchmarking_failed = mp.Event()
    can_start = mp.Event()
    timestamps = dict(started=time.perf_counter())

    try:
        # start clients and await server
        # Note: client processes must be launched BEFORE touching gpu, even torch.cuda.is_available can cause trouble
        clients = [
            mp.Process(
                target=client_process, name=f'client_process-{i}',
                args=(can_start, benchmarking_failed, port, num_experts, batch_size,
                      hid_dim, num_batches_per_client, backprop), daemon=True)
            for i in range(num_clients)]

        for client in clients:
            client.start()

        timestamps['launched_clients'] = timestamps['began_launching_server'] = time.perf_counter()

        # start server
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        experts = {}
        for i in range(num_experts):
            expert = torch.jit.script(layers.name_to_block[expert_cls](hid_dim))
            experts[f'expert{i}'] = hivemind.ExpertBackend(name=f'expert{i}',
                                                           expert=expert, opt=torch.optim.Adam(expert.parameters()),
                                                           args_schema=(hivemind.BatchTensorProto(hid_dim),),
                                                           outputs_schema=hivemind.BatchTensorProto(hid_dim),
                                                           max_batch_size=max_batch_size,
                                                           )
        timestamps['created_experts'] = time.perf_counter()
        server = hivemind.Server(None, experts, port=port, conn_handler_processes=num_handlers, device=device)
        server.start()
        server.ready.wait()
        timestamps['server_ready'] = time.perf_counter()
        can_start.set()

        for client in clients:
            client.join()
        timestamps['clients_finished'] = time.perf_counter()
    except BaseException as e:
        benchmarking_failed.set()
        raise e
    finally:
        for client in clients:
            if client.is_alive():
                client.terminate()
                client.join()
        server.shutdown()
        timestamps['server_shutdown_finished'] = time.perf_counter()
        server.join()

    sys.stdout.flush()
    sys.stderr.flush()
    time_between = lambda key1, key2: \
        abs(timestamps[key2] - timestamps[key1]) if (key1 in timestamps and key2 in timestamps) else float('nan')
    total_examples = batch_size * num_clients * num_batches_per_client

    print('\n' * 3)
    print("Benchmark finished, status:" + ["Success", "Failure"][benchmarking_failed.is_set()])
    print(f"Server parameters: num_experts={num_experts}, num_handlers={num_handlers}, max_batch_size={max_batch_size},"
          f" expert_cls={expert_cls}, hid_dim={hid_dim}, device={device}")
    print(f"Client parameters: num_clients={num_clients}, num_batches_per_client={num_batches_per_client}, "
          f"batch_size={batch_size}, backprop={backprop}")

    print("Results: ")
    print(f"\tServer startup took {time_between('began_launching_server', 'server_ready') :.3f} s. "
          f"({time_between('began_launching_server', 'created_experts') :.3f} s. experts + "
          f"{time_between('created_experts', 'server_ready') :.3f} s. networking)")
    print(f"\tProcessed {total_examples} examples in {time_between('server_ready', 'clients_finished') :.3f}")
    print(f"\tThroughput for {'forward + backward' if backprop else 'forward'} passes: "
          f"{total_examples / time_between('server_ready', 'clients_finished') :.3f} samples / s.")
    print(f"\tBenchmarking took {time_between('started', 'server_shutdown_finished') :.3f} s.")
    if benchmarking_failed.is_set():
        print("Note: benchmark code failed, timing/memory results only indicate time till failure!")
    print_device_info(device)
    print(flush=True)

    assert not benchmarking_failed.is_set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', type=str, default='default', required=False)
    parser.add_argument('--num_batches_per_client', type=int, default=16, required=False)
    args = parser.parse_args()

    if args.preset in ('default', 'ffn_forward_backward'):
        benchmark_throughput()
    elif args.preset == 'ffn_forward':
        benchmark_throughput(backprop=False, num_batches_per_client=args.num_batches_per_client)
    elif args.preset == 'ffn_small_batch':
        benchmark_throughput(backprop=False, num_experts=4, batch_size=32, max_batch_size=8192,
                             num_batches_per_client=args.num_batches_per_client)
    elif args.preset == 'ffn_massive':
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            print("Setting open file limit to soft={}, hard={}".format(max(soft, 2 ** 15), max(hard, 2 ** 15)))
            resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 2 ** 15), max(hard, 2 ** 15)))
        except:
            print("Could not increase open file limit, currently at soft={}, hard={}".format(soft, hard))
        benchmark_throughput(backprop=False, num_clients=512, batch_size=512,
                             max_batch_size=8192, num_batches_per_client=args.num_batches_per_client)
    elif args.preset == 'minimalistic':
        benchmark_throughput(num_experts=1, num_clients=1, num_handlers=1)
    elif args.preset == 'nop':
        benchmark_throughput(expert_cls='nop', backprop=False, num_batches_per_client=args.num_batches_per_client)
    else:
        raise ValueError(f"No such benchmark preset: {args.preset}")
