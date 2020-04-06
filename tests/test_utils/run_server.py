import resource
from contextlib import contextmanager
import multiprocessing as mp
import argparse

import torch
import hivemind
from .layers import name_to_block, name_to_input


def make_dummy_server(host='0.0.0.0', port=None, num_experts=1, expert_cls='ffn', hidden_dim=1024, num_handlers=None,
                      expert_prefix='expert', expert_offset=0, max_batch_size=16384, device=None, no_optimizer=False,
                      no_dht=False, initial_peers=(), dht_port=None, root_port=None, verbose=True, start=False,
                      UID_DELIMETER=hivemind.DHTNode.UID_DELIMETER, **kwargs) -> hivemind.Server:
    """ A context manager that creates server in a background thread, awaits .ready on entry and shutdowns on exit """
    if verbose and len(kwargs) != 0:
        print("Ignored kwargs:", kwargs)
    assert expert_cls in name_to_block
    num_handlers = num_handlers if num_handlers is not None else num_experts * 8
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize dht
    dht = None
    if not no_dht:
        if not len(initial_peers):
            print("No initial peers provided. Starting additional dht as an initial peer.")
            dht_root = hivemind.DHTNode(
                *initial_peers, port=root_port or hivemind.find_open_port(), start=True)
            print(f"Initializing DHT with port {dht_root.port}")
            initial_peers = (('localhost', dht_root.port),)
        else:
            print("Bootstrapping dht with peers:", initial_peers)
            if root_port is not None:
                print(f"Warning: root_port={root_port} will not be used since we already have peers.")

        dht = hivemind.DHTNode(
            *initial_peers, port=dht_port or hivemind.find_open_port(), start=True)
        if verbose:
            print(f"Running dht node on port {dht.port}")

    sample_input = name_to_input[expert_cls](4, hidden_dim)
    if isinstance(sample_input, tuple):
        args_schema = tuple(hivemind.BatchTensorProto.from_tensor(arg) for arg in sample_input)
    else:
        args_schema = (hivemind.BatchTensorProto.from_tensor(sample_input),)

    # initialize experts
    experts = {}
    for i in range(num_experts):
        expert = name_to_block[expert_cls](hidden_dim)
        opt = torch.optim.SGD(expert.parameters(), 0.0) if no_optimizer else torch.optim.Adam(expert.parameters())
        expert_uid = f'{expert_prefix}{UID_DELIMETER}{i + expert_offset}'
        experts[expert_uid] = hivemind.ExpertBackend(name=expert_uid, expert=expert, opt=opt,
                                                     args_schema=args_schema,
                                                     outputs_schema=hivemind.BatchTensorProto(hidden_dim),
                                                     max_batch_size=max_batch_size,
                                                     )
    # actually start server
    server = hivemind.Server(
        dht, experts, addr=host, port=port or hivemind.find_open_port(),
        conn_handler_processes=num_handlers, device=device)

    if start:
        server.run_in_background(await_ready=True)
        if verbose:
            print(f"Server started at {server.addr}:{server.port}")
            print(f"Got {num_experts} active experts of type {expert_cls}: {list(experts.keys())}")
    return server


@contextmanager
def background_server(*args, verbose=True, **kwargs):
    """ Runs server in a background process and returns a reference to it. """
    recv_addr, send_addr = mp.Pipe(duplex=True)
    trigger_shutdown = mp.Event()

    def server_runner():
        try:
            server = make_dummy_server(*args, verbose=verbose, start=True, **kwargs)
            dht_port = server.dht.port if server.dht is not None else None
            send_addr.send((server.addr, server.port, dht_port))
            trigger_shutdown.wait()
        finally:
            if verbose:
                print("Shutting down server...")
            trigger_shutdown.set()  # if server failed internally, set the shutdown trigger anyway
            server.shutdown()
            if verbose:
                print("Server shut down successfully.")

    try:
        runner = mp.Process(target=server_runner)
        runner.start()
        yield recv_addr.recv()  # yield tuple(hostname, port)

    finally:
        trigger_shutdown.set()
        runner.terminate()
        runner.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0', required=False)
    parser.add_argument('--port', type=int, default=None, required=False)
    parser.add_argument('--num_experts', type=int, default=1, required=False)
    parser.add_argument('--expert_cls', type=str, default='ffn', required=False)
    parser.add_argument('--hidden_dim', type=int, default=1024, required=False)
    parser.add_argument('--num_handlers', type=int, default=None, required=False)
    parser.add_argument('--expert_prefix', type=str, default='expert', required=False)
    parser.add_argument('--expert_offset', type=int, default=0, required=False)
    parser.add_argument('--max_batch_size', type=int, default=16384, required=False)
    parser.add_argument('--device', type=str, default=None, required=False)
    parser.add_argument('--no_optimizer', action='store_true')
    parser.add_argument('--no_dht', action='store_true')
    parser.add_argument('--initial_peers', type=str, default="[]", required=False)
    parser.add_argument('--dht_port', type=int, default=None, required=False)
    parser.add_argument('--root_port', type=int, default=None, required=False)

    parser.add_argument('--increase_file_limit', action='store_true')

    args = vars(parser.parse_args())

    if args.pop('increase_file_limit'):
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            print("Setting open file limit to soft={}, hard={}".format(max(soft, 2 ** 15), max(hard, 2 ** 15)))
            resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 2 ** 15), max(hard, 2 ** 15)))
        except:
            print("Could not increase open file limit, currently at soft={}, hard={}".format(soft, hard))

    args['initial_peers'] = eval(args['initial_peers'])

    try:
        server = make_dummy_server(**args, start=True, verbose=True)
        server.join()
    finally:
        server.shutdown()
