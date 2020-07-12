import argparse
import multiprocessing as mp
from contextlib import contextmanager

import resource
import torch

import hivemind
from test_utils.layers import name_to_block, name_to_input


def make_dummy_server(interface='0.0.0.0', port=None, num_experts=1, expert_cls='ffn', hidden_dim=1024,
                      num_handlers=None, expert_prefix='expert', expert_offset=0, max_batch_size=16384, device=None,
                      no_optimizer=False, no_dht=False, initial_peers=(), dht_port=None, root_port=None, verbose=True,
                      UID_DELIMETER=hivemind.DHT.UID_DELIMETER, start=False, **kwargs) -> hivemind.Server:
    """
    Instantiate a server with several identical experts. See argparse comments below for details
    :param interface: 'localhost' for local connections only, '0.0.0.0' for ipv4 '::' for ipv6
    :param port: main server will listen to this port, default = find open port
    :param num_experts: run this many identical experts
    :param expert_cls: expert type from test_utils.layers, e.g. 'ffn', 'transformer', 'det_dropout' or 'nop';
    :param hidden_dim: main dimension for expert_cls
    :param num_handlers: server will use this many parallel processes to handle incoming requests
    :param expert_prefix: all expert uids will be {expert_prefix}.{index}
    :param expert_offset: expert uid will use indices in range(expert_offset, expert_offset + num_experts)
    :param max_batch_size: total num examples in the same batch will not exceed this value
    :param device: all experts will use this device in torch notation; default: cuda if available else cpu
    :param no_optimizer: if specified, all optimizers use learning rate=0
    :param no_dht: if specified, the server will not be attached to a dht
    :param initial_peers: a list of peers that will introduce this node to the dht,
      e.g. [("1.2.3.4", 1337), ("127.0.0.1", 4321)]'), default = no peers
    :param dht_port:  DHT node will listen on this port, default = find open port
    :param root_port: if this server does not have initial_peers, it will create a virtual dht node on this port.
        You can then use this node as initial peer for subsequent servers.
    :param verbose: whether to print server started / finished / terminated events
    :param start: if True, starts server right away and returns when server is ready for requests
    """
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
            dht_root = hivemind.DHT(
                *initial_peers, listen_on=f"{hivemind.LOCALHOST}:{root_port or hivemind.find_open_port()}", start=True)
            print(f"Initializing DHT with port {dht_root.port}")
            initial_peers = [f"{hivemind.LOCALHOST}:{dht_root.port}"]
        else:
            print("Bootstrapping dht with peers:", initial_peers)
            if root_port is not None:
                print(f"Warning: root_port={root_port} will not be used since we already have peers.")

        dht = hivemind.DHT(
            *initial_peers, listen_on=f"{hivemind.LOCALHOST}:{dht_port or hivemind.find_open_port()}", start=True)
        if verbose:
            print(f"Running dht node on port {dht.port}")

    sample_input = name_to_input[expert_cls](4, hidden_dim)
    if isinstance(sample_input, tuple):
        args_schema = tuple(hivemind.BatchTensorDescriptor.from_tensor(arg) for arg in sample_input)
    else:
        args_schema = (hivemind.BatchTensorDescriptor.from_tensor(sample_input),)

    # initialize experts
    experts = {}
    for i in range(num_experts):
        expert = name_to_block[expert_cls](hidden_dim)
        opt = torch.optim.SGD(expert.parameters(), 0.0 if no_optimizer else 0.05)
        expert_uid = f'{expert_prefix}{UID_DELIMETER}{i + expert_offset}'
        experts[expert_uid] = hivemind.ExpertBackend(name=expert_uid, expert=expert, opt=opt,
                                                     args_schema=args_schema,
                                                     outputs_schema=hivemind.BatchTensorDescriptor(hidden_dim),
                                                     max_batch_size=max_batch_size,
                                                     )
    # actually start server
    server = hivemind.Server(
        dht, experts, addr=interface, port=port or hivemind.find_open_port(),
        conn_handler_processes=num_handlers, device=device)

    if start:
        server.run_in_background(await_ready=True)
        if verbose:
            print(f"Server started at {server.addr}:{server.port}")
            print(f"Got {num_experts} active experts of type {expert_cls}: {list(experts.keys())}")
    return server


@contextmanager
def background_server(*args, shutdown_timeout=5, verbose=True, **kwargs):
    """ A context manager that creates server in a background thread, awaits .ready on entry and shutdowns on exit """
    pipe, runners_pipe = mp.Pipe(duplex=True)
    runner = mp.get_context("spawn").Process(
        target=_server_runner, args=(runners_pipe, *args), kwargs=dict(verbose=verbose, **kwargs))

    try:
        runner.start()
        yield pipe.recv()  # once the server is ready, runner will send us a tuple(hostname, port, dht port)
        pipe.send('SHUTDOWN')  # on exit from context, send shutdown signal
    finally:
        try:
            runner.join(timeout=shutdown_timeout)
        finally:
            if verbose:
                print("Server failed to shutdown gracefully, terminating it the hard way...")
            runner.terminate()
            if verbose:
                print("Server terminated.")


def _server_runner(pipe, *args, verbose, **kwargs):
    server = make_dummy_server(*args, verbose=verbose, start=True, **kwargs)
    try:
        dht_port = server.dht.port if server.dht is not None else None
        pipe.send((server.addr, server.port, dht_port))
        pipe.recv()  # wait for shutdown signal
    finally:
        if verbose:
            print("Shutting down server...")
        server.shutdown()
        if verbose:
            print("Server shut down successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interface', type=str, default='0.0.0.0', required=False,
                        help="'localhost' for local connections only, '0.0.0.0' for ipv4 '::' for ipv6")
    parser.add_argument('--port', type=int, default=None, required=False, help="server will listen to this port")
    parser.add_argument('--num_experts', type=int, default=1, required=False, help="run this many identical experts")
    parser.add_argument('--expert_cls', type=str, default='ffn', required=False,
                        help="expert type from test_utils.layers, e.g. 'ffn', 'transformer', 'det_dropout' or 'nop'.")
    parser.add_argument('--hidden_dim', type=int, default=1024, required=False, help='main dimension for expert_cls')
    parser.add_argument('--num_handlers', type=int, default=None, required=False,
                        help='server will use this many processes to handle incoming requests')
    parser.add_argument('--expert_prefix', type=str, default='expert', required=False,
                        help='all expert uids will be {expert_prefix}.{index}')
    parser.add_argument('--expert_offset', type=int, default=0, required=False,
                        help='expert uid will use indices in range(expert_offset, expert_offset + num_experts)')
    parser.add_argument('--max_batch_size', type=int, default=16384, required=False,
                        help='total num examples in the same batch will not exceed this value')
    parser.add_argument('--device', type=str, default=None, required=False,
                        help='all experts will use this device in torch notation; default: cuda if available else cpu')
    parser.add_argument('--no_optimizer', action='store_true', help='if specified, all optimizers use learning rate=0')
    parser.add_argument('--no_dht', action='store_true', help='if specified, the server will not be attached to a dht')
    parser.add_argument('--initial_peers', type=str, default="[]", required=False, help='a list of peers that will'
                                                                                        ' introduce this node to the dht, e.g. [("1.2.3.4", 1337), ("127.0.0.1", 4321)]')
    parser.add_argument('--dht_port', type=int, default=None, required=False, help='DHT node will listen on this port')
    parser.add_argument('--root_port', type=int, default=None, required=False, help='If this server does not have peers'
                                                                                    ', it will create a virtual dht node on this port. You can then use this node as initial peer.')
    parser.add_argument('--increase_file_limit', action='store_true', help='On *nix, this will increase the max number'
                                                                           ' of processes a server can spawn before hitting "Too many open files"; Use at your own risk.')

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
