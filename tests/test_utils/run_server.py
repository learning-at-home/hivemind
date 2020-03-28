from contextlib import contextmanager
import multiprocessing as mp

import torch
import tesseract
from .layers import name_to_block


def make_dummy_server(host='0.0.0.0', port=None, num_experts=1, expert_cls='ffn', hidden_dim=1024, num_handlers=None,
                      expert_prefix='expert.', expert_offset=0, max_batch_size=16384, device='cpu', no_optimizer=False,
                      no_network=False, initial_peers=(), network_port=None, verbose=True, start=True, **kwargs
                      ) -> tesseract.TesseractServer:
    """ A context manager that creates server in a background thread, awaits .ready on entry and shutdowns on exit """
    if verbose and len(kwargs) != 0:
        print("Ignored kwargs:", kwargs)
    assert expert_cls in name_to_block
    num_handlers = num_handlers if num_handlers is not None else num_experts * 8
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize network
    network = None
    if not no_network:
        network = tesseract.TesseractNetwork(
            *initial_peers, port=network_port or tesseract.find_open_port(), start=True)
        if verbose:
            print("Parsed initial peers:", initial_peers)
            print(f"Running network node on port {network.port}")

    # initialize experts
    experts = {}
    for i in range(num_experts):
        expert = torch.jit.script(name_to_block[expert_cls](hidden_dim))
        opt = torch.optim.SGD(expert.parameters(), 0.0) if no_optimizer else torch.optim.Adam(expert.parameters())
        expert_uid = f'{expert_prefix}{i + expert_offset}'
        experts[expert_uid] = tesseract.ExpertBackend(name=expert_uid, expert=expert, opt=opt,
                                                      args_schema=(tesseract.BatchTensorProto(hidden_dim),),
                                                      outputs_schema=tesseract.BatchTensorProto(hidden_dim),
                                                      max_batch_size=max_batch_size,
                                                      )
    # actually start server
    server = tesseract.TesseractServer(
        network, experts, addr=host, port=port or tesseract.find_open_port(),
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
            send_addr.send((server.addr, server.port))
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
        runner.join()


if __name__ == '__main__':
    with background_server() as (host, port):
        mp.Event().wait()  # aka fall asleep forever
