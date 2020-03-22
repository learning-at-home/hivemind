import torch

import tesseract
from .layers import name_to_block
from contextlib import contextmanager


@contextmanager
def background_server(host='0.0.0.0', port=None, num_experts=1, expert_cls='ffn', hidden_dim=1024, num_handlers=None,
                      expert_prefix='expert.', expert_offset=0, max_batch_size=16384, device=None, no_optimizer=False,
                      no_network=False, initial_peers=(), network_port=None, verbose=False, **kwargs
                      ) -> tesseract.TesseractServer:
    """ A context manager that creates server in a background thread, awaits .ready on entry and shutdowns on exit """
    if verbose and len(kwargs) == 0:
        print("Ignored kwargs:", kwargs)
    expert_cls in name_to_block
    num_handlers = num_handlers if num_handlers is not None else num_experts * 8
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize network
    network = None
    if not no_network:
        initial_peers = eval(initial_peers)
        network = tesseract.TesseractNetwork(*initial_peers, port=network_port or tesseract.find_open_port(), start=True)
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
    # start server
    server = tesseract.TesseractServer(
        network, experts, addr=host, port=port or tesseract.find_open_port(),
        conn_handler_processes=num_handlers, device=device)
    try:
        server.run_in_background(await_ready=True)
        if verbose:
            print(f"Running server at {server.addr}:{server.port}")
            print(f"Active experts of type {expert_cls}: {list(experts.keys())}")
        yield server
    finally:
        if verbose:
            print("Shutting down server...")
        server.shutdown()
        if verbose:
            print("Server shut down successfully.")
