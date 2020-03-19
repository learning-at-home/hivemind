import argparse
import multiprocessing as mp
import random
import resource
import os
import sys
import time

import torch
sys.path.append(os.path.dirname(__file__) + '/../tests')
from test_utils import layers, find_open_port
import tesseract


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_cls', type=str, default='ffn', required=False)
    parser.add_argument('--num_experts', type=int, default=1, required=False)
    parser.add_argument('--num_handlers', type=int, default=None, required=False)
    parser.add_argument('--hidden_dim', type=int, default=1024, required=False)
    parser.add_argument('--max_batch_size', type=int, default=16384, required=False)
    parser.add_argument('--expert_prefix', type=str, default='expert', required=False)
    parser.add_argument('--expert_offset', type=int, default=0, required=False)
    parser.add_argument('--device', type=str, default=None, required=False)
    parser.add_argument('--port', type=int, default=None, required=False)
    parser.add_argument('--host', type=str, default='0.0.0.0', required=False)
    parser.add_argument('--no_network', action='store_true')
    parser.add_argument('--initial_peers', type=str, default="[]", required=False)
    parser.add_argument('--network_port', type=int, default=None, required=False)
    parser.add_argument('--increase_file_limit', action='store_true')

    args = parser.parse_args()
    if args.increase_file_limit:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            print("Setting open file limit to soft={}, hard={}".format(max(soft, 2 ** 15), max(hard, 2 ** 15)))
            resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 2 ** 15), max(hard, 2 ** 15)))
        except:
            print("Could not increase open file limit, currently at soft={}, hard={}".format(soft, hard))

    assert args.expert_cls in layers.name_to_block
    args.num_handlers = args.num_handlers or args.num_experts * 8

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize network
    network = None
    if not args.no_network:
        initial_peers = eval(args.initial_peers)
        print("Parsed initial peers:", initial_peers)

        network = tesseract.TesseractNetwork(*initial_peers, port=args.network_port or find_open_port(), start=True)
        print(f"Running network node on port {network.port}")

    # initialize experts
    experts = {}
    for i in range(args.num_experts):
        expert = torch.jit.script(layers.name_to_block[args.expert_cls](args.hidden_dim))
        expert_uid = f'{args.expert_prefix}.{i + args.expert_offset}'
        experts[expert_uid] = tesseract.ExpertBackend(name=expert_uid,
                                                      expert=expert, opt=torch.optim.Adam(expert.parameters()),
                                                      args_schema=(tesseract.BatchTensorProto(args.hidden_dim),),
                                                      outputs_schema=tesseract.BatchTensorProto(args.hidden_dim),
                                                      max_batch_size=args.max_batch_size,
                                                      )
    # start server
    server = tesseract.TesseractServer(
        network, experts, addr=args.host, port=args.port or find_open_port(),
        conn_handler_processes=args.num_handlers, device=device)
    print(f"Running server at {server.addr}:{server.port}")
    print(f"Active expert uids: {experts}")
    try:
        server.run()
    finally:
        server.shutdown()
