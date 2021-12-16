from functools import partial
from pathlib import Path

import configargparse
import torch

from hivemind.moe.server import Server
from hivemind.moe.server.layers import schedule_name_to_scheduler
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.limits import increase_file_limit
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


def main():
    # fmt:off
    parser = configargparse.ArgParser(default_config_files=["config.yml"])
    parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')
    parser.add_argument('--listen_on', type=str, default='0.0.0.0:*', required=False,
                        help="'localhost' for local connections only, '0.0.0.0' for ipv4 '[::]' for ipv6")
    parser.add_argument('--num_experts', type=int, default=None, required=False, help="The number of experts to serve")
    parser.add_argument('--expert_pattern', type=str, default=None, required=False,
                        help='all expert uids will follow this pattern, e.g. "myexpert.[0:256].[0:1024]" will'
                             ' sample random expert uids between myexpert.0.0 and myexpert.255.1023 . Use either'
                             ' num_experts and this or expert_uids')
    parser.add_argument('--expert_uids', type=str, nargs="*", default=None, required=False,
                        help="specify the exact list of expert uids to create. Use either this or num_experts"
                             " and expert_pattern, not both")
    parser.add_argument('--expert_cls', type=str, default='ffn', required=False,
                        help="expert type from test_utils.layers, e.g. 'ffn', 'transformer', 'det_dropout' or 'nop'")
    parser.add_argument('--hidden_dim', type=int, default=1024, required=False, help='main dimension for expert_cls')

    parser.add_argument('--num_handlers', type=int, default=None, required=False,
                        help='server will use this many processes to handle incoming requests')
    parser.add_argument('--min_batch_size', type=int, default=1,
                        help='Minimum required batch size for all expert operations')
    parser.add_argument('--max_batch_size', type=int, default=16384,
                        help='The total number of examples in the same batch will not exceed this value')
    parser.add_argument('--device', type=str, default=None, required=False,
                        help='all experts will use this device in torch notation; default: cuda if available else cpu')

    parser.add_argument('--optimizer', type=str, default='adam', required=False, help='adam, sgd or none')
    parser.add_argument('--scheduler', type=str, choices=schedule_name_to_scheduler.keys(), default='none',
                        help='LR scheduler type to use')
    parser.add_argument('--num_warmup_steps', type=int, required=False,
                        help='The number of warmup steps for LR schedule')
    parser.add_argument('--num_total_steps', type=int, required=False, help='The total number of steps for LR schedule')
    parser.add_argument('--clip_grad_norm', type=float, required=False, help='Maximum gradient norm used for clipping')

    parser.add_argument('--no_dht', action='store_true', help='if specified, the server will not be attached to a dht')
    parser.add_argument('--initial_peers', type=str, nargs='*', required=False, default=[],
                        help='multiaddrs of one or more active DHT peers (if you want to join an existing DHT)')
    parser.add_argument('--increase_file_limit', action='store_true',
                        help='On *nix, this will increase the max number of processes '
                             'a server can spawn before hitting "Too many open files"; Use at your own risk.')
    parser.add_argument('--compression', type=str, default='NONE', required=False, help='Tensor compression for gRPC')
    parser.add_argument('--checkpoint_dir', type=Path, required=False, help='Directory to store expert checkpoints')
    parser.add_argument('--stats_report_interval', type=int, required=False,
                        help='Interval between two reports of batch processing performance statistics')

    parser.add_argument('--custom_module_path', type=str, required=False,
                        help='Path of a file with custom nn.modules, wrapped into special decorator')

    # fmt:on
    args = vars(parser.parse_args())
    args.pop("config", None)
    optimizer = args.pop("optimizer")
    if optimizer == "adam":
        optim_cls = torch.optim.Adam
    elif optimizer == "sgd":
        optim_cls = partial(torch.optim.SGD, lr=0.01)
    elif optimizer == "none":
        optim_cls = None
    else:
        raise ValueError("optim_cls must be adam, sgd or none")

    if args.pop("increase_file_limit"):
        increase_file_limit()

    compression_type = args.pop("compression")
    compression = getattr(CompressionType, compression_type)

    server = Server.create(**args, optim_cls=optim_cls, start=True, compression=compression)

    try:
        server.join()
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, shutting down")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
