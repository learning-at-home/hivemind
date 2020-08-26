from typing import Optional
import configargparse
import resource
from hivemind.server import Server

if __name__ == '__main__':
    # fmt:off
    parser = configargparse.ArgParser(default_config_files=["config.yml"])
    parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')
    parser.add_argument('--listen_on', type=str, default='0.0.0.0:*', required=False,
                        help="'localhost' for local connections only, '0.0.0.0' for ipv4 '::' for ipv6")
    parser.add_argument('--num_experts', type=int, default=None, required=False, help="run this many experts")
    parser.add_argument('--expert_pattern', type=str, default=None, required=False, help='all expert uids will follow'
                        ' this pattern, e.g. "myexpert.[0:256].[0:1024]" will sample random expert uids'
                        ' between myexpert.0.0 and myexpert.255.1023 . Use either num_experts and this or expert_uids')
    parser.add_argument('--expert_uids', type=str, nargs="*", default=None, required=False,
                        help="specify the exact list of expert uids to create. Use either this or num_experts"
                             " and expert_pattern, not both")
    parser.add_argument('--expert_cls', type=str, default='ffn', required=False,
                        help="expert type from test_utils.layers, e.g. 'ffn', 'transformer', 'det_dropout' or 'nop'.")
    parser.add_argument('--hidden_dim', type=int, default=1024, required=False, help='main dimension for expert_cls')
    parser.add_argument('--num_handlers', type=int, default=None, required=False,
                        help='server will use this many processes to handle incoming requests')
    parser.add_argument('--max_batch_size', type=int, default=16384, required=False,
                        help='total num examples in the same batch will not exceed this value')
    parser.add_argument('--device', type=str, default=None, required=False,
                        help='all experts will use this device in torch notation; default: cuda if available else cpu')
    parser.add_argument('--no_optimizer', action='store_true', help='if specified, all optimizers use learning rate=0')
    parser.add_argument('--no_dht', action='store_true', help='if specified, the server will not be attached to a dht')
    parser.add_argument('--initial_peers', type=str, nargs='*', required=False, default=[], help='one or more peers'
                        ' that can welcome you to the dht, e.g. 1.2.3.4:1337 192.132.231.4:4321')
    parser.add_argument('--dht_port', type=int, default=None, required=False, help='DHT node will listen on this port')
    parser.add_argument('--increase_file_limit', action='store_true', help='On *nix, this will increase the max number'
                        ' of processes a server can spawn before hitting "Too many open files"; Use at your own risk.')
    # fmt:on
    args = vars(parser.parse_args())
    args.pop('config', None)

    if args.pop('increase_file_limit'):
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            print("Setting open file limit to soft={}, hard={}".format(max(soft, 2 ** 15), max(hard, 2 ** 15)))
            resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 2 ** 15), max(hard, 2 ** 15)))
        except:
            print("Could not increase open file limit, currently at soft={}, hard={}".format(soft, hard))

    try:
        server = Server.create(**args, start=True, verbose=True)
        server.join()
    finally:
        server.shutdown()