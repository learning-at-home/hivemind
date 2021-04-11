#!/usr/bin/env python

import time
import uuid
import argparse
import hivemind


def get_public_ip():
    from whatsmyip.ip import get_ip
    from whatsmyip.providers import GoogleDnsProvider
    return get_ip(GoogleDnsProvider)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--address', type=str, required=False, default=None,
                        help="this machine's network address. Use public IP for global experiments, "
                             "local address for private runs.")
    parser.add_argument('--listen_on', type=str, default='0.0.0.0:*', required=False,
                        help="'localhost' for local connections only, '0.0.0.0' for ipv4 '[::]' for ipv6")
    parser.add_argument('--refresh_period', type=float, default=30, required=False,
                        help="coordinator will fetch random keys every :this many: seconds to detect inactive peers")

    args = parser.parse_args()
    if args.address is None:
        print("No address specified. Attempting to infer address from DNS.")
        try:
            args.address = get_public_ip()
        except ImportError as e:
            print("Could not infer network address, please specify --address manually.")
            exit(-1)

    dht = hivemind.DHT(start=True, listen_on=args.listen_on, endpoint=f"{args.address}:*")
    print(f"Running DHT root at {args.address}:{dht.port}", flush=True)
    while True:
        dht.get(uuid.uuid4().bytes, latest=True)
        time.sleep(args.refresh_period)
