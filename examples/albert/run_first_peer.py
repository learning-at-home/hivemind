#!/usr/bin/env python

import time
import argparse
import wandb
from whatsmyip.providers import GoogleDnsProvider
from whatsmyip.ip import get_ip

import hivemind
from hivemind.utils.logging import get_logger


logger = get_logger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--address', type=str, required=False,
                        help="this machine's network address. Use public IP for global experiments, "
                             "local address for private runs.")
    parser.add_argument('--listen_on', type=str, default='0.0.0.0:*', required=False,
                        help="'localhost' for local connections only, '0.0.0.0' for ipv4 '[::]' for ipv6")
    parser.add_argument('--refresh_period', type=float, default=30, required=False,
                        help="coordinator will fetch keys from DHT once in this many seconds")
    parser.add_argument('--experiment_prefix', type=str, required=True,
                        help="a prefix where peers store their metrics for aggregation")
    parser.add_argument('--wandb_project', type=str, required=True,
                        help="Weights & Biases project name to publish learning curves")

    args = parser.parse_args()
    if args.address is None:
        logger.warning("No address specified. Attempting to infer address from DNS.")
        args.address = get_ip(GoogleDnsProvider)

    dht = hivemind.DHT(start=True, listen_on=args.listen_on, endpoint=f"{args.address}:*")
    logger.info(f"Running DHT root at {args.address}:{dht.port}")

    wandb.init(project=args.wandb_project)
    current_step = 0

    while True:
        metrics_dict = dht.get(args.experiment_prefix + '_metrics', latest=True)
        if metrics_dict is not None:
            metrics_dict = metrics_dict.value
            metrics = [metrics_dict[peer].value for peer in metrics_dict]
            latest_step = max(metrics)[0]
            if latest_step != current_step:
                current_step = latest_step
                alive_peers = 0
                num_batches = 0
                sum_loss = 0
                num_samples = 0
                sum_perf = 0
                for step, perf, samples, loss in metrics:
                    sum_loss += loss
                    alive_peers += 1
                    sum_perf += perf
                    num_samples += samples
                wandb.log({
                    "loss": sum_loss / alive_peers,
                    "alive peers": alive_peers,
                    "samples": num_samples,
                    "performance": sum_perf
                })
                logger.info(f"Step #{current_step}\tloss = {sum_loss / alive_peers:.5f}")
        logger.debug("Peer is still alive...")
        time.sleep(args.refresh_period)
