#!/usr/bin/env python

import time
import argparse
import hivemind
import wandb
import torch
from transformers import (AlbertTokenizerFast, AlbertConfig, AlbertForPreTraining)

from hivemind import get_logger


def get_public_ip():
    from whatsmyip.ip import get_ip
    from whatsmyip.providers import GoogleDnsProvider
    return get_ip(GoogleDnsProvider)


logger = get_logger(__name__)


class CheckpointUploader:
    def __init__(self, upload_checkpoint_interval):
        import multiprocessing as mp

        self.upload_checkpoint_interval = upload_checkpoint_interval
        if self.upload_checkpoint_interval:
            self.albert_config_file = 'https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-config.json'
            self.cache_dir = './data'
            self.tokenizer_path = './data/tokenizer'

            self.pipe = mp.Pipe(duplex=False)  # a control pipe used to communicate with a background process
            # TODO: somehow connect pipe with another peers

        self.previous_timestamp = time.time()

    def is_time_to_upload(self):
        if self.upload_checkpoint_interval is None:
            return False
        elif time.time() - self.previous_timestamp >= self.upload_checkpoint_interval:
            return True
        else:
            return False

    def upload_checkpoint(self):
        config = AlbertConfig.from_pretrained(self.albert_config_file, cache_dir=self.cache_dir)
        tokenizer = AlbertTokenizerFast.from_pretrained(self.tokenizer_path, cache_dir=self.cache_dir)
        model = AlbertForPreTraining(config)
        model.resize_token_embeddings(len(tokenizer))

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            },
        ]

        parameters = [param for param_group in optimizer_grouped_parameters for param in param_group['params']]
        num_local_tensors = len(parameters)

        future, _future = hivemind.utils.MPFuture.make_pair()
        self.pipe.send(('_TRIGGER_GET_CURRENT_STATE', _future))
        loaded_state = future.result()

        if loaded_state is None:
            return

        metadata, flat_tensors = loaded_state
        loaded_parameters = flat_tensors[:num_local_tensors]
        loaded_opt_tensors = flat_tensors[num_local_tensors:]

        with torch.no_grad():
            for local_param, loaded_param in zip(parameters, loaded_parameters):
                local_param[...] = loaded_param

            optimizer_state = {
                'optimizer_metadata': metadata['optimizer_metadata'],
                'optimizer_tensors': loaded_opt_tensors
            }

        # TODO: push model and optimizer_state to HuggingFace Model Hub

        self.previous_timestamp = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--address', type=str, required=False, default=None,
                        help="this machine's network address. Use public IP for global experiments, "
                             "local address for private runs.")
    parser.add_argument('--listen_on', type=str, default='0.0.0.0:*', required=False,
                        help="'localhost' for local connections only, '0.0.0.0' for ipv4 '[::]' for ipv6")
    parser.add_argument('--refresh_period', type=float, default=30, required=False,
                        help="coordinator will fetch keys from DHT once in this many seconds")
    parser.add_argument('--experiment_prefix', type=str, required=True,
                        help="a prefix where peers store their metrics for aggregation")
    parser.add_argument('--wandb_project', type=str, required=False,
                        help="learning curves will be published there. Specify this for only one coordinator")
    parser.add_argument('--initial_peers', type=str, required=False,
                        help="you might want to have several initial peers so that if one dies, new workers still "
                             "can join the collaboration via alive initial peers' addresses")
    parser.add_argument('--upload_checkpoint_interval', type=float, required=False,
                        help="coordinator will upload current model checkpoint to  HuggingFace Model Hub "
                             "in this many seconds")

    args = parser.parse_args()
    if args.address is None:
        logger.warning("No address specified. Attempting to infer address from DNS.")
        try:
            args.address = get_public_ip()
        except ImportError as e:
            logger.error("Could not infer network address, please specify --address manually.")
            exit(-1)

    if args.initial_peers is None:
        dht = hivemind.DHT(start=True, listen_on=args.listen_on, endpoint=f"{args.address}:*")
    else:
        args.initial_peers = list(map(str.strip, args.initial_peers.split(',')))
        dht = hivemind.DHT(start=True, listen_on=args.listen_on, endpoint=f"{args.address}:*",
                           initial_peers=args.initial_peers)

    logger.info(f"Running DHT root at {args.address}:{dht.port}")

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project)

    current_step = 0

    while True:
        metrics_dict = dht.get(args.dht_key_for_averaging + '_metrics', latest=True)
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
                    if step == latest_step:
                        sum_loss += loss
                        alive_peers += 1
                        sum_perf += perf
                        num_samples += samples
                if args.wandb_project is not None:
                    wandb.log({
                        "loss": sum_loss / alive_peers,
                        "alive peers": alive_peers,
                        "samples": num_samples,
                        "performance": sum_perf
                    })
                logger.info(f"{sum_loss / alive_peers:.5f}")
            logger.debug("Peer is still alive...")
        time.sleep(args.refresh_period)
