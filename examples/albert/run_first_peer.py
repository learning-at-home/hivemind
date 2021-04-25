#!/usr/bin/env python

import time
import subprocess
import wandb
from dataclasses import dataclass, asdict
from typing import Optional, List
from whatsmyip.providers import GoogleDnsProvider
from whatsmyip.ip import get_ip

import hivemind
import torch
from torch_optimizer import Lamb
from transformers import AlbertForPreTraining, AlbertConfig, HfArgumentParser
from hivemind.utils.logging import get_logger

from run_trainer import DatasetArguments, CollaborationArguments, \
    get_model, get_optimizer_and_scheduler


logger = get_logger(__name__)


@dataclass
class CoordinatorArguments(CollaborationArguments):
    address: Optional[str] = None  # this machine's network address. Use public IP for global experiments,
    # local address for private runs
    refresh_period: float = 30  # coordinator will fetch keys from DHT once in this many seconds
    wandb_project: Optional[str] = None  # learning curves will be published there
    save_checkpoint_step_interval: int = 5  # coordinator will load and save state from peers once every that many steps
    upload_model_as: Optional[str] = None  # coordinator will upload the checkpoint to that HuggingFace repo
    upload_interval: Optional[float] = None  # coordinator will upload model once in this many seconds
    # Note: You might want to have several initial peers so that if one dies,
    # new workers still can join the collaboration via alive initial peers' addresses.
    # Specify initial_peers argument for that purpose


class CheckpointHandler:
    def __init__(self, coordinator_args: CoordinatorArguments, dataset_args: DatasetArguments, dht: hivemind.DHT):
        self.save_checkpoint_step_interval = coordinator_args.save_checkpoint_step_interval
        self.upload_model_as = coordinator_args.upload_model_as
        self.upload_interval = coordinator_args.upload_interval
        self.previous_step = -1

        config = AlbertConfig.from_pretrained(dataset_args.config_path)
        self.model = AlbertForPreTraining(config)

        collaboration_args_dict = asdict(coordinator_args)
        collaboration_args_dict.pop('address')
        collaboration_args_dict.pop('refresh_period')
        collaboration_args_dict.pop('wandb_project')
        collaboration_args_dict.pop('save_checkpoint_step_interval')
        collaboration_args_dict.pop('upload_model_as')
        collaboration_args_dict.pop('upload_interval')

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        opt = Lamb(
            optimizer_grouped_parameters,
            lr=0.00176, weight_decay=0.01, clamp_value=10000.0, debias=True,
        )

        adjusted_target_batch_size = collaboration_args_dict.pop('target_batch_size') - \
            collaboration_args_dict.pop('batch_size_lead')

        collaboration_args_dict.pop('initial_peers')
        collaboration_args_dict.pop('trainer_uuid')
        collaboration_args_dict.pop('dht_listen_on')
        collaboration_args_dict.pop('statistics_expiration')
        collaboration_args_dict.pop('endpoint')

        self.collaborative_optimizer = hivemind.CollaborativeOptimizer(
            opt=opt, dht=dht, prefix=collaboration_args_dict.pop('experiment_prefix'),
            compression_type=hivemind.utils.CompressionType.Value(collaboration_args_dict.pop('compression')),
            throughput=collaboration_args_dict.pop('bandwidth'),
            target_batch_size=adjusted_target_batch_size, client_mode=collaboration_args_dict.pop('client_mode'),
            verbose=True, start=True, **collaboration_args_dict
        )
        self.previous_timestamp = time.time()

    def is_time_to_save_state(self, cur_step):
        if self.save_checkpoint_step_interval is None:
            return False
        elif cur_step - self.previous_step >= self.save_checkpoint_step_interval:
            return True
        else:
            return False

    def save_state(self, cur_step):
        self.collaborative_optimizer.load_state_from_peers()
        self.previous_step = cur_step

    def is_time_to_upload(self):
        if self.upload_model_as is None:
            return False
        elif time.time() - self.previous_timestamp >= self.upload_interval:
            return True
        else:
            return False

    def upload_checkpoint(self):
        self.model.save_pretrained(self.upload_model_as)
        torch.save(self.collaborative_optimizer.opt.state_dict(), f"{self.upload_model_as}/optimizer_state.pt")
        self.previous_timestamp = time.time()
        try:
            subprocess.run("git add --all".split(), check=True, cwd=self.upload_model_as)
            subprocess.run("git commit -m Updating".split(), check=True, cwd=self.upload_model_as)
            subprocess.run("git push".split(), check=True, cwd=self.upload_model_as)
        except subprocess.CalledProcessError:
            logger.warning("Error while uploading model")


if __name__ == '__main__':
    parser = HfArgumentParser((CoordinatorArguments, DatasetArguments))
    coordinator_args, dataset_args = parser.parse_args_into_dataclasses()

    if coordinator_args.address is None:
        logger.warning("No address specified. Attempting to infer address from DNS.")
        coordinator_args.address = get_ip(GoogleDnsProvider)

    dht = hivemind.DHT(start=True, listen_on=coordinator_args.dht_listen_on,
                       endpoint=f"{coordinator_args.address}:*", initial_peers=coordinator_args.initial_peers)

    logger.info(f"Running DHT root at {coordinator_args.address}:{dht.port}")

    if coordinator_args.wandb_project is not None:
        wandb.init(project=coordinator_args.wandb_project)

    current_step = 0
    checkpoint_handler = CheckpointHandler(coordinator_args, dataset_args, dht)

    while True:
        metrics_dict = dht.get(coordinator_args.experiment_prefix + '_metrics', latest=True)
        if metrics_dict is not None:
            metrics_dict = metrics_dict.value
            metrics = [metrics_dict[peer].value for peer in metrics_dict]
            latest_step = max(metrics)[0]
            if latest_step != current_step:
                current_step = latest_step

                if checkpoint_handler.is_time_to_save_state(current_step):
                    checkpoint_handler.save_state(current_step)
                    if checkpoint_handler.is_time_to_upload():
                        checkpoint_handler.upload_checkpoint()

                alive_peers = 0
                num_batches = 0
                sum_loss = 0
                num_samples = 0
                sum_perf = 0
                sum_mini_steps = 0
                for step, perf, samples, loss, mini_steps in metrics:
                    sum_loss += loss
                    alive_peers += 1
                    sum_perf += perf
                    num_samples += samples
                    sum_mini_steps += mini_steps
                if coordinator_args.wandb_project is not None:
                    wandb.log({
                        "loss": sum_loss / sum_mini_steps,
                        "alive peers": alive_peers,
                        "samples": num_samples,
                        "performance": sum_perf
                    })
                logger.info(f"Step #{current_step}\tloss = {sum_loss / sum_mini_steps:.5f}")
        logger.debug("Peer is still alive...")
        time.sleep(coordinator_args.refresh_period)
