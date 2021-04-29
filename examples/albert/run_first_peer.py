#!/usr/bin/env python

import time
import subprocess
import wandb
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict
from whatsmyip.providers import GoogleDnsProvider
from whatsmyip.ip import get_ip

import hivemind
import torch
from torch_optimizer import Lamb
from transformers import AlbertForPreTraining, AlbertConfig, HfArgumentParser
from hivemind.utils.logging import get_logger
from arguments import BaseTrainingArguments


logger = get_logger(__name__)


@dataclass
class CoordinatorArguments(BaseTrainingArguments):
    address: Optional[str] = field(
        default=None,
        metadata={"help": "This machine's network address. Use public IP for global experiments, "
                          "local address for private runs"}
    )
    refresh_period: float = field(
        default=30,
        metadata={"help": "Coordinator will fetch keys from DHT once in this many seconds"}
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Learning curves will be published there"}
    )
    save_checkpoint_step_interval: int = field(
        default=5,
        metadata={"help": "Coordinator will load and save state from peers once every that many steps"}
    )
    model_config_path: str = field(
        default='https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-config.json',
        metadata={"help": "Path to the model config"}
    )
    repo_path: Optional[str] = field(
        default=None,
        metadata={"help":  "Path to HuggingFace repo in which coordinator will upload the model and optimizer states"}
    )
    upload_interval: Optional[float] = field(
        default=None,
        metadata={"help":  "Coordinator will upload model once in this many seconds"}
    )

    # Note: You might want to have several initial peers so that if one dies,
    # new workers still can join the collaboration via alive initial peers' addresses.
    # Specify initial_peers argument for that purpose


class CheckpointHandler:
    def __init__(self, experiment_prefix: str, coordinator_args_dict: Dict[str, Any], dht: hivemind.DHT):
        self.save_checkpoint_step_interval = coordinator_args_dict.pop('save_checkpoint_step_interval')
        self.repo_path = coordinator_args_dict.pop('repo_path')
        self.upload_interval = coordinator_args_dict.pop('upload_interval')
        self.previous_step = -1

        config = AlbertConfig.from_pretrained(coordinator_args_dict.pop('model_config_path'))
        self.model = AlbertForPreTraining(config)

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

        adjusted_target_batch_size = coordinator_args_dict.pop('target_batch_size') - \
            coordinator_args_dict.pop('batch_size_lead')

        self.collaborative_optimizer = hivemind.CollaborativeOptimizer(
            opt=opt, dht=dht, prefix=experiment_prefix,
            compression_type=hivemind.utils.CompressionType.Value(coordinator_args_dict.pop('compression')),
            throughput=coordinator_args_dict.pop('bandwidth'),
            target_batch_size=adjusted_target_batch_size, client_mode=coordinator_args_dict.pop('client_mode'),
            verbose=True, start=True, **coordinator_args_dict
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
        if self.repo_path is None:
            return False
        elif time.time() - self.previous_timestamp >= self.upload_interval:
            return True
        else:
            return False

    def upload_checkpoint(self, current_loss):
        self.model.save_pretrained(self.repo_path)
        torch.save(self.collaborative_optimizer.opt.state_dict(), f"{self.repo_path}/optimizer_state.pt")
        self.previous_timestamp = time.time()
        try:
            subprocess.run("git add --all", shell=True, check=True, cwd=self.repo_path)
            current_step = self.collaborative_optimizer.collaboration_state.optimizer_step
            subprocess.run(f"git commit -m 'Step {current_step}, loss {round(current_loss, 3)}'", shell=True, check=True, cwd=self.repo_path)
            subprocess.run("git push", shell=True, check=True, cwd=self.repo_path)
        except subprocess.CalledProcessError as e:
            logger.warning("Error while uploading model:", e.output)


if __name__ == '__main__':
    parser = HfArgumentParser(CoordinatorArguments)
    coordinator_args = parser.parse_args_into_dataclasses()[0]
    coordinator_args_dict = asdict(coordinator_args)

    coordinator_address = coordinator_args_dict.pop('address')
    if coordinator_address is None:
        logger.warning("No address specified. Attempting to infer address from DNS.")
        coordinator_address = get_ip(GoogleDnsProvider)

    dht = hivemind.DHT(start=True, listen_on=coordinator_args_dict.pop('dht_listen_on'),
                       endpoint=f"{coordinator_address}:*", initial_peers=coordinator_args_dict.pop('initial_peers'))

    logger.info(f"Running DHT root at {coordinator_address}:{dht.port}")

    wandb_project = coordinator_args_dict.pop('wandb_project')
    if wandb_project is not None:
        wandb.init(project=wandb_project)

    current_step = 0
    experiment_prefix = coordinator_args_dict.pop('experiment_prefix')
    refresh_period = coordinator_args_dict.pop('refresh_period')
    checkpoint_handler = CheckpointHandler(experiment_prefix, coordinator_args_dict, dht)

    while True:
        metrics_dict = dht.get(experiment_prefix + '_metrics', latest=True)
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
                sum_mini_steps = 0
                for step, perf, samples, loss, mini_steps in metrics:
                    sum_loss += loss
                    alive_peers += 1
                    sum_perf += perf
                    num_samples += samples
                    sum_mini_steps += mini_steps
                if wandb_project is not None:
                    wandb.log({
                        "loss": sum_loss / sum_mini_steps,
                        "alive peers": alive_peers,
                        "samples": num_samples,
                        "performance": sum_perf
                    })
                logger.info(f"Step #{current_step}\tloss = {sum_loss / sum_mini_steps:.5f}")
                if checkpoint_handler.is_time_to_save_state(current_step):
                    checkpoint_handler.save_state(current_step)
                    if checkpoint_handler.is_time_to_upload():
                        checkpoint_handler.upload_checkpoint(sum_loss / sum_mini_steps)
        logger.debug("Peer is still alive...")
        time.sleep(refresh_period)
