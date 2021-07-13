#!/usr/bin/env python

import logging
import time
from dataclasses import asdict, dataclass, field
from ipaddress import ip_address
from typing import Optional

import requests
import torch
import wandb
from torch_optimizer import Lamb
from transformers import AlbertForPreTraining, AlbertConfig, HfArgumentParser

import hivemind
import utils
from arguments import BaseTrainingArguments, CollaborativeOptimizerArguments, AveragerArguments

logger = logging.getLogger(__name__)


@dataclass
class TrainingMonitorArguments(BaseTrainingArguments):
    """
    Note: You might want to have several initial peers so that if one dies,
    new workers still can join the collaboration via alive initial peers' addresses.
    Specify initial_peers argument for that purpose
    """

    use_google_dns: bool = field(
        default=False,
        metadata={
            "help": "Use Google DNS to determine the public IP address of this machine (and add it to --announce_maddrs)"
        },
    )
    refresh_period: float = field(default=30, metadata={"help": "Period (in seconds) for fetching the keys from DHT"})
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "Name of Weights & Biases project to report the training progress to"}
    )
    save_checkpoint_step_interval: int = field(
        default=5, metadata={"help": "Frequency (in steps) of fetching and saving state from peers"}
    )
    model_config_path: str = field(
        default="https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-config.json",
        metadata={"help": "Path to the model config"},
    )
    repo_path: Optional[str] = field(
        default=None, metadata={"help": "Path to local repository to store the model and optimizer states"}
    )
    repo_url: Optional[str] = field(
        default=None, metadata={"help": "URL of Hugging Face Hub repository to upload the model and optimizer states"}
    )
    upload_interval: Optional[float] = field(
        default=None, metadata={"help": "Frequency (in seconds) of uploading the model to Hub"}
    )
    store_checkpoins: bool = field(default=False, metadata={"help": "If True, enables CheckpointHandler"})


class CheckpointHandler:
    def __init__(
        self,
        monitor_args: TrainingMonitorArguments,
        collab_optimizer_args: CollaborativeOptimizerArguments,
        averager_args: AveragerArguments,
        dht: hivemind.DHT,
    ):
        self.save_checkpoint_step_interval = monitor_args.save_checkpoint_step_interval
        self.repo_path = monitor_args.repo_path
        self.repo_url = monitor_args.repo_url
        self.upload_interval = monitor_args.upload_interval
        self.previous_step = -1

        config = AlbertConfig.from_pretrained(monitor_args.model_config_path)
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
            lr=0.00176,
            weight_decay=0.01,
            clamp_value=10000.0,
            debias=True,
        )

        adjusted_target_batch_size = collab_optimizer_args.target_batch_size - collab_optimizer_args.batch_size_lead

        self.collaborative_optimizer = hivemind.CollaborativeOptimizer(
            opt=opt,
            dht=dht,
            prefix=experiment_prefix,
            compression_type=hivemind.utils.CompressionType.Value(collab_optimizer_args.compression),
            throughput=collab_optimizer_args.bandwidth,
            target_batch_size=adjusted_target_batch_size,
            client_mode=collab_optimizer_args.client_mode,
            verbose=True,
            start=True,
            **asdict(averager_args),
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
        logger.info("Saving state from peers")
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
        logger.info("Saving optimizer")
        torch.save(self.collaborative_optimizer.opt.state_dict(), f"{self.repo_path}/optimizer_state.pt")
        self.previous_timestamp = time.time()
        logger.info("Started uploading to Model Hub")
        self.model.push_to_hub(
            repo_name=self.repo_path,
            repo_url=self.repo_url,
            commit_message=f"Step {current_step}, loss {current_loss:.3f}",
        )
        logger.info("Finished uploading to Model Hub")


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingMonitorArguments, CollaborativeOptimizerArguments, AveragerArguments))
    monitor_args, collab_optimizer_args, averager_args = parser.parse_args_into_dataclasses()

    if monitor_args.use_google_dns:
        request = requests.get("https://api.ipify.org")
        request.raise_for_status()

        address = request.text
        logger.info(f"Received public IP address of this machine: {address}")
        version = ip_address(address).version
        monitor_args.announce_maddrs += [f"/ip{version}/{address}/tcp/0", f"/ip{version}/{address}/udp/0/quic"]

    experiment_prefix = monitor_args.experiment_prefix
    validators, local_public_key = utils.make_validators(experiment_prefix)

    dht = hivemind.DHT(
        start=True,
        initial_peers=monitor_args.initial_peers,
        record_validators=validators,
        use_ipfs=monitor_args.use_ipfs,
        host_maddrs=monitor_args.host_maddrs,
        announce_maddrs=monitor_args.announce_maddrs,
    )
    utils.log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=monitor_args.use_ipfs)

    if monitor_args.wandb_project is not None:
        wandb.init(project=monitor_args.wandb_project)

    current_step = 0
    if monitor_args.store_checkpoins:
        checkpoint_handler = CheckpointHandler(monitor_args, collab_optimizer_args, averager_args, dht)

    while True:
        metrics_dict = dht.get(experiment_prefix + "_metrics", latest=True)
        if metrics_dict is not None:
            metrics_dict = metrics_dict.value
            metrics = [utils.LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]
            latest_step = max(item.step for item in metrics)

            if latest_step != current_step:
                logger.debug(f"Got metrics from {len(metrics)} peers")

                for i, metrics_for_peer in enumerate(metrics):
                    logger.debug(f"{i} peer {metrics_for_peer}")

                current_step = latest_step
                alive_peers = 0
                sum_loss = 0
                num_samples = 0
                sum_perf = 0
                sum_mini_steps = 0

                for item in metrics:
                    sum_loss += item.loss
                    alive_peers += 1
                    sum_perf += item.samples_per_second
                    num_samples += item.samples_accumulated
                    sum_mini_steps += item.mini_steps
                current_loss = sum_loss / sum_mini_steps
                logger.info(f"Step #{current_step}\tloss = {current_loss:.5f}")

                if monitor_args.wandb_project is not None:
                    wandb.log(
                        {
                            "loss": current_loss,
                            "alive peers": alive_peers,
                            "samples": num_samples,
                            "performance": sum_perf,
                            "step": latest_step,
                        }
                    )

                if monitor_args.store_checkpoins:
                    if checkpoint_handler.is_time_to_save_state(current_step):
                        checkpoint_handler.save_state(current_step)
                        if checkpoint_handler.is_time_to_upload():
                            checkpoint_handler.upload_checkpoint(current_loss)
        logger.debug("Peer is still alive...")
        time.sleep(monitor_args.refresh_period)
