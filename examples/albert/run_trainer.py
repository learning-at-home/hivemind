#!/usr/bin/env python

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import uuid

from datasets import load_from_disk
import transformers
from torch.utils.data import DataLoader
from transformers import (set_seed, HfArgumentParser, TrainingArguments,
                          DataCollatorForLanguageModeling, AlbertTokenizerFast, AlbertConfig, AlbertForPreTraining)
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import is_main_process
from transformers.trainer import Trainer
from torch_optimizer import Lamb
import torch

import hivemind


logger = logging.getLogger(__name__)
LRSchedulerBase = getattr(torch.optim.lr_scheduler, '_LRScheduler', None)


@dataclass
class CollaborationArguments:
    """ define how peers interact with each other while training"""

    # primary parameters
    initial_peers: str  # one or more peers (comma-separated) that will welcome you into the collaboration
    experiment_prefix: str  # a unique "name" of this experiment, used to store metadata on the DHT
    averaging_expiration: float = 5.0  # averaging group will wait for stragglers for at most this many
    averaging_timeout: float = 30.0  # give up on averaging step after this many seconds
    target_batch_size: int = 4096  # perform optimizer step after all peers collectively accumulate this many samples
    client_mode: bool = False  # if True, runs training without incoming connections, in a firewall-compatible mode
    trainer_uuid: str = uuid.uuid4().hex  # this peer's name - used when publishing metadata to DHT, default = random

    # optional tweaks
    target_group_size: int = 64  # maximum group size for all-reduce, default = "everything that fits"
    metadata_expiration: float = 30  # peer's metadata will be removed if not updated in this many seconds
    statistics_expiration: float = 3600  # statistics will be removed if not updated in this many seconds
    dht_listen_on: str = '[::]:*'  # network interface used for incoming DHT communication. Default: all ipv6
    listen_on: str = '[::]:*'  # network interface used for incoming averager communication. Default: all ipv6
    endpoint: Optional[str] = None  # this node's IP for inbound connections, used when running from behind a proxy
    compression: str = 'FLOAT16'

    min_refresh_period: float = 0.25  # wait for at least this many seconds before fetching new collaboration state
    max_refresh_period: float = 30  # wait for at most this many seconds before fetching new collaboration state
    default_refresh_period: float = 3  # attempt to fetch collaboration state every this often until successful
    expected_drift_peers: float = 3  # trainer assumes that this many new peers can join per step
    expected_drift_rate = 0.2  # trainer assumes that this fraction of current size can join per step

    bandwidth: float = 1000.0  # available network bandwidth, in mbps (used for load balancing in all-reduce)
    performance_ema_alpha: float = 0.1  # uses this alpha for moving average estimate of samples per second


@dataclass
class DatasetArguments:
    dataset_path: Optional[str] = field(default='./data/albert_tokenized_wikitext',
                                        metadata={"help": "Path to the tokenized dataset"})
    tokenizer_path: Optional[str] = field(default='./data/tokenizer',
                                          metadata={"help": "Path to the tokenizer"})
    config_path: Optional[str] = field(
        default='https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-config.json',
        metadata={"help": "Path to the model config"})
    cache_dir: Optional[str] = field(default='./data', metadata={"help": "Path to the cache"})


@dataclass
class AlbertTrainingArguments(TrainingArguments):
    dataloader_num_workers: int = 4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    seq_length: int = 512

    max_steps: int = 1_000_000  # Albert is actually ready after 125000 steps
    learning_rate: float = 0.00176
    warmup_steps: int = 5000
    adam_epsilon: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    clamp_value: float = 10000.0

    fp16: bool = True
    fp16_opt_level: str = 'O2'
    do_train: bool = True

    save_total_limit: int = 2
    save_steps: int = 500
    disable_tqdm: bool = False


def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)


def get_model(training_args, config, tokenizer):
    # Find latest checkpoint in output_dir
    output_dir = Path(training_args.output_dir)
    logger.info(f'Checkpoint dir {output_dir}, contents {list(output_dir.glob("checkpoint*"))}')
    latest_checkpoint_dir = max(output_dir.glob('checkpoint*'), default=None, key=os.path.getctime)

    if latest_checkpoint_dir is not None:
        logger.info(f'Loading model from {latest_checkpoint_dir}')
        model = AlbertForPreTraining.from_pretrained(latest_checkpoint_dir)
    else:
        logger.info(f'Training from scratch')
        model = AlbertForPreTraining(config)
        model.resize_token_embeddings(len(tokenizer))

    return model


def get_optimizer_and_scheduler(training_args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    opt = Lamb(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        clamp_value=training_args.clamp_value,
        debias=True,
    )

    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps
    )

    return opt, scheduler


class CollaborativeCallback(transformers.TrainerCallback):
    def __init__(self, dht: hivemind.DHT, collaborative_optimizer: hivemind.CollaborativeOptimizer,
                 trainer_uuid: str, statistics_expiration: float):
        self.dht, self.collaborative_optimizer = dht, collaborative_optimizer
        self.trainer_uuid, self.statistics_expiration = trainer_uuid, statistics_expiration
        super().__init__()

    def on_step_end(self, args: TrainingArguments, state: transformers.TrainerState,
                    control: transformers.TrainerControl, **kwargs):
        control.should_log = True

        if state.log_history:
            tr_loss = state.log_history[-1]['loss']

            my_info = [
                self.collaborative_optimizer.local_step, tr_loss
            ]

            self.dht.store("my_progress", subkey=self.trainer_uuid, value=my_info,
                           expiration_time=hivemind.get_dht_time() + self.statistics_expiration,
                           return_future=True)
        return control


class NoOpScheduler(LRSchedulerBase):
    """ Dummy scheduler for transformers.Trainer. The real scheduler is defined in CollaborativeOptimizer.scheduler """

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def print_lr(self, *args, **kwargs):
        if self.optimizer.scheduler:
            return self.optimizer.scheduler.print_lr(*args, **kwargs)

    def step(self):
        logger.debug("Called NoOpScheduler.step")
        self._last_lr = self.get_lr()

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        logger.debug("Called NoOpScheduler.load_state_dict")


def main():
    parser = HfArgumentParser((AlbertTrainingArguments, DatasetArguments, CollaborationArguments))
    training_args, dataset_args, collaboration_args = parser.parse_args_into_dataclasses()

    collaboration_args.initial_peers = list(map(str.strip, collaboration_args.initial_peers.split(',')))
    logger.info(f"Found {len(collaboration_args.initial_peers)} initial peers: {collaboration_args.initial_peers}")
    if len(collaboration_args.initial_peers) == 0:
        raise ValueError("Please specify at least one network endpoint in initial peers.")

    collaboration_args_dict = asdict(collaboration_args)
    setup_logging(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AlbertConfig.from_pretrained(dataset_args.config_path, cache_dir=dataset_args.cache_dir)
    tokenizer = AlbertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)
    model = get_model(training_args, config, tokenizer)
    model.to(training_args.device)

    tokenized_datasets = load_from_disk(Path(dataset_args.dataset_path))
    # This data collator will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    opt, scheduler = get_optimizer_and_scheduler(training_args, model)

    dht = hivemind.DHT(
        initial_peers=collaboration_args_dict.pop('initial_peers'),
        listen=not collaboration_args_dict['client_mode'], listen_on=collaboration_args_dict.pop('dht_listen_on'),
        endpoint=collaboration_args_dict.pop('endpoint'), start=True)

    total_batch_size_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    trainer_uuid = collaboration_args_dict.pop('trainer_uuid')
    statistics_expiration = collaboration_args_dict.pop('statistics_expiration')

    collaborative_optimizer = hivemind.CollaborativeOptimizer(
        opt=opt, dht=dht, scheduler=scheduler, prefix=collaboration_args_dict.pop('experiment_prefix'),
        compression_type=hivemind.utils.CompressionType.Value(collaboration_args_dict.pop('compression')),
        batch_size_per_step=total_batch_size_per_step, throughput=collaboration_args_dict.pop('bandwidth'),
        verbose=True, start=True, **collaboration_args_dict
    )

    class TrainerWithIndependentShuffling(Trainer):
        def get_train_dataloader(self) -> DataLoader:
            """ Shuffle data independently for each peer to avoid duplicating batches [important for quality] """
            torch.manual_seed(hash(trainer_uuid))
            return super().get_train_dataloader()

    trainer = TrainerWithIndependentShuffling(
        model=model, args=training_args, tokenizer=tokenizer, data_collator=data_collator,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        optimizers=(collaborative_optimizer, NoOpScheduler(collaborative_optimizer)),
        callbacks=[CollaborativeCallback(dht, collaborative_optimizer, trainer_uuid, statistics_expiration)]
    )

    # Training
    if training_args.do_train:
        latest_checkpoint_dir = max(
            Path(training_args.output_dir).glob('checkpoint*'),
            default=None,
            key=os.path.getctime
        )

        trainer.train(model_path=latest_checkpoint_dir)


if __name__ == "__main__":
    main()
