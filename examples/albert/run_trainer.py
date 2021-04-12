#!/usr/bin/env python

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import uuid

from datasets import load_from_disk
import transformers
from transformers import (set_seed, HfArgumentParser, TrainingArguments,
                          DataCollatorForLanguageModeling, AlbertTokenizerFast, AlbertConfig, AlbertForPreTraining)
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import is_main_process
from transformers.trainer import Trainer
from torch_optimizer import Lamb


import hivemind
from hivemind.client import CollaborationArguments
from hivemind.client.optim import CollaborativeOptimizer
from hivemind import DHT


logger = logging.getLogger(__name__)


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
    dataloader_num_workers: int = 8
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    seq_length: int = 512

    max_steps: int = 1_000_000  # Albert is actually ready after 125000 steps
    learning_rate: float = 0.00176
    warmup_steps: int = 5000
    adam_epsilon: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    clamp_value: float = 10.0

    fp16: bool = True
    fp16_opt_level: str = 'O2'
    do_train: bool = True

    save_total_limit: int = 2
    save_steps: int = 500


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
        debias=True, clamp_value=training_args.clamp_value,
    )

    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps
    )

    return opt, scheduler


class CustomLoggingCallback(transformers.TrainerCallback):
    def __init__(self, dht: DHT, collaborative_optimizer: CollaborativeOptimizer,
                 collaboration_args: CollaborationArguments, trainer_uuid):
        self.dht = dht
        self.collaborative_optimizer = collaborative_optimizer
        self.statistics_expiration = collaboration_args.statistics_expiration
        self.trainer_uuid = trainer_uuid

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


def main():
    parser = HfArgumentParser((AlbertTrainingArguments, DatasetArguments, CollaborationArguments))
    training_args, dataset_args, collaboration_args = parser.parse_args_into_dataclasses()

    setup_logging(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AlbertConfig.from_pretrained(dataset_args.config_path, cache_dir=dataset_args.cache_dir)
    tokenizer = AlbertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)
    model = get_model(training_args, config, tokenizer)
    model.to(training_args.device)

    tokenized_dataset_path = Path(dataset_args.dataset_path)
    tokenized_datasets = load_from_disk(tokenized_dataset_path)
    # This data collator will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    opt, scheduler = get_optimizer_and_scheduler(training_args, model)

    print('Starting DHT')
    dht = DHT(initial_peers=[collaboration_args.initial_peers], start=True)
    print('DHT OK')

    collaborative_optimizer = CollaborativeOptimizer(
        opt=opt,
        scheduler=scheduler,
        dht=dht,
        prefix=collaboration_args.dht_key_for_averaging,
        target_group_size=collaboration_args.target_group_size,
        target_batch_size=collaboration_args.target_batch_size,
        batch_size_per_step=training_args.per_device_train_batch_size,
        start=True,
        client_mode=collaboration_args.client_mode,
        verbose=True
    )

    print('ColOpt is created')

    def noop(*args, **kwargs):
        if noop.visited < 5:
            print('Zero grad is successfully overwritten')
            noop.visited += 1

    noop.visited = 0

    model.zero_grad = noop

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(collaborative_optimizer, scheduler),
        callbacks=[CustomLoggingCallback(dht, collaborative_optimizer, collaboration_args, uuid.uuid4().hex)]
    )

    print('Trainer is created')

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
