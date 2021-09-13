#!/usr/bin/env python

import os
import pickle
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch_optimizer import Lamb
from transformers import DataCollatorForLanguageModeling, HfArgumentParser, TrainingArguments, set_seed
from transformers.models.albert import AlbertConfig, AlbertForPreTraining, AlbertTokenizerFast
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer import Trainer
from transformers.trainer_utils import is_main_process

import hivemind
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

import utils
from arguments import AlbertTrainingArguments, AveragerArguments, CollaborationArguments, DatasetArguments

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

LRSchedulerBase = getattr(torch.optim.lr_scheduler, "_LRScheduler", None)


def setup_transformers_logging(process_rank: int):
    if is_main_process(process_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.disable_default_handler()
        transformers.utils.logging.enable_propagation()


def get_model(training_args, config, tokenizer):
    # Find latest checkpoint in output_dir
    output_dir = Path(training_args.output_dir)
    logger.info(f'Checkpoint dir {output_dir}, contents {list(output_dir.glob("checkpoint*"))}')
    latest_checkpoint_dir = max(output_dir.glob("checkpoint*"), default=None, key=os.path.getctime)

    if latest_checkpoint_dir is not None:
        logger.info(f"Loading model from {latest_checkpoint_dir}")
        model = AlbertForPreTraining.from_pretrained(latest_checkpoint_dir)
    else:
        logger.info(f"Training from scratch")
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
        opt, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps
    )

    return opt, scheduler


class CollaborativeCallback(transformers.TrainerCallback):
    """
    This callback monitors and reports collaborative training progress.
    In case of a catastrophic failure, it can also revert training to a backup.
    """

    def __init__(
        self,
        dht: hivemind.DHT,
        optimizer: hivemind.CollaborativeOptimizer,
        model: torch.nn.Module,
        local_public_key: bytes,
        statistics_expiration: float,
        backup_every_steps: int,
    ):
        super().__init__()
        self.model = model
        self.dht, self.collaborative_optimizer = dht, optimizer
        self.local_public_key = local_public_key
        self.statistics_expiration = statistics_expiration
        self.last_reported_collaboration_step = -1
        self.samples = 0
        self.steps = 0
        self.loss = 0
        self.total_samples_processed = 0
        self.backup_every_steps = backup_every_steps
        self.latest_backup = self.backup_state()

    def on_train_begin(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        logger.info("Loading state from peers")
        self.collaborative_optimizer.load_state_from_peers()

    def on_step_end(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        control.should_log = True
        if not self.params_are_finite():
            self.restore_from_backup(self.latest_backup)
            return control

        if state.log_history:
            self.loss += state.log_history[-1]["loss"]
            self.steps += 1
            if self.collaborative_optimizer.local_step != self.last_reported_collaboration_step:
                self.last_reported_collaboration_step = self.collaborative_optimizer.local_step
                self.total_samples_processed += self.samples
                samples_per_second = self.collaborative_optimizer.performance_ema.samples_per_second
                statistics = utils.LocalMetrics(
                    step=self.collaborative_optimizer.local_step,
                    samples_per_second=samples_per_second,
                    samples_accumulated=self.samples,
                    loss=self.loss,
                    mini_steps=self.steps,
                )
                logger.info(f"Step #{self.collaborative_optimizer.local_step}")
                logger.info(f"Your current contribution: {self.total_samples_processed} samples")
                logger.info(f"Performance: {samples_per_second} samples per second.")
                if self.steps:
                    logger.info(f"Local loss: {self.loss / self.steps}")
                if self.collaborative_optimizer.local_step % self.backup_every_steps == 0:
                    self.latest_backup = self.backup_state()

                self.loss = 0
                self.steps = 0
                if self.collaborative_optimizer.is_synchronized:
                    self.dht.store(
                        key=self.collaborative_optimizer.prefix + "_metrics",
                        subkey=self.local_public_key,
                        value=statistics.dict(),
                        expiration_time=hivemind.get_dht_time() + self.statistics_expiration,
                        return_future=True,
                    )

        self.samples = self.collaborative_optimizer.local_samples_accumulated

        return control

    @torch.no_grad()
    def params_are_finite(self):
        for param in self.model.parameters():
            if not torch.all(torch.isfinite(param)):
                return False
        return True

    @torch.no_grad()
    def backup_state(self) -> bytes:
        return pickle.dumps(
            {"model": self.model.state_dict(), "optimizer": self.collaborative_optimizer.opt.state_dict()}
        )

    @torch.no_grad()
    def restore_from_backup(self, backup: bytes):
        state = pickle.loads(backup)
        self.model.load_state_dict(state["model"])
        self.collaborative_optimizer.opt.load_state_dict(state["optimizer"])


class NoOpScheduler(LRSchedulerBase):
    """Dummy scheduler for transformers.Trainer. The real scheduler is defined in CollaborativeOptimizer.scheduler"""

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, *args, **kwargs):
        if self.optimizer.scheduler:
            return self.optimizer.scheduler.print_lr(*args, **kwargs)

    def step(self):
        self._last_lr = self.get_lr()

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        logger.debug("Called NoOpScheduler.load_state_dict")


def main():
    parser = HfArgumentParser((AlbertTrainingArguments, DatasetArguments, CollaborationArguments, AveragerArguments))
    training_args, dataset_args, collaboration_args, averager_args = parser.parse_args_into_dataclasses()

    logger.info(f"Found {len(collaboration_args.initial_peers)} initial peers: {collaboration_args.initial_peers}")
    if len(collaboration_args.initial_peers) == 0:
        raise ValueError("Please specify at least one network endpoint in initial peers.")

    setup_transformers_logging(training_args.local_rank)
    logger.info(f"Training/evaluation parameters:\n{training_args}")

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

    validators, local_public_key = utils.make_validators(collaboration_args.experiment_prefix)

    dht = hivemind.DHT(
        start=True,
        initial_peers=collaboration_args.initial_peers,
        client_mode=collaboration_args.client_mode,
        record_validators=validators,
        use_ipfs=collaboration_args.use_ipfs,
        host_maddrs=collaboration_args.host_maddrs,
        announce_maddrs=collaboration_args.announce_maddrs,
        identity_path=collaboration_args.identity_path,
    )
    utils.log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=collaboration_args.use_ipfs)

    total_batch_size_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    if torch.cuda.device_count() != 0:
        total_batch_size_per_step *= torch.cuda.device_count()

    adjusted_target_batch_size = collaboration_args.target_batch_size - collaboration_args.batch_size_lead

    collaborative_optimizer = hivemind.CollaborativeOptimizer(
        opt=opt,
        dht=dht,
        scheduler=scheduler,
        prefix=collaboration_args.experiment_prefix,
        compression=hivemind.Float16Compression(),
        batch_size_per_step=total_batch_size_per_step,
        bandwidth=collaboration_args.bandwidth,
        target_batch_size=adjusted_target_batch_size,
        client_mode=collaboration_args.client_mode,
        verbose=True,
        start=True,
        **asdict(averager_args),
    )

    class TrainerWithIndependentShuffling(Trainer):
        def get_train_dataloader(self) -> DataLoader:
            """Shuffle data independently for each peer to avoid duplicating batches [important for quality]"""
            torch.manual_seed(hash(local_public_key))
            return super().get_train_dataloader()

    trainer = TrainerWithIndependentShuffling(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        optimizers=(collaborative_optimizer, NoOpScheduler(collaborative_optimizer)),
        callbacks=[
            CollaborativeCallback(
                dht,
                collaborative_optimizer,
                model,
                local_public_key,
                collaboration_args.statistics_expiration,
                collaboration_args.backup_every_steps,
            )
        ],
    )
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)

    # Training
    if training_args.do_train:
        latest_checkpoint_dir = max(
            Path(training_args.output_dir).glob("checkpoint*"), default=None, key=os.path.getctime
        )

        trainer.train(model_path=latest_checkpoint_dir)


if __name__ == "__main__":
    main()
