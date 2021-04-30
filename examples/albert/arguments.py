from typing import Optional, List
from dataclasses import dataclass, field
import uuid

from transformers import TrainingArguments


@dataclass
class BaseTrainingArguments:
    # Primary parameters
    experiment_prefix: str = field(metadata={
        "help": "A unique 'name' of this experiment, used to store metadata on the DHT"})
    initial_peers: List[str] = field(
        default_factory=list,
        metadata={"help": "One or more peers (comma-separated) that will welcome you into the collaboration"}
    )
    averaging_expiration: float = field(
        default=5.0,
        metadata={"help": "Averaging group will wait for stragglers for at most this many seconds"}
    )
    averaging_timeout: float = field(
        default=30.0,
        metadata={"help": "Give up on averaging step after this many seconds"}
    )
    target_batch_size: int = field(
        default=4096,
        metadata={"help": "Perform optimizer step after all peers collectively accumulate this many samples"}
    )
    client_mode: bool = field(
        default=False,
        metadata={"help": "Of True, runs training without incoming connections, in a firewall-compatible mode"}
    )

    # Optional tweaks
    target_group_size: int = field(
        default=256,
        metadata={"help": "Maximum group size for all-reduce"}
    )
    metadata_expiration: float = field(
        default=30,
        metadata={"help": "Peer's metadata will be removed if not updated in this many seconds"}
    )
    dht_listen_on: str = field(
        default="[::]:*",
        metadata={"help": "Network interface used for incoming DHT communication. Default: all ipv6"}
    )
    listen_on: str = field(
        default="[::]:*",
        metadata={"help": "Network interface used for incoming averager communication. Default: all ipv6"}
    )
    batch_size_lead: int = field(
        default=0,
        metadata={"help": "Optional: begin looking for group in advance, this many samples before target_batch_size"}
    )
    compression: str = field(
        default="FLOAT16",
        metadata={"help": "Use this compression when averaging parameters/gradients"}
    )
    min_refresh_period: float = field(
        default=0.5,
        metadata={"help": "Wait for at least this many seconds before fetching new collaboration state"}
    )
    max_refresh_period: float = field(
        default=30,
        metadata={"help": "Wait for at most this many seconds before fetching new collaboration state"}
    )
    default_refresh_period: float = field(
        default=3,
        metadata={"help": "Attempt to fetch collaboration state every this often until successful"}
    )
    expected_drift_peers: float = field(
        default=3,
        metadata={"help": "Trainer assumes that this many new peers can join per step"}
    )
    expected_drift_rate: float = field(
        default=0.2,
        metadata={"help": "Trainer assumes that this fraction of current size can join per step"}
    )
    bandwidth: float = field(
        default=100.0,
        metadata={"help": "Available network bandwidth, in mbps (used for load balancing in all-reduce)"}
    )
    performance_ema_alpha: float = field(
        default=0.1,
        metadata={"help": "Uses this alpha for moving average estimate of samples per second"}
    )


@dataclass
class CollaborationArguments(BaseTrainingArguments):
    trainer_uuid: str = field(
        default=uuid.uuid4().hex,
        metadata={"help": "this peer's name - used when publishing metadata to DHT, default = random"}
    )
    statistics_expiration: float = field(
        default=600,
        metadata={"help": "statistics will be removed if not updated in this many seconds"}
    )
    endpoint: Optional[str] = field(
        default=None,
        metadata={"help": "this node's IP for inbound connections, used when running from behind a proxy"}
    )


@dataclass
class DatasetArguments:
    dataset_path: Optional[str] = field(
        default='data/albert_tokenized_wikitext',
        metadata={"help": "Path to the tokenized dataset"}
    )
    tokenizer_path: Optional[str] = field(
        default='data/tokenizer',
        metadata={"help": "Path to the tokenizer"}
    )
    config_path: Optional[str] = field(
        default='https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-config.json',
        metadata={"help": "Path to the model config"}
    )
    cache_dir: Optional[str] = field(
        default='data',
        metadata={"help": "Path to the cache"}
    )


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

    logging_steps: int = 100
    save_total_limit: int = 2
    save_steps: int = 500

    output_dir: str = 'outputs'
