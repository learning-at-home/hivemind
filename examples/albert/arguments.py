from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments


@dataclass
class BaseTrainingArguments:
    run_id: str = field(
        default="albert", metadata={"help": "A unique 'name' of this experiment, used to store metadata on the DHT"}
    )
    initial_peers: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Multiaddrs of the peers that will welcome you into the existing collaboration. "
            "Example: /ip4/203.0.113.1/tcp/31337/p2p/XXXX /ip4/203.0.113.2/tcp/7777/p2p/YYYY"
        },
    )
    use_ipfs: bool = field(
        default=False,
        metadata={
            "help": "Use IPFS to find initial_peers. If enabled, you only need to provide /p2p/XXXX part of the multiaddrs "
            "for the initial_peers (no need to specify a particular IPv4/IPv6 host and port)"
        },
    )
    host_maddrs: List[str] = field(
        default_factory=lambda: ["/ip4/0.0.0.0/tcp/0"],
        metadata={
            "help": "Multiaddrs to listen for external connections from other p2p instances. "
            "Defaults to all IPv4 interfaces and the TCP protocol: /ip4/0.0.0.0/tcp/0"
        },
    )
    announce_maddrs: List[str] = field(
        default_factory=list,
        metadata={"help": "Visible multiaddrs the host announces for external connections from other p2p instances"},
    )
    identity_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a pre-generated private key file. If defined, makes the peer ID deterministic. "
            "If the file does not exist yet, writes a new private key to this file."
        },
    )


@dataclass
class AveragerArguments:
    target_group_size: int = field(default=256, metadata={"help": "Maximum group size for all-reduce"})


@dataclass
class ProgressTrackerArguments:
    min_refresh_period: float = field(
        default=0.5, metadata={"help": "Wait for at least this many seconds before fetching new collaboration state"}
    )
    max_refresh_period: float = field(
        default=30, metadata={"help": "Wait for at most this many seconds before fetching new collaboration state"}
    )
    default_refresh_period: float = field(
        default=3, metadata={"help": "Attempt to fetch collaboration state every this often until successful"}
    )
    expected_drift_peers: float = field(
        default=3, metadata={"help": "Trainer assumes that this many new peers can join per step"}
    )
    expected_drift_rate: float = field(
        default=0.2, metadata={"help": "Trainer assumes that this fraction of current size can join per step"}
    )
    metadata_expiration: float = field(
        default=120, metadata={"help": "Peer's metadata will be removed if not updated in this many seconds"}
    )


@dataclass
class OptimizerArguments:
    target_batch_size: int = field(
        default=4096,
        metadata={"help": "Perform optimizer step after all peers collectively accumulate this many samples"},
    )
    client_mode: bool = field(
        default=False,
        metadata={"help": "Of True, runs training without incoming connections, in a firewall-compatible mode"},
    )
    batch_size_lead: int = field(
        default=0,
        metadata={"help": "Optional: begin looking for group in advance, this many samples before target_batch_size"},
    )
    bandwidth: float = field(
        default=100.0,
        metadata={"help": "Available network bandwidth, in mbps (used for load balancing in all-reduce)"},
    )
    averaging_timeout: float = field(
        default=60.0, metadata={"help": "Give up on averaging step after this many seconds"}
    )
    matchmaking_time: float = field(
        default=5.0, metadata={"help": "When looking for group, wait for requests for at least this many seconds"}
    )


@dataclass
class CollaborationArguments(OptimizerArguments, BaseTrainingArguments):
    statistics_expiration: float = field(
        default=600, metadata={"help": "Statistics will be removed if not updated in this many seconds"}
    )
    backup_every_steps: int = field(
        default=10, metadata={"help": "Frequency of backups to restore from in case of encountering NaN values"}
    )


@dataclass
class DatasetArguments:
    dataset_path: Optional[str] = field(
        default="data/albert_tokenized_wikitext", metadata={"help": "Path to the tokenized dataset"}
    )
    tokenizer_path: Optional[str] = field(default="data/tokenizer", metadata={"help": "Path to the tokenizer"})
    config_path: Optional[str] = field(
        default="https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-config.json",
        metadata={"help": "Path to the model config"},
    )
    cache_dir: Optional[str] = field(default="data", metadata={"help": "Path to the cache"})


@dataclass
class AlbertTrainingArguments(TrainingArguments):
    dataloader_num_workers: int = 4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    seq_length: int = 512

    total_steps: int = 125_000  # please note: this only affects the learning rate schedule
    learning_rate: float = 0.00176
    warmup_steps: int = 5000
    adam_epsilon: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    clamp_value: float = 10000.0

    fp16: bool = True
    fp16_opt_level: str = "O2"
    do_train: bool = True
    do_eval: bool = False

    logging_dir: str = "logs"
    output_dir: str = "outputs"
    logging_steps: int = 100
    logging_first_step: bool = True
    overwrite_output_dir: bool = True

    save_total_limit: int = 2
    save_steps: int = 500
    max_steps: int = 10**30  # meant as "peer should compute gradients forever"
