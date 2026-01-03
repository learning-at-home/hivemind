import os

import torch

torch.multiprocessing.set_sharing_strategy(os.environ.get("HIVEMIND_MEMORY_SHARING_STRATEGY", "file_system"))

from hivemind.averaging import DecentralizedAverager
from hivemind.dht import DHT
from hivemind.moe import (
    ModuleBackend,
    RemoteExpert,
    RemoteMixtureOfExperts,
    RemoteSwitchMixtureOfExperts,
    Server,
    register_expert_class,
)
from hivemind.optim import GradScaler, Optimizer, TrainingAverager

__version__ = "1.2.0.dev0"
