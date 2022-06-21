from hivemind.averaging import DecentralizedAverager
from hivemind.compression import *
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
from hivemind.p2p import P2P, P2PContext, P2PHandlerError, PeerID, PeerInfo
from hivemind.utils import *

__version__ = "1.2.0.dev0"
