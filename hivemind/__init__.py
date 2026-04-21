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
from hivemind.inference import (
    PipelineParallelRunner,
    ModelChunkProvider,
    ContributionTracker,
    TokenRewardCalculator,
    LayerDiscoveryProtocol,
    ResourceRegistry,
    run_inference_cli,
)

__version__ = "1.2.0.dev0"
