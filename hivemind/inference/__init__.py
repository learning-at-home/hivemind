# Hivemind Inference Module - Ghost Protocol Edition
# Exports all performance-critical components

from .causal_speculator import CausalSpeculator
from .adaptive_bandwidth import AdaptiveCompressor, EntropyAnalyzer
from .reputation_guard import ReputationEngine, NodeProfile
from .swarm_learner import SwarmLearner, MicroGradient, SecureAggregator
from .geo_shard_manager import GeoShardManager
from .pipeline import PipelineParallelRunner

__all__ = [
    'CausalSpeculator',
    'AdaptiveCompressor', 
    'EntropyAnalyzer',
    'ReputationEngine',
    'NodeProfile',
    'SwarmLearner',
    'MicroGradient',
    'SecureAggregator',
    'GeoShardManager',
    'PipelineParallelRunner'
]

__version__ = "2.0.0-ghost"
__status__ = "Production Ready - Real World Optimized"
