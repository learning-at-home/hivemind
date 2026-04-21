"""
Neuro-Plastic Topology - Self-Learning Network Routing

Uses reinforcement learning to optimize DHT routing tables in real-time.
Strengthens successful paths, decays failing ones.
The network evolves like a biological nervous system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import time

from hivemind import DHT, PeerID

logger = logging.getLogger(__name__)


@dataclass
class PathExperience:
    """Record of a routing decision and its outcome"""
    from_peer: PeerID
    to_peer: PeerID
    timestamp: float
    latency_ms: float
    success: bool
    payload_size_bytes: int
    retry_count: int


@dataclass
class PeerWeights:
    """Dynamic weights for routing to a peer"""
    peer_id: PeerID
    base_weight: float = 1.0
    latency_factor: float = 1.0
    reliability_factor: float = 1.0
    recency_factor: float = 1.0
    last_updated: float = field(default_factory=time.time)
    
    def get_combined_weight(self) -> float:
        """Calculate combined routing weight"""
        return (
            self.base_weight * 0.4 +
            self.latency_factor * 0.3 +
            self.reliability_factor * 0.2 +
            self.recency_factor * 0.1
        )


@dataclass
class NeuroPlasticConfig:
    """Configuration for Neuro-Plastic Topology"""
    learning_rate: float = 0.01  # How fast to update weights
    decay_rate: float = 0.99  # Decay factor for old experiences
    min_experiences_for_learning: int = 5
    weight_exploration_rate: float = 0.1  # Random exploration rate
    max_path_history: int = 1000
    success_reward: float = 0.1  # Reward for successful routing
    failure_penalty: float = 0.3  # Penalty for failed routing
    latency_threshold_ms: float = 200.0  # Target latency


class NeuroPlasticTopology:
    """
    Self-learning network topology that optimizes routing through experience.
    
    Features:
    - Reinforcement learning for path selection
    - Dynamic weight adjustment based on success/failure
    - Exploration vs exploitation balance
    - Biological-inspired synaptic strengthening/weakening
    """
    
    def __init__(self, dht: DHT, config: Optional[NeuroPlasticConfig] = None):
        self.dht = dht
        self.config = config or NeuroPlasticConfig()
        self.peer_id = dht.peer_id
        
        # Peer weights
        self.peer_weights: Dict[PeerID, PeerWeights] = {}
        
        # Experience history
        self.path_history: List[PathExperience] = []
        self.success_counts: Dict[PeerID, int] = defaultdict(int)
        self.failure_counts: Dict[PeerID, int] = defaultdict(int)
        
        # Statistics
        self.update_count = 0
        self.total_routing_decisions = 0
        
        # State
        self._running = False
        self._learning_task: Optional[asyncio.Task] = None
        
        logger.info("NeuroPlasticTopology initialized")
    
    async def start(self):
        """Start background learning task"""
        self._running = True
        self._learning_task = asyncio.create_task(self._continuous_learning())
        logger.info("NeuroPlasticTopology started")
    
    async def stop(self):
        """Stop learning task"""
        self._running = False
        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass
        logger.info("NeuroPlasticTopology stopped")
    
    async def select_best_peer(
        self,
        candidate_peers: List[PeerID],
        context: Optional[Dict] = None
    ) -> PeerID:
        """
        Select best peer using learned weights with exploration.
        
        Uses epsilon-greedy strategy: mostly exploit best known path,
        occasionally explore alternatives.
        """
        self.total_routing_decisions += 1
        
        # Exploration: random choice
        if np.random.random() < self.config.weight_exploration_rate:
            return np.random.choice(candidate_peers)
        
        # Exploitation: choose highest weighted peer
        best_peer = None
        best_weight = -1
        
        for peer in candidate_peers:
            weight = self._get_peer_weight(peer)
            if weight > best_weight:
                best_weight = weight
                best_peer = peer
        
        return best_peer or candidate_peers[0]
    
    def record_experience(
        self,
        from_peer: PeerID,
        to_peer: PeerID,
        latency_ms: float,
        success: bool,
        payload_size_bytes: int = 0,
        retry_count: int = 0
    ):
        """Record routing experience for learning"""
        experience = PathExperience(
            from_peer=from_peer,
            to_peer=to_peer,
            timestamp=time.time(),
            latency_ms=latency_ms,
            success=success,
            payload_size_bytes=payload_size_bytes,
            retry_count=retry_count
        )
        
        self.path_history.append(experience)
        
        # Trim history if too long
        if len(self.path_history) > self.config.max_path_history:
            self.path_history = self.path_history[-self.config.max_path_history:]
        
        # Update counters
        if success:
            self.success_counts[to_peer] += 1
        else:
            self.failure_counts[to_peer] += 1
        
        # Immediate weight update
        self._update_peer_weight(to_peer, experience)
    
    async def get_peer_reliability(self, peer_id: PeerID) -> float:
        """Get reliability score for a peer (0.0 to 1.0)"""
        if peer_id not in self.success_counts and peer_id not in self.failure_counts:
            return 0.5  # Unknown peer
        
        total = self.success_counts[peer_id] + self.failure_counts[peer_id]
        if total == 0:
            return 0.5
        
        return self.success_counts[peer_id] / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get topology statistics"""
        avg_weight = np.mean([w.get_combined_weight() for w in self.peer_weights.values()]) if self.peer_weights else 0
        
        return {
            "peers_tracked": len(self.peer_weights),
            "total_experiences": len(self.path_history),
            "routing_decisions": self.total_routing_decisions,
            "update_count": self.update_count,
            "average_peer_weight": float(avg_weight),
            "exploration_rate": self.config.weight_exploration_rate,
            "learning_rate": self.config.learning_rate
        }
    
    def _get_peer_weight(self, peer_id: PeerID) -> float:
        """Get current weight for a peer"""
        if peer_id not in self.peer_weights:
            # Initialize new peer with default weights
            self.peer_weights[peer_id] = PeerWeights(peer_id=peer_id)
        
        weights = self.peer_weights[peer_id]
        
        # Apply recency decay
        time_since_update = time.time() - weights.last_updated
        weights.recency_factor = np.exp(-time_since_update / 3600)  # Decay over 1 hour
        
        return weights.get_combined_weight()
    
    def _update_peer_weight(self, peer_id: PeerID, experience: PathExperience):
        """Update peer weights based on experience"""
        if peer_id not in self.peer_weights:
            self.peer_weights[peer_id] = PeerWeights(peer_id=peer_id)
        
        weights = self.peer_weights[peer_id]
        
        # Calculate reward/penalty
        reward = 0.0
        
        if experience.success:
            reward += self.config.success_reward
            
            # Bonus for low latency
            if experience.latency_ms < self.config.latency_threshold_ms:
                reward += self.config.success_reward * 0.5
        else:
            reward -= self.config.failure_penalty
        
        # Penalty for retries
        reward -= experience.retry_count * 0.05
        
        # Update base weight with learning rate
        delta = reward * self.config.learning_rate
        weights.base_weight = np.clip(weights.base_weight + delta, 0.1, 2.0)
        
        # Update latency factor
        latency_ratio = experience.latency_ms / max(self.config.latency_threshold_ms, 1)
        weights.latency_factor = np.clip(1.0 / max(latency_ratio, 0.1), 0.5, 2.0)
        
        # Update reliability factor
        total = self.success_counts[peer_id] + self.failure_counts[peer_id]
        if total > 0:
            reliability = self.success_counts[peer_id] / total
            weights.reliability_factor = 0.5 + reliability  # Range: 0.5 to 1.5
        
        weights.last_updated = experience.timestamp
        self.update_count += 1
        
        logger.debug(f"Updated weights for {peer_id}: base={weights.base_weight:.3f}, reward={reward:.3f}")
    
    async def _continuous_learning(self):
        """Continuously learn from accumulated experiences"""
        while self._running:
            try:
                await asyncio.sleep(10.0)  # Learn every 10 seconds
                
                # Skip if not enough experiences
                if len(self.path_history) < self.config.min_experiences_for_learning:
                    continue
                
                # Apply decay to all weights (forgetting mechanism)
                for peer_id, weights in list(self.peer_weights.items()):
                    weights.base_weight *= self.config.decay_rate
                    weights.base_weight = max(0.1, weights.base_weight)
                
                # Log periodic stats
                if self.update_count % 100 == 0:
                    stats = self.get_stats()
                    logger.debug(f"Topology learning stats: {stats}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Learning error: {e}")
    
    async def optimize_routing_table(self):
        """Explicitly optimize DHT routing table based on learned weights"""
        # Get current routing table from DHT
        # In production, this would modify the actual DHT routing
        
        # Sort peers by weight
        sorted_peers = sorted(
            self.peer_weights.keys(),
            key=lambda p: self._get_peer_weight(p),
            reverse=True
        )
        
        logger.info(f"Optimized routing table: top 5 peers = {sorted_peers[:5]}")
        
        return sorted_peers
