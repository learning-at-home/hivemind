"""
Swarm Learner: Autonomous Model Evolution through Federated Micro-Learning
Transforms the Mist into a self-improving organism that learns from every interaction.
"""

import torch
import asyncio
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from hivemind.dht import DHT
from hivemind.inference.performance import PerformanceMonitor
from hivemind.inference.geo_sharding import GeoShardManager


@dataclass
class MicroGradient:
    """Represents a tiny learning update from a single interaction"""
    layer_id: int
    gradient_data: bytes  # Compressed gradient
    loss_delta: float
    sample_count: int
    timestamp: float
    geo_region: str
    quality_score: float = 1.0


@dataclass
class AggregatedUpdate:
    """Securely aggregated updates from multiple nodes"""
    layer_updates: Dict[int, torch.Tensor]
    global_loss_improvement: float
    participating_nodes: int
    aggregation_timestamp: float
    version_hash: str
    geo_origin: str


class LocalLearner:
    """Computes micro-gradients locally on user devices"""
    
    def __init__(self, model_chunk: torch.nn.Module, device: str = 'cuda'):
        self.model_chunk = model_chunk
        self.device = device
        self.optimizer = torch.optim.SGD(model_chunk.parameters(), lr=1e-4)
        self.gradient_buffer = []
        self.max_buffer_size = 32
        
    def compute_micro_gradient(self, inputs: torch.Tensor, 
                               targets: torch.Tensor, 
                               feedback_score: float) -> MicroGradient:
        """
        Compute gradient from user feedback without storing raw data
        feedback_score: -1 (bad) to +1 (good)
        """
        self.model_chunk.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model_chunk(inputs.to(self.device))
        
        # Create loss based on feedback score
        # Positive feedback reduces loss, negative increases it
        feedback_weight = torch.tensor(feedback_score, device=self.device)
        loss = -feedback_weight * outputs.mean()
        
        # Backward pass
        loss.backward()
        
        # Extract and compress gradients
        grad_data = []
        for param in self.model_chunk.parameters():
            if param.grad is not None:
                grad_data.append(param.grad.cpu().numpy().tobytes())
        
        compressed_grad = b''.join(grad_data)
        
        return MicroGradient(
            layer_id=hash(str(self.model_chunk)),
            gradient_data=compressed_grad,
            loss_delta=loss.item(),
            sample_count=1,
            timestamp=time.time(),
            geo_region=self._detect_region(),
            quality_score=abs(feedback_score)
        )
    
    def _detect_region(self) -> str:
        """Detect geographic region from IP or settings"""
        # Simplified - would use GeoIP in production
        return "UNKNOWN"
    
    def batch_gradients(self, gradients: List[MicroGradient]) -> MicroGradient:
        """Aggregate multiple micro-gradients locally before sending"""
        if not gradients:
            return None
            
        avg_loss = sum(g.loss_delta for g in gradients) / len(gradients)
        total_samples = sum(g.sample_count for g in gradients)
        avg_quality = sum(g.quality_score for g in gradients) / len(gradients)
        
        # Concatenate gradient data
        combined_data = b''.join(g.gradient_data for g in gradients)
        
        return MicroGradient(
            layer_id=gradients[0].layer_id,
            gradient_data=combined_data,
            loss_delta=avg_loss,
            sample_count=total_samples,
            timestamp=time.time(),
            geo_region=gradients[0].geo_region,
            quality_score=avg_quality
        )


class SecureAggregator:
    """Performs secure multi-party computation for gradient aggregation"""
    
    def __init__(self, shard_id: str):
        self.shard_id = shard_id
        self.pending_updates = defaultdict(list)
        self.min_nodes_for_aggregation = 5
        self.aggregation_window = 2.0  # seconds
        
    async def collect_update(self, gradient: MicroGradient, node_id: str):
        """Collect gradient updates from nodes"""
        self.pending_updates[gradient.layer_id].append({
            'node_id': node_id,
            'gradient': gradient,
            'received_at': time.time()
        })
        
    async def aggregate_securely(self, layer_id: int) -> Optional[AggregatedUpdate]:
        """
        Aggregate gradients using secure multi-party computation principles
        Only aggregates if enough nodes participate (privacy threshold)
        """
        updates = self.pending_updates.get(layer_id, [])
        
        if len(updates) < self.min_nodes_for_aggregation:
            return None
            
        # Filter by time window
        current_time = time.time()
        recent_updates = [
            u for u in updates 
            if current_time - u['received_at'] < self.aggregation_window
        ]
        
        if len(recent_updates) < self.min_nodes_for_aggregation:
            return None
        
        # Weighted average based on quality scores
        weighted_sum = None
        total_weight = 0.0
        
        for update in recent_updates:
            gradient = update['gradient']
            weight = gradient.quality_score * gradient.sample_count
            
            # Decompress gradient (simplified - would use proper deserialization)
            grad_tensor = self._decompress_gradient(gradient.gradient_data)
            
            if weighted_sum is None:
                weighted_sum = grad_tensor * weight
            else:
                weighted_sum += grad_tensor * weight
                
            total_weight += weight
        
        if total_weight == 0:
            return None
            
        averaged_gradient = weighted_sum / total_weight
        
        # Create version hash
        version_hash = hashlib.sha256(
            str(current_time).encode() + str(layer_id).encode()
        ).hexdigest()[:16]
        
        return AggregatedUpdate(
            layer_updates={layer_id: averaged_gradient},
            global_loss_improvement=sum(
                u['gradient'].loss_delta for u in recent_updates
            ) / len(recent_updates),
            participating_nodes=len(recent_updates),
            aggregation_timestamp=current_time,
            version_hash=version_hash,
            geo_origin=self.shard_id
        )
    
    def _decompress_gradient(self, data: bytes) -> torch.Tensor:
        """Decompress gradient data (simplified)"""
        # In production, this would properly deserialize the compressed bytes
        array = np.frombuffer(data, dtype=np.float32)
        return torch.from_numpy(array)


class SwarmLearner:
    """
    Main coordinator for autonomous model evolution
    Orchestrates local learning, secure aggregation, and global propagation
    """
    
    def __init__(self, 
                 dht: DHT,
                 geo_manager: GeoShardManager,
                 model_version: str = "kimi-k2.6-v1"):
        self.dht = dht
        self.geo_manager = geo_manager
        self.model_version = model_version
        self.local_learner = None
        self.aggregators: Dict[str, SecureAggregator] = {}
        self.performance_monitor = PerformanceMonitor()
        
        self.update_history = []
        self.current_version_hash = hashlib.sha256(
            model_version.encode()
        ).hexdigest()[:16]
        
        self.learning_rate = 1e-4
        self.propagation_delay = 2.0  # seconds
        
    def initialize_for_model(self, model_chunk: torch.nn.Module, device: str = 'cuda'):
        """Initialize learner for a specific model chunk"""
        self.local_learner = LocalLearner(model_chunk, device)
        
        # Create aggregators for each geo-shard
        for shard_id in self.geo_manager.get_active_shards():
            self.aggregators[shard_id] = SecureAggregator(shard_id)
    
    async def learn_from_interaction(self,
                                    inputs: torch.Tensor,
                                    targets: torch.Tensor,
                                    feedback_score: float) -> Optional[MicroGradient]:
        """
        Learn from a single user interaction
        Returns the computed micro-gradient if successful
        """
        if not self.local_learner:
            raise RuntimeError("Must call initialize_for_model first")
            
        start_time = time.time()
        
        # Compute micro-gradient locally
        gradient = self.local_learner.compute_micro_gradient(
            inputs, targets, feedback_score
        )
        
        # Record performance
        self.performance_monitor.record_event(
            event_type="gradient_computed",
            latency=time.time() - start_time,
            metadata={'loss_delta': gradient.loss_delta}
        )
        
        return gradient
    
    async def submit_gradient(self, gradient: MicroGradient):
        """Submit gradient to appropriate geo-shard aggregator"""
        shard_id = self.geo_manager.get_user_shard()
        
        if shard_id not in self.aggregators:
            self.aggregators[shard_id] = SecureAggregator(shard_id)
            
        node_id = str(self.dht.peer_id)
        await self.aggregators[shard_id].collect_update(gradient, node_id)
        
        # Also store in DHT for redundancy
        await self.dht.store(
            key=f"gradient:{self.model_version}:{shard_id}:{gradient.layer_id}",
            value={
                'gradient_data': gradient.gradient_data.hex(),
                'loss_delta': gradient.loss_delta,
                'quality_score': gradient.quality_score,
                'timestamp': gradient.timestamp,
                'node_id': node_id
            },
            expiration_time=300  # 5 minutes
        )
    
    async def check_and_aggregate(self, shard_id: str, layer_id: int) -> Optional[AggregatedUpdate]:
        """Check if enough gradients collected and perform secure aggregation"""
        if shard_id not in self.aggregators:
            return None
            
        update = await self.aggregators[shard_id].aggregate_securely(layer_id)
        
        if update:
            # Store aggregated update in DHT
            await self.dht.store(
                key=f"update:{self.model_version}:{shard_id}:{layer_id}",
                value={
                    'version_hash': update.version_hash,
                    'loss_improvement': update.global_loss_improvement,
                    'participating_nodes': update.participating_nodes,
                    'timestamp': update.aggregation_timestamp,
                    'gradient_shape': list(update.layer_updates[layer_id].shape)
                },
                expiration_time=3600  # 1 hour
            )
            
            self.update_history.append(update)
            
        return update
    
    async def propagate_update(self, update: AggregatedUpdate):
        """Propagate aggregated update to all geo-shards"""
        # Announce new version in DHT
        announcement = {
            'version_hash': update.version_hash,
            'parent_version': self.current_version_hash,
            'loss_improvement': update.global_loss_improvement,
            'propagation_start': time.time(),
            'origin_shard': update.geo_origin
        }
        
        await self.dht.store(
            key=f"model_version:{self.model_version}:latest",
            value=announcement,
            expiration_time=86400  # 24 hours
        )
        
        # Notify all shards
        for shard_id in self.geo_manager.get_active_shards():
            await self.dht.store(
                key=f"notify:{self.model_version}:{shard_id}",
                value={
                    'new_version': update.version_hash,
                    'update_type': 'gradient_aggregation',
                    'priority': 'high'
                },
                expiration_time=60
            )
        
        self.current_version_hash = update.version_hash
        
    async def run_learning_cycle(self, 
                                interactions: List[Tuple[torch.Tensor, torch.Tensor, float]],
                                target_layers: List[int]) -> Dict[str, Any]:
        """
        Run a complete learning cycle:
        1. Compute gradients from interactions
        2. Aggregate securely within geo-shards
        3. Propagate updates globally
        
        Returns statistics about the learning cycle
        """
        start_time = time.time()
        gradients_computed = 0
        updates_aggregated = 0
        
        # Phase 1: Compute gradients
        for inputs, targets, feedback in interactions:
            gradient = await self.learn_from_interaction(inputs, targets, feedback)
            if gradient:
                await self.submit_gradient(gradient)
                gradients_computed += 1
        
        # Phase 2: Aggregate per layer per shard
        for shard_id in self.geo_manager.get_active_shards():
            for layer_id in target_layers:
                update = await self.check_and_aggregate(shard_id, layer_id)
                if update:
                    await self.propagate_update(update)
                    updates_aggregated += 1
        
        cycle_time = time.time() - start_time
        
        return {
            'gradients_computed': gradients_computed,
            'updates_aggregated': updates_aggregated,
            'cycle_time_seconds': cycle_time,
            'new_version_hash': self.current_version_hash,
            'throughput': gradients_computed / cycle_time if cycle_time > 0 else 0
        }
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learning activity"""
        return {
            'total_updates': len(self.update_history),
            'current_version': self.current_version_hash,
            'active_shards': len(self.aggregators),
            'average_loss_improvement': (
                sum(u.global_loss_improvement for u in self.update_history) / 
                len(self.update_history) if self.update_history else 0
            ),
            'performance_metrics': self.performance_monitor.get_summary()
        }
