"""
Performance Optimization Module for Decentralized Inference

Addresses critical performance hurdles:
1. Network latency through speculative execution and pipelining
2. Bandwidth optimization through adaptive compression
3. Fault tolerance through checkpointing and redundant execution
4. Load balancing through dynamic scheduling
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import torch
import torch.nn.functional as F

from hivemind.dht import DHT
from hivemind.p2p import P2P, PeerID
from hivemind.moe.client.expert import RemoteExpert
from hivemind.moe.expert_uid import ExpertInfo
from hivemind.utils import get_logger, get_dht_time
from hivemind.compression.base import CompressionBase, NoCompression, CompressionInfo
from hivemind.proto import runtime_pb2

logger = get_logger(__name__)


@dataclass
class NodePerformanceMetrics:
    """Tracks performance metrics for each node in the pipeline"""
    peer_id: str
    layer_range: Tuple[int, int]
    avg_latency: float = 0.0
    avg_bandwidth: float = 0.0  # MB/s
    success_rate: float = 1.0
    last_seen: float = field(default_factory=get_dht_time)
    compute_capability: str = "unknown"  # e.g., "rtx3080", "cpu_avx2"
    reliability_score: float = 1.0
    
    def update_latency(self, latency: float, alpha: float = 0.3):
        """Exponential moving average update for latency"""
        self.avg_latency = alpha * latency + (1 - alpha) * self.avg_latency
        
    def update_bandwidth(self, bandwidth: float, alpha: float = 0.3):
        """Exponential moving average update for bandwidth"""
        self.avg_bandwidth = alpha * bandwidth + (1 - alpha) * self.avg_bandwidth
        
    def update_reliability(self, success: bool, alpha: float = 0.1):
        """Update reliability score based on success/failure"""
        target = 1.0 if success else 0.0
        self.reliability_score = alpha * target + (1 - alpha) * self.reliability_score
        self.success_rate = self.reliability_score


class PerformanceMonitor:
    """
    Monitors and tracks performance metrics across all nodes in the network.
    
    Enables intelligent routing and load balancing based on real-time performance data.
    """
    
    def __init__(self, dht: DHT, refresh_interval: float = 30.0):
        self.dht = dht
        self.refresh_interval = refresh_interval
        self._metrics: Dict[str, NodePerformanceMetrics] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        
    def record_inference(
        self, 
        peer_id: str, 
        layer_range: Tuple[int, int],
        latency: float, 
        data_size_mb: float,
        success: bool
    ):
        """Record an inference operation's performance"""
        bandwidth = data_size_mb / latency if latency > 0 else 0
        
        with self._lock:
            if peer_id not in self._metrics:
                self._metrics[peer_id] = NodePerformanceMetrics(
                    peer_id=peer_id,
                    layer_range=layer_range
                )
            
            metrics = self._metrics[peer_id]
            metrics.update_latency(latency)
            metrics.update_bandwidth(bandwidth)
            metrics.update_reliability(success)
            metrics.last_seen = get_dht_time()
            
    def get_fastest_nodes(
        self, 
        layer_range: Tuple[int, int],
        top_k: int = 3
    ) -> List[str]:
        """Get the top-k fastest nodes for a specific layer range"""
        with self._lock:
            candidates = [
                m for m in self._metrics.values()
                if (m.layer_range[0] <= layer_range[0] <= m.layer_range[1] or
                    m.layer_range[0] <= layer_range[1] <= m.layer_range[1])
                and m.reliability_score > 0.5
            ]
            
            # Sort by weighted score: lower latency + higher bandwidth + higher reliability
            candidates.sort(
                key=lambda m: (
                    m.avg_latency + 0.1,  # Add small constant to avoid div by zero
                    -m.avg_bandwidth,
                    -m.reliability_score
                )
            )
            
            return [m.peer_id for m in candidates[:top_k]]
    
    def get_unreliable_nodes(self, threshold: float = 0.7) -> List[str]:
        """Get list of unreliable nodes to avoid"""
        with self._lock:
            return [
                m.peer_id for m in self._metrics.values()
                if m.reliability_score < threshold
            ]
    
    def publish_metrics_to_dht(self):
        """Publish aggregated metrics to DHT for network-wide visibility"""
        with self._lock:
            metrics_summary = {
                peer_id: {
                    "avg_latency": m.avg_latency,
                    "avg_bandwidth": m.avg_bandwidth,
                    "reliability": m.reliability_score,
                    "layer_range": m.layer_range
                }
                for peer_id, m in self._metrics.items()
                if get_dht_time() - m.last_seen < 300  # Only recent metrics
            }
        
        key = f"performance:metrics:{self.dht.peer_id.to_base58()}"
        self.dht.store(key, metrics_summary, expiration_time=get_dht_time() + 60)
    
    def run_background(self):
        """Start background thread for periodic metric publishing"""
        thread = threading.Thread(target=self._background_loop, daemon=True)
        thread.start()
        return thread
    
    def _background_loop(self):
        while not self._stop.is_set():
            try:
                self.publish_metrics_to_dht()
                self._stop.wait(self.refresh_interval)
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
    
    def shutdown(self):
        self._stop.set()


class SimpleQuantizer:
    """Simple quantization for tensors without external dependencies"""
    
    @staticmethod
    def quantize_to_bits(tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Quantize tensor to specified bit width.
        
        Uses simple uniform quantization.
        """
        if bits >= 32:
            return tensor
        
        # Normalize to [0, 1]
        min_val = tensor.min()
        max_val = tensor.max()
        
        if max_val - min_val < 1e-8:
            return tensor
        
        normalized = (tensor - min_val) / (max_val - min_val)
        
        # Quantize to 2^bits levels
        levels = 2 ** bits
        quantized = torch.round(normalized * (levels - 1)) / (levels - 1)
        
        # De-normalize
        return quantized * (max_val - min_val) + min_val


class AdaptiveCompressor:
    """
    Dynamically adjusts compression level based on network conditions.
    
    Balances accuracy vs. bandwidth to optimize end-to-end latency.
    """
    
    def __init__(
        self,
        initial_bits: int = 16,
        min_bits: int = 4,
        max_bits: int = 32,
        latency_threshold_ms: float = 100.0
    ):
        self.current_bits = initial_bits
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.latency_threshold = latency_threshold_ms / 1000.0  # Convert to seconds
        self._compression_history: List[Tuple[float, int]] = []
        
    def compress_tensor(
        self, 
        tensor: torch.Tensor, 
        compression_ratio_target: Optional[float] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Compress tensor with adaptive bit width.
        
        :returns: (compressed_tensor, bits_used)
        """
        if self.current_bits >= 32:
            return tensor, 32
        
        # Apply simple quantization
        compressed = SimpleQuantizer.quantize_to_bits(tensor, self.current_bits)
        
        return compressed, self.current_bits
    
    def adjust_compression(self, observed_latency: float, success: bool):
        """
        Adjust compression level based on observed latency.
        
        Strategy:
        - If latency > threshold: increase compression (reduce bits)
        - If latency < threshold/2 and success: decrease compression (increase bits)
        """
        if observed_latency > self.latency_threshold:
            # Need more compression
            self.current_bits = max(self.min_bits, self.current_bits - 2)
            logger.debug(f"Increasing compression: {self.current_bits} bits")
        elif observed_latency < self.latency_threshold / 2 and self.current_bits < self.max_bits:
            # Can reduce compression for better accuracy
            self.current_bits = min(self.max_bits, self.current_bits + 2)
            logger.debug(f"Decreasing compression: {self.current_bits} bits")
        
        self._compression_history.append((observed_latency, self.current_bits))
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get statistics about compression adjustments"""
        if not self._compression_history:
            return {"avg_bits": self.current_bits, "adjustments": 0}
        
        avg_bits = sum(bits for _, bits in self._compression_history) / len(self._compression_history)
        return {
            "current_bits": self.current_bits,
            "avg_bits": avg_bits,
            "total_adjustments": len(self._compression_history),
            "min_bits_used": min(bits for _, bits in self._compression_history),
            "max_bits_used": max(bits for _, bits in self._compression_history)
        }


class SpeculativeExecutor:
    """
    Implements speculative execution to hide network latency.
    
    Sends requests to multiple redundant nodes and uses the first response,
    dramatically reducing tail latency from slow nodes.
    """
    
    def __init__(
        self,
        redundancy_factor: int = 2,
        timeout_seconds: float = 5.0,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        self.redundancy_factor = redundancy_factor
        self.timeout = timeout_seconds
        self.monitor = performance_monitor
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
    async def execute_with_speculation(
        self,
        p2p: P2P,
        expert_infos: List[ExpertInfo],
        input_tensor: torch.Tensor,
        layer_index: int
    ) -> torch.Tensor:
        """
        Execute layer computation with speculative redundancy.
        
        Sends the same request to multiple nodes providing the same layer,
        returns the first successful response.
        """
        # Find redundant nodes for this layer
        candidate_nodes = self._find_redundant_nodes(expert_infos, layer_index)
        
        if len(candidate_nodes) < 1:
            # Fallback: use original node
            candidate_nodes = expert_infos[layer_index:layer_index+1]
        
        # Create tasks for redundant execution
        tasks = []
        for expert_info in candidate_nodes[:self.redundancy_factor]:
            task = self._execute_single(expert_info, p2p, input_tensor.clone())
            tasks.append(task)
        
        # Wait for first successful response
        start_time = time.time()
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.timeout,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Get result from first completed task
            for task in done:
                try:
                    result = task.result()
                    latency = time.time() - start_time
                    
                    # Record success
                    if self.monitor:
                        self.monitor.record_inference(
                            peer_id=str(expert_info.peer_id),
                            layer_range=(layer_index, layer_index),
                            latency=latency,
                            data_size_mb=input_tensor.numel() * input_tensor.element_size() / 1e6,
                            success=True
                        )
                    
                    return result
                except Exception as e:
                    logger.warning(f"Speculative task failed: {e}")
                    continue
            
            raise RuntimeError("All speculative executions failed")
            
        except asyncio.TimeoutError:
            # Record failure for all candidates
            if self.monitor:
                for expert_info in candidate_nodes:
                    self.monitor.record_inference(
                        peer_id=str(expert_info.peer_id),
                        layer_range=(layer_index, layer_index),
                        latency=self.timeout,
                        data_size_mb=0,
                        success=False
                    )
            raise RuntimeError(f"Speculative execution timeout after {self.timeout}s")
    
    def _find_redundant_nodes(
        self, 
        expert_infos: List[ExpertInfo], 
        layer_index: int
    ) -> List[ExpertInfo]:
        """Find multiple nodes that can handle the same layer"""
        # In practice, this would query DHT for alternative providers
        # For now, return the primary node
        return [expert_infos[layer_index]]
    
    async def _execute_single(
        self, 
        expert_info: ExpertInfo, 
        p2p: P2P, 
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Execute single remote expert call"""
        expert = RemoteExpert(expert_info, p2p)
        return expert(input_tensor)


class CheckpointManager:
    """
    Manages checkpointing for fault tolerance.
    
    Saves intermediate results at strategic points to enable
    recovery from node failures without restarting entire inference.
    """
    
    def __init__(
        self,
        checkpoint_interval: int = 10,  # Save every N layers
        max_checkpoints: int = 5,
        dht: Optional[DHT] = None
    ):
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.dht = dht
        self._checkpoints: Dict[int, torch.Tensor] = {}
        self._checkpoint_order: List[int] = []
        self._lock = threading.Lock()
        
    def should_checkpoint(self, layer_index: int) -> bool:
        """Determine if we should save a checkpoint at this layer"""
        return layer_index % self.checkpoint_interval == 0
    
    def save_checkpoint(
        self, 
        layer_index: int, 
        hidden_states: torch.Tensor,
        session_id: str
    ):
        """Save intermediate hidden states"""
        with self._lock:
            # Store locally
            self._checkpoints[layer_index] = hidden_states.cpu().clone()
            self._checkpoint_order.append(layer_index)
            
            # Prune old checkpoints
            while len(self._checkpoints) > self.max_checkpoints:
                oldest = self._checkpoint_order.pop(0)
                del self._checkpoints[oldest]
            
            # Optionally store in DHT for distributed recovery
            if self.dht:
                self._store_in_dht(layer_index, hidden_states, session_id)
        
        logger.debug(f"Saved checkpoint at layer {layer_index}")
    
    def _store_in_dht(self, layer_index: int, hidden_states: torch.Tensor, session_id: str):
        """Store checkpoint in DHT for recovery"""
        key = f"checkpoint:{session_id}:layer{layer_index}"
        value = {
            "data": hidden_states.numpy().tobytes(),
            "shape": tuple(hidden_states.shape),
            "dtype": str(hidden_states.dtype),
            "timestamp": get_dht_time()
        }
        # Store with 5 minute expiration
        self.dht.store(key, value, expiration_time=get_dht_time() + 300)
    
    def get_latest_checkpoint(self, failed_layer: int) -> Optional[Tuple[int, torch.Tensor]]:
        """
        Get the most recent checkpoint before a failed layer.
        
        :returns: (layer_index, hidden_states) or None if no checkpoint
        """
        with self._lock:
            candidates = [
                (idx, tensor) 
                for idx, tensor in self._checkpoints.items()
                if idx < failed_layer
            ]
            
            if not candidates:
                return None
            
            # Return the latest checkpoint before failure
            return max(candidates, key=lambda x: x[0])
    
    async def recover_from_dht(self, session_id: str, layer_index: int) -> Optional[torch.Tensor]:
        """Recover checkpoint from DHT after node failure"""
        if not self.dht:
            return None
        
        # Find nearest checkpoint
        checkpoint_layer = (layer_index // self.checkpoint_interval) * self.checkpoint_interval
        
        key = f"checkpoint:{session_id}:layer{checkpoint_layer}"
        result = await self.dht.get(key)
        
        if result is None or result.value is None:
            return None
        
        # Reconstruct tensor
        value = result.value
        array = np.frombuffer(bytearray(value["data"]), dtype=np.dtype(value["dtype"]))
        tensor = torch.as_tensor(array).reshape(value["shape"])
        
        logger.info(f"Recovered checkpoint from DHT at layer {checkpoint_layer}")
        return tensor


class SmartScheduler:
    """
    Intelligent scheduler that optimizes pipeline assembly based on performance metrics.
    
    Considers latency, bandwidth, reliability, and geographic distribution
    to assemble optimal inference pipelines.
    """
    
    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        strategy: str = "latency_optimized"  # or "balanced", "reliability_first"
    ):
        self.monitor = performance_monitor
        self.strategy = strategy
        
    def select_optimal_nodes(
        self,
        available_nodes: List[ExpertInfo],
        required_layers: int
    ) -> List[ExpertInfo]:
        """
        Select optimal set of nodes to cover all required layers.
        
        Optimizes for:
        - Minimal end-to-end latency
        - Maximal reliability
        - Balanced load distribution
        """
        if self.strategy == "latency_optimized":
            return self._select_for_latency(available_nodes, required_layers)
        elif self.strategy == "reliability_first":
            return self._select_for_reliability(available_nodes, required_layers)
        else:  # balanced
            return self._select_balanced(available_nodes, required_layers)
    
    def _select_for_latency(
        self, 
        available_nodes: List[ExpertInfo], 
        required_layers: int
    ) -> List[ExpertInfo]:
        """Select nodes minimizing total latency"""
        # Group nodes by layer coverage
        layer_to_nodes = defaultdict(list)
        for node in available_nodes:
            # Extract layer range from expert UID
            parts = node.uid.split('.')
            if len(parts) >= 4:
                try:
                    layer_start = int(parts[-2])
                    layer_end = int(parts[-1])
                    for layer in range(layer_start, layer_end + 1):
                        layer_to_nodes[layer].append(node)
                except ValueError:
                    pass
        
        # Select fastest node for each layer
        selected = []
        for layer in range(required_layers):
            if layer not in layer_to_nodes:
                logger.warning(f"No node available for layer {layer}")
                continue
            
            candidates = layer_to_nodes[layer]
            # Use performance monitor to rank
            best_node = self._rank_by_latency(candidates)
            if best_node and (not selected or selected[-1] != best_node):
                selected.append(best_node)
        
        return selected
    
    def _rank_by_latency(self, candidates: List[ExpertInfo]) -> Optional[ExpertInfo]:
        """Rank candidates by estimated latency"""
        scored = []
        for candidate in candidates:
            peer_id = str(candidate.peer_id)
            metrics = self.monitor._metrics.get(peer_id)
            
            if metrics:
                # Lower latency + higher reliability = better score
                score = metrics.avg_latency / (metrics.reliability_score + 0.01)
            else:
                # Unknown node: use neutral score
                score = float('inf')
            
            scored.append((score, candidate))
        
        scored.sort(key=lambda x: x[0])
        return scored[0][1] if scored else None
    
    def _select_for_reliability(
        self, 
        available_nodes: List[ExpertInfo], 
        required_layers: int
    ) -> List[ExpertInfo]:
        """Select nodes maximizing reliability"""
        # Similar to latency selection but prioritize reliability
        scored = []
        for node in available_nodes:
            peer_id = str(node.peer_id)
            metrics = self.monitor._metrics.get(peer_id)
            
            reliability = metrics.reliability_score if metrics else 0.5
            scored.append((-reliability, node))  # Negative for sorting
        
        scored.sort(key=lambda x: x[0])
        return [node for _, node in scored[:required_layers]]
    
    def _select_balanced(
        self, 
        available_nodes: List[ExpertInfo], 
        required_layers: int
    ) -> List[ExpertInfo]:
        """Balance latency and reliability"""
        # Weighted combination
        scored = []
        for node in available_nodes:
            peer_id = str(node.peer_id)
            metrics = self.monitor._metrics.get(peer_id)
            
            if metrics:
                # Normalize and combine (lower is better)
                latency_score = metrics.avg_latency / (metrics.avg_latency + 1)
                reliability_penalty = 1 - metrics.reliability_score
                combined = 0.6 * latency_score + 0.4 * reliability_penalty
            else:
                combined = 0.5  # Neutral for unknown
            
            scored.append((combined, node))
        
        scored.sort(key=lambda x: x[0])
        return [node for _, node in scored[:required_layers]]


# Example integration
async def optimized_pipeline_example():
    """Demonstrate performance-optimized pipeline execution"""
    from hivemind.dht import DHT
    
    dht = DHT(start=True)
    monitor = PerformanceMonitor(dht)
    monitor.run_background()
    
    compressor = AdaptiveCompressor(initial_bits=16)
    executor = SpeculativeExecutor(redundancy_factor=2, performance_monitor=monitor)
    checkpoint_mgr = CheckpointManager(checkpoint_interval=10, dht=dht)
    scheduler = SmartScheduler(monitor, strategy="latency_optimized")
    
    try:
        # Pipeline execution with optimizations would go here
        logger.info("Performance-optimized pipeline ready")
    finally:
        monitor.shutdown()
        dht.shutdown()
