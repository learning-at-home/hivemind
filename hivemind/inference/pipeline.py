"""
Pipeline Parallelism for Decentralized Inference

Enables splitting large language models across multiple nodes globally,
allowing consumer hardware to collectively run massive models.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

import torch
import torch.nn as nn

from hivemind.dht import DHT
from hivemind.moe.client.expert import RemoteExpert, create_remote_experts
from hivemind.moe.server.dht_handler import get_experts
from hivemind.moe.expert_uid import ExpertUID, ExpertInfo
from hivemind.p2p import P2P, PeerID
from hivemind.utils import get_logger, get_dht_time
from hivemind.inference.discovery import LayerDiscoveryProtocol
from hivemind.inference.performance import (
    PerformanceMonitor,
    AdaptiveCompressor,
    SpeculativeExecutor,
    CheckpointManager,
    SmartScheduler
)

logger = get_logger(__name__)


@dataclass
class ModelChunkConfig:
    """Configuration for a model chunk provider"""
    model_name: str
    layer_start: int
    layer_end: int
    device: str = "cuda"
    hidden_dim: int = 4096
    expert_cls: str = "transformer"


class ModelChunkProvider:
    """
    A node that contributes GPU/CPU resources by hosting a chunk of a large model.
    
    This allows users with consumer hardware to contribute specific layers
    of a large model (e.g., Kimi K2.6) to the global compute pool.
    """
    
    def __init__(
        self,
        dht: DHT,
        config: ModelChunkConfig,
        initial_peers: List[str] = None,
        update_period: float = 30.0,
        expiration: float = 120.0
    ):
        self.dht = dht
        self.config = config
        self.initial_peers = initial_peers or []
        self.update_period = update_period
        self.expiration = expiration
        
        # Load the model chunk
        self.model_chunk = self._load_model_chunk()
        self.expert_uid = self._generate_expert_uid()
        
        # Track contributions
        self._compute_times: List[float] = []
        self._tokens_processed: int = 0
        self._lock = threading.Lock()
        
        logger.info(
            f"ModelChunkProvider ready: {config.model_name} "
            f"layers {config.layer_start}-{config.layer_end} on {config.device}"
        )
    
    def _load_model_chunk(self) -> nn.Module:
        """Load the specific chunk of the model this node will host"""
        from hivemind.moe.server.layers import name_to_block
        
        if self.config.expert_cls not in name_to_block:
            raise ValueError(f"Unknown expert class: {self.config.expert_cls}")
            
        model_chunk = name_to_block[self.config.expert_cls](self.config.hidden_dim)
        model_chunk = model_chunk.to(self.config.device)
        return model_chunk
    
    def _generate_expert_uid(self) -> ExpertUID:
        """Generate unique expert UID for this model chunk"""
        return f"{self.config.model_name}.layer.{self.config.layer_start}.{self.config.layer_end}"
    
    def advertise(self):
        """Advertise this model chunk to the DHT network"""
        from hivemind.moe.server.dht_handler import declare_experts
        
        # Store compute capability in DHT
        key = f"compute:{self.config.model_name}:{self.config.layer_start}-{self.config.layer_end}"
        value = {
            "peer_id": self.dht.peer_id.to_base58(),
            "layer_start": self.config.layer_start,
            "layer_end": self.config.layer_end,
            "device_type": "gpu" if "cuda" in self.config.device else "cpu",
            "hidden_dim": self.config.hidden_dim,
            "timestamp": get_dht_time()
        }
        
        self.dht.store(key, value, expiration_time=get_dht_time() + self.expiration)
        logger.debug(f"Advertised model chunk: {key}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Process incoming hidden states through this model chunk.
        
        :param hidden_states: Input tensor from previous layer
        :returns: Output tensor to send to next layer
        """
        start_time = time.time()
        
        with torch.no_grad():
            hidden_states = hidden_states.to(self.config.device)
            output = self.model_chunk(hidden_states)
        
        compute_time = time.time() - start_time
        
        with self._lock:
            self._compute_times.append(compute_time)
            self._tokens_processed += hidden_states.shape[0]
        
        return output.cpu()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this provider's contributions"""
        with self._lock:
            avg_compute_time = sum(self._compute_times) / len(self._compute_times) if self._compute_times else 0
            return {
                "expert_uid": self.expert_uid,
                "total_compute_time": sum(self._compute_times),
                "avg_compute_time": avg_compute_time,
                "tokens_processed": self._tokens_processed,
                "device": self.config.device
            }


class PipelineParallelRunner:
    """
    Runs inference by chaining multiple model chunks across different nodes.
    
    This enables running very large models (like Kimi K2.6) by splitting
    the pipeline across multiple consumer GPUs worldwide.
    
    Performance optimizations included:
    - Speculative execution to hide network latency
    - Adaptive compression for bandwidth optimization
    - Checkpointing for fault tolerance
    - Smart scheduling based on performance metrics
    """
    
    def __init__(
        self, 
        dht: DHT, 
        model_name: str, 
        p2p: P2P = None,
        enable_optimizations: bool = True
    ):
        self.dht = dht
        self.model_name = model_name
        self.p2p = p2p
        self.discovery = LayerDiscoveryProtocol(dht, model_name)
        self._topology: Optional[List[ExpertInfo]] = None
        
        # Performance optimization components
        self.enable_optimizations = enable_optimizations
        if enable_optimizations:
            self.performance_monitor = PerformanceMonitor(dht)
            self.compressor = AdaptiveCompressor(initial_bits=16)
            self.executor = SpeculativeExecutor(
                redundancy_factor=2,
                performance_monitor=self.performance_monitor
            )
            self.checkpointer = CheckpointManager(checkpoint_interval=10, dht=dht)
            self.scheduler = SmartScheduler(self.performance_monitor)
            
            # Start background monitoring
            self.performance_monitor.run_background()
            logger.info("Performance optimizations enabled")
        else:
            self.performance_monitor = None
            self.compressor = None
            self.executor = None
            self.checkpointer = None
            self.scheduler = None
        
    async def discover_topology(self, max_layers: int = 100) -> List[ExpertInfo]:
        """
        Discover available model chunks and assemble the inference pipeline.
        
        :param max_layers: Maximum number of layers to discover
        :returns: Ordered list of experts forming the pipeline
        """
        self._topology = await self.discovery.assemble_pipeline(max_layers)
        
        if not self._topology:
            raise RuntimeError(f"No compute resources found for model {self.model_name}")
        
        logger.info(
            f"Assembled pipeline with {len(self._topology)} chunks "
            f"from {len(set(e.peer_id for e in self._topology))} unique peers"
        )
        
        return self._topology
    
    def _get_p2p(self) -> P2P:
        """Get P2P instance, creating one if necessary"""
        if self.p2p is None:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.p2p = loop.run_until_complete(self.dht.replicate_p2p())
        return self.p2p
    
    async def generate(self, prompt: str, max_tokens: int = 100, session_id: str = None) -> str:
        """
        Generate text by passing through the distributed pipeline with performance optimizations.
        
        :param prompt: Input text prompt
        :param max_tokens: Maximum tokens to generate
        :param session_id: Optional session ID for checkpointing and contribution tracking
        :returns: Generated text
        """
        if self._topology is None:
            await self.discover_topology()
        
        # Tokenize input (simplified - in practice use actual tokenizer)
        hidden_states = self._encode_prompt(prompt)
        
        # Get P2P instance
        p2p = self._get_p2p()
        
        # Use optimized execution if enabled
        if self.enable_optimizations and self.executor:
            hidden_states = await self._generate_optimized(p2p, hidden_states, session_id)
        else:
            # Fallback to simple sequential execution
            for i, expert_info in enumerate(self._topology):
                expert = RemoteExpert(expert_info, p2p)
                
                # Apply compression if enabled
                if self.compressor:
                    compressed, bits_used = self.compressor.compress_tensor(hidden_states)
                    hidden_states = compressed
                
                hidden_states = expert(hidden_states)
                
                # Save checkpoint at intervals
                if self.checkpointer and session_id and self.checkpointer.should_checkpoint(i):
                    self.checkpointer.save_checkpoint(i, hidden_states, session_id)
        
        # Decode output
        generated_text = self._decode_output(hidden_states)
        return generated_text
    
    async def _generate_optimized(
        self, 
        p2p: P2P, 
        hidden_states: torch.Tensor,
        session_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Optimized generation with speculative execution and adaptive compression.
        """
        import numpy as np
        
        for i, expert_info in enumerate(self._topology):
            start_time = time.time()
            
            try:
                # Apply adaptive compression before sending
                if self.compressor:
                    compressed, bits_used = self.compressor.compress_tensor(hidden_states)
                    data_size_mb = compressed.numel() * compressed.element_size() / 1e6
                else:
                    compressed = hidden_states
                    data_size_mb = hidden_states.numel() * hidden_states.element_size() / 1e6
                
                # Execute with speculative redundancy
                result = await self.executor.execute_with_speculation(
                    p2p=p2p,
                    expert_infos=self._topology,
                    input_tensor=compressed,
                    layer_index=i
                )
                
                latency = time.time() - start_time
                
                # Adjust compression based on observed latency
                if self.compressor:
                    self.compressor.adjust_compression(latency, success=True)
                
                # Record performance metrics
                self.performance_monitor.record_inference(
                    peer_id=str(expert_info.peer_id),
                    layer_range=(i, i),
                    latency=latency,
                    data_size_mb=data_size_mb,
                    success=True
                )
                
                hidden_states = result
                
                # Save checkpoint at intervals
                if self.checkpointer and session_id and self.checkpointer.should_checkpoint(i):
                    self.checkpointer.save_checkpoint(i, hidden_states, session_id)
                    
            except Exception as e:
                logger.warning(f"Layer {i} failed: {e}. Attempting recovery...")
                
                # Record failure
                self.performance_monitor.record_inference(
                    peer_id=str(expert_info.peer_id),
                    layer_range=(i, i),
                    latency=5.0,  # Timeout
                    data_size_mb=0,
                    success=False
                )
                
                # Try to recover from checkpoint
                if self.checkpointer and session_id:
                    checkpoint = self.checkpointer.get_latest_checkpoint(i)
                    if checkpoint:
                        ckpt_layer, hidden_states = checkpoint
                        logger.info(f"Recovered from checkpoint at layer {ckpt_layer}")
                        # Retry from checkpoint with different node selection
                        continue
                
                # If recovery fails, raise error
                raise RuntimeError(f"Failed at layer {i} and no checkpoint available")
        
        return hidden_states
    
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Convert prompt to initial hidden states (placeholder)"""
        # In practice, this would use the model's actual tokenizer and embeddings
        batch_size = 1
        seq_len = len(prompt.split())
        hidden_dim = 4096  # Should match model config
        return torch.randn(batch_size, seq_len, hidden_dim)
    
    def _decode_output(self, hidden_states: torch.Tensor) -> str:
        """Convert final hidden states to text (placeholder)"""
        # In practice, this would use the model's actual decoder
        return "[Generated text placeholder]"
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the current pipeline"""
        if not self._topology:
            return {"status": "not_initialized"}
        
        peer_ids = [str(e.peer_id) for e in self._topology]
        unique_peers = set(peer_ids)
        
        stats = {
            "model_name": self.model_name,
            "total_layers": len(self._topology),
            "unique_peers": len(unique_peers),
            "peer_distribution": {peer: peer_ids.count(peer) for peer in unique_peers},
            "optimizations_enabled": self.enable_optimizations
        }
        
        # Add performance metrics if optimizations are enabled
        if self.enable_optimizations and self.performance_monitor:
            stats["performance"] = {
                "tracked_nodes": len(self.performance_monitor._metrics),
                "unreliable_nodes": len(self.performance_monitor.get_unreliable_nodes())
            }
        
        if self.compressor:
            stats["compression"] = self.compressor.get_compression_stats()
        
        return stats
    
    def shutdown(self):
        """Cleanly shutdown all components"""
        if self.performance_monitor:
            self.performance_monitor.shutdown()
        logger.info("PipelineParallelRunner shutdown complete")


async def run_pipeline_example():
    """Example usage of pipeline parallelism with performance optimizations"""
    from hivemind.dht import DHT
    
    # Initialize DHT
    dht = DHT(start=True)
    
    # Create pipeline runner with optimizations enabled (default)
    runner = PipelineParallelRunner(dht, "kimi-k2.6", enable_optimizations=True)
    
    try:
        # Discover available compute resources
        topology = await runner.discover_topology()
        print(f"Found {len(topology)} model chunks")
        
        # Run inference with session tracking
        result = await runner.generate(
            "What is quantum computing?", 
            max_tokens=50,
            session_id="session_001"
        )
        print(f"Generated: {result}")
        
        # Get detailed statistics
        stats = runner.get_pipeline_stats()
        print(f"Pipeline stats: {stats}")
        
    finally:
        runner.shutdown()
        dht.shutdown()
