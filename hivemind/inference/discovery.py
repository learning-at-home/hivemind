"""
Layer Discovery Protocol

Discovers and assembles available model chunks from across the DHT network
to form complete inference pipelines for large language models.
"""

import asyncio
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

from hivemind.dht import DHT
from hivemind.moe.expert_uid import ExpertInfo
from hivemind.p2p import PeerID
from hivemind.utils import get_logger, get_dht_time

logger = get_logger(__name__)


@dataclass
class ResourceAdvertisement:
    """Represents an advertised compute resource"""
    peer_id: str
    model_name: str
    layer_start: int
    layer_end: int
    device_type: str  # 'gpu' or 'cpu'
    hidden_dim: int
    timestamp: float
    expiration: float
    
    @property
    def layer_range(self) -> Tuple[int, int]:
        return (self.layer_start, self.layer_end)
    
    @property
    def is_expired(self) -> bool:
        return get_dht_time() > self.expiration


class ResourceRegistry:
    """
    Maintains a registry of available compute resources for each model.
    
    This class queries the DHT to find all available model chunks
    and organizes them by model name and layer range.
    """
    
    def __init__(self, dht: DHT):
        self.dht = dht
        self._cache: Dict[str, Dict[Tuple[int, int], ResourceAdvertisement]] = defaultdict(dict)
        self._last_refresh: float = 0
        self._refresh_interval: float = 60.0  # Refresh every 60 seconds
    
    async def discover_resources(self, model_name: str) -> List[ResourceAdvertisement]:
        """
        Discover all available compute resources for a specific model.
        
        :param model_name: Name of the model (e.g., 'kimi-k2.6')
        :returns: List of resource advertisements
        """
        # Check cache first
        current_time = get_dht_time()
        if (current_time - self._last_refresh < self._refresh_interval and 
            model_name in self._cache):
            resources = list(self._cache[model_name].values())
            # Filter out expired
            return [r for r in resources if not r.is_expired]
        
        # Query DHT for compute resources
        prefix = f"compute:{model_name}:"
        resources = await self._query_dht_for_prefix(prefix)
        
        # Update cache
        self._cache[model_name] = {
            (r.layer_start, r.layer_end): r for r in resources
        }
        self._last_refresh = current_time
        
        logger.info(
            f"Discovered {len(resources)} compute resources for {model_name}"
        )
        
        return resources
    
    async def _query_dht_for_prefix(self, prefix: str) -> List[ResourceAdvertisement]:
        """Query DHT for all keys matching a prefix"""
        resources = []
        
        # In practice, DHT would support prefix queries
        # For now, we'll use a simplified approach
        # This would need to be implemented based on actual DHT capabilities
        
        try:
            # Try to get values with the prefix
            # Note: This is a simplified implementation
            # A real implementation would use DHT traversal or indexing
            
            # For demonstration, we'll store and retrieve a test key
            test_key = f"{prefix}0-1"
            value = await self.dht.get(test_key)
            
            if value is not None and isinstance(value.value, dict):
                resource = ResourceAdvertisement(
                    peer_id=value.value.get("peer_id"),
                    model_name=prefix.split(':')[1],
                    layer_start=value.value.get("layer_start", 0),
                    layer_end=value.value.get("layer_end", 1),
                    device_type=value.value.get("device_type", "cpu"),
                    hidden_dim=value.value.get("hidden_dim", 4096),
                    timestamp=value.value.get("timestamp", 0),
                    expiration=value.value.get("expiration", float('inf'))
                )
                resources.append(resource)
                
        except Exception as e:
            logger.debug(f"No resources found for prefix {prefix}: {e}")
        
        return resources
    
    def get_layer_coverage(self, model_name: str) -> Dict[str, any]:
        """
        Analyze layer coverage for a model.
        
        :returns: Dict with coverage statistics
        """
        if model_name not in self._cache:
            return {"total_layers": 0, "covered_layers": set(), "gaps": []}
        
        resources = list(self._cache[model_name].values())
        covered_layers = set()
        
        for resource in resources:
            if not resource.is_expired:
                for layer in range(resource.layer_start, resource.layer_end + 1):
                    covered_layers.add(layer)
        
        # Find gaps (assuming we know total layers, e.g., 100 for Kimi K2.6)
        max_expected_layers = 100
        all_layers = set(range(max_expected_layers))
        gaps = sorted(all_layers - covered_layers)
        
        return {
            "total_layers": max_expected_layers,
            "covered_layers": covered_layers,
            "coverage_percentage": len(covered_layers) / max_expected_layers * 100,
            "gaps": gaps[:10],  # Show first 10 gaps
            "num_providers": len(resources)
        }


class LayerDiscoveryProtocol:
    """
    Protocol for discovering and assembling model layers into a complete pipeline.
    
    This class finds available compute resources and assembles them into
    an ordered pipeline that can run inference for a complete model.
    """
    
    def __init__(self, dht: DHT, model_name: str):
        self.dht = dht
        self.model_name = model_name
        self.registry = ResourceRegistry(dht)
    
    async def assemble_pipeline(
        self, 
        max_layers: int = 100,
        prefer_gpu: bool = True
    ) -> List[ExpertInfo]:
        """
        Assemble a complete inference pipeline from available resources.
        
        :param max_layers: Maximum number of layers to assemble
        :param prefer_gpu: If True, prefer GPU resources over CPU
        :returns: Ordered list of ExpertInfo forming the pipeline
        """
        # Discover available resources
        resources = await self.registry.discover_resources(self.model_name)
        
        if not resources:
            logger.warning(f"No resources found for model {self.model_name}")
            return []
        
        # Sort resources by layer range
        resources.sort(key=lambda r: r.layer_start)
        
        # Filter expired resources
        active_resources = [r for r in resources if not r.is_expired]
        
        # Optionally prefer GPU
        if prefer_gpu:
            gpu_resources = [r for r in active_resources if r.device_type == "gpu"]
            cpu_resources = [r for r in active_resources if r.device_type == "cpu"]
            
            # Use GPU resources first, fill gaps with CPU
            prioritized = gpu_resources + cpu_resources
        else:
            prioritized = active_resources
        
        # Assemble pipeline ensuring no gaps
        pipeline = self._assemble_continuous_pipeline(prioritized, max_layers)
        
        if len(pipeline) < max_layers:
            logger.warning(
                f"Incomplete pipeline: {len(pipeline)}/{max_layers} layers. "
                f"Model may not function correctly."
            )
        
        return pipeline
    
    def _assemble_continuous_pipeline(
        self, 
        resources: List[ResourceAdvertisement],
        max_layers: int
    ) -> List[ExpertInfo]:
        """
        Assemble a continuous pipeline from resources.
        
        Ensures there are no gaps in the layer sequence.
        """
        if not resources:
            return []
        
        # Group by peer to minimize cross-peer communication
        peer_resources = defaultdict(list)
        for resource in resources:
            peer_resources[resource.peer_id].append(resource)
        
        pipeline = []
        current_layer = 0
        
        while current_layer < max_layers:
            # Find resource that covers current_layer
            found = False
            
            for peer_id, peer_res_list in peer_resources.items():
                for resource in peer_res_list:
                    if (resource.layer_start <= current_layer <= resource.layer_end 
                        and not resource.is_expired):
                        
                        # Create ExpertInfo for this layer chunk
                        expert_info = self._create_expert_info(resource)
                        pipeline.append(expert_info)
                        
                        current_layer = resource.layer_end + 1
                        found = True
                        break
                
                if found:
                    break
            
            if not found:
                # Gap in pipeline
                logger.warning(f"Gap in pipeline at layer {current_layer}")
                break
        
        return pipeline
    
    def _create_expert_info(self, resource: ResourceAdvertisement) -> ExpertInfo:
        """Create ExpertInfo from a resource advertisement"""
        expert_uid = f"{self.model_name}.layer.{resource.layer_start}.{resource.layer_end}"
        peer_id = PeerID.from_base58(resource.peer_id)
        return ExpertInfo(uid=expert_uid, peer_id=peer_id)
    
    async def get_available_models(self) -> List[str]:
        """Get list of models that have compute resources available"""
        # This would query DHT for all compute:* keys
        # For now, return cached model names
        return list(self.registry._cache.keys())
    
    def get_coverage_report(self) -> Dict[str, Dict]:
        """Get coverage report for all known models"""
        report = {}
        for model_name in self.registry._cache:
            report[model_name] = self.registry.get_layer_coverage(model_name)
        return report
