"""
Geo-Topological Sharding Manager
Solves the "Speed of Light" latency wall by clustering compute geographically.
Routes requests to the nearest "Super-Shard" instead of random global nodes.
"""
import asyncio
import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class GeoNode:
    node_id: str
    latitude: float
    longitude: float
    latency_ms: float
    bandwidth_mbps: float
    load_factor: float  # 0.0 (idle) to 1.0 (full)
    specs: dict

@dataclass
class GeoShard:
    shard_id: str
    region_name: str
    center_lat: float
    center_lon: float
    nodes: List[GeoNode]
    total_flops: float
    avg_latency_internal: float

class GeoShardManager:
    """
    Manages geographic clustering and intelligent routing.
    """
    def __init__(self):
        self.shards: Dict[str, GeoShard] = {}
        self.node_to_shard: Dict[str, str] = {}
        
        # Pre-defined major compute regions
        self.regions = {
            "NA-EAST": (40.7128, -74.0060),   # New York
            "NA-WEST": (37.7749, -122.4194),  # San Francisco
            "EU-CENTRAL": (52.5200, 13.4050), # Berlin
            "ASIA-EAST": (35.6762, 139.6503), # Tokyo
            "ASIA-SOUTH": (1.3521, 103.8198), # Singapore
            "SA-EAST": (-23.5505, -46.6333),  # Sao Paulo
        }

    def haversine_distance(self, lat1, lon1, lat2, lon2) -> float:
        """Calculate distance between two points in km."""
        R = 6371  # Earth's radius in km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)
        a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def assign_node_to_shard(self, node: GeoNode) -> str:
        """Assign a node to the nearest geographic shard."""
        min_dist = float('inf')
        assigned_shard = None
        
        for region_name, (center_lat, center_lon) in self.regions.items():
            dist = self.haversine_distance(node.latitude, node.longitude, center_lat, center_lon)
            if dist < min_dist:
                min_dist = dist
                assigned_shard = region_name
        
        if assigned_shard:
            if assigned_shard not in self.shards:
                self.shards[assigned_shard] = GeoShard(
                    shard_id=assigned_shard,
                    region_name=assigned_shard,
                    center_lat=self.regions[assigned_shard][0],
                    center_lon=self.regions[assigned_shard][1],
                    nodes=[],
                    total_flops=0.0,
                    avg_latency_internal=10.0 # Default 10ms internal
                )
            
            shard = self.shards[assigned_shard]
            shard.nodes.append(node)
            shard.total_flops += node.specs.get('flops', 1e12)
            self.node_to_shard[node.node_id] = assigned_shard
            
            return assigned_shard
        return "UNKNOWN"

    def get_optimal_shard_for_user(self, user_lat: float, user_lon: float) -> Optional[GeoShard]:
        """Find the best shard for a user based on proximity and load."""
        best_shard = None
        best_score = float('inf')
        
        for shard in self.shards.values():
            dist = self.haversine_distance(user_lat, user_lon, shard.center_lat, shard.center_lon)
            # Estimate network latency (approx 1ms per 100km + internal processing)
            estimated_latency = (dist / 100) + 10 
            
            # Score = Latency * (1 + LoadFactor)
            # Penalize heavily loaded shards
            avg_load = sum(n.load_factor for n in shard.nodes) / len(shard.nodes) if shard.nodes else 0
            score = estimated_latency * (1 + avg_load)
            
            if score < best_score:
                best_score = score
                best_shard = shard
        
        return best_shard

    def route_request(self, user_location: Tuple[float, float], required_flops: float) -> List[GeoNode]:
        """
        Route a request to the optimal set of nodes within a shard.
        Returns a list of nodes that can handle the workload.
        """
        user_lat, user_lon = user_location
        target_shard = self.get_optimal_shard_for_user(user_lat, user_lon)
        
        if not target_shard:
            raise Exception("No available shards nearby")
        
        # Select nodes within the shard based on load and speed
        available_nodes = [n for n in target_shard.nodes if n.load_factor < 0.9]
        available_nodes.sort(key=lambda n: n.load_factor) # Prefer idle nodes
        
        # Simple round-robin or load-balanced selection
        # In production: bin-packing algorithm
        selected_nodes = available_nodes[:max(1, int(required_flops / 1e12))] # 1 node per TFLOP approx
        
        if not selected_nodes:
            # Fallback: Expand to neighboring shards (cross-shard routing)
            print(f"Warning: Shard {target_shard.shard_id} overloaded. Expanding search...")
            return self._expand_search(user_location, required_flops)
            
        return selected_nodes

    def _expand_search(self, user_location: Tuple[float, float], required_flops: float) -> List[GeoNode]:
        """Fallback mechanism to use neighboring shards if local shard is full."""
        # Sort all shards by distance to user
        sorted_shards = sorted(
            self.shards.values(),
            key=lambda s: self.haversine_distance(user_location[0], user_location[1], s.center_lat, s.center_lon)
        )
        
        selected = []
        remaining_flops = required_flops
        
        for shard in sorted_shards:
            if remaining_flops <= 0:
                break
            
            # Take available capacity from this shard
            for node in shard.nodes:
                if node.load_factor < 0.8:
                    selected.append(node)
                    remaining_flops -= node.specs.get('flops', 1e12)
                    
        return selected

    def update_node_stats(self, node_id: str, latency: float, load: float):
        """Real-time update of node performance metrics."""
        if node_id in self.node_to_shard:
            shard_id = self.node_to_shard[node_id]
            shard = self.shards[shard_id]
            for node in shard.nodes:
                if node.node_id == node_id:
                    node.latency_ms = latency
                    node.load_factor = load
                    break

# --- Usage Example ---
if __name__ == "__main__":
    manager = GeoShardManager()
    
    # Simulate nodes joining from around the world
    nodes_data = [
        ("node_us_1", 40.7, -74.0, 1e15),   # NY
        ("node_us_2", 37.7, -122.4, 4e12),  # SF
        ("node_eu_1", 52.5, 13.4, 8e12),    # Berlin
        ("node_jp_1", 35.6, 139.6, 1e15),   # Tokyo
        ("node_sg_1", 1.35, 103.8, 2e12),   # Singapore
    ]
    
    for nid, lat, lon, flops in nodes_data:
        node = GeoNode(
            node_id=nid,
            latitude=lat,
            longitude=lon,
            latency_ms=20.0,
            bandwidth_mbps=100.0,
            load_factor=0.2,
            specs={"flops": flops, "type": "GPU"}
        )
        shard = manager.assign_node_to_shard(node)
        print(f"Assigned {nid} to shard {shard}")
    
    # User in London requests inference
    user_loc = (51.5074, -0.1278)
    required_compute = 5e12 # 5 TFLOPs
    
    selected_nodes = manager.route_request(user_loc, required_compute)
    print(f"\nUser in London routed to {len(selected_nodes)} nodes:")
    for n in selected_nodes:
        print(f"  - {n.node_id} ({n.specs['flops']/1e12:.1f} TFLOPs)")
        
    print("\n✅ Geo-Sharding Manager Ready. Latency optimized by region.")
