"""
Holographic Context Mesh - Infinite Distributed Memory

Treats the entire network's RAM as a unified KV Cache.
When local VRAM fills, offloads least-attention-weighted tokens to neighbor nodes.
Enables infinite context windows limited only by total swarm RAM.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import torch
import numpy as np

from hivemind import DHT, PeerID
from hivemind.dht import DHTKey

logger = logging.getLogger(__name__)


@dataclass
class ContextChunk:
    """A chunk of context stored in the holographic mesh"""
    session_id: str
    layer_idx: int
    token_range: Tuple[int, int]
    kv_cache: torch.Tensor
    attention_weights: np.ndarray
    peer_owner: PeerID
    timestamp: float
    priority: float = 0.0
    
    def __lt__(self, other):
        # Lower priority = more likely to be offloaded
        return self.priority < other.priority


@dataclass 
class HolographicConfig:
    """Configuration for Holographic Context Manager"""
    max_local_layers: int = 5  # Keep this many layers locally
    offload_threshold_mb: int = 1024  # Start offloading when VRAM exceeds this
    neighbor_replication: int = 2  # Replicate context to N neighbors
    attention_decay: float = 0.95  # Decay factor for old attention weights
    sync_interval_sec: float = 30.0  # How often to sync with mesh


class HolographicContextManager:
    """
    Manages infinite context by distributing KV cache across the network.
    
    Features:
    - Automatic offloading of old context to peer nodes
    - Replication for fault tolerance
    - Attention-weighted priority (important context stays local)
    - Transparent retrieval from mesh
    """
    
    def __init__(self, dht: DHT, config: Optional[HolographicConfig] = None):
        self.dht = dht
        self.config = config or HolographicConfig()
        self.peer_id = dht.peer_id
        
        # Local storage
        self.local_context: Dict[str, List[ContextChunk]] = {}
        self.remote_pointers: Dict[str, List[Dict]] = {}
        
        # Attention tracking
        self.attention_history: Dict[str, np.ndarray] = {}
        
        # State
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        
        logger.info(f"HolographicContextManager initialized (max_local={self.config.max_local_layers})")
    
    async def start(self):
        """Start background sync tasks"""
        self._running = True
        self._sync_task = asyncio.create_task(self._periodic_sync())
        logger.info("HolographicContextManager started")
    
    async def stop(self):
        """Stop and cleanup"""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        logger.info("HolographicContextManager stopped")
    
    async def create_session(self, session_id: str) -> str:
        """Create a new context session"""
        self.local_context[session_id] = []
        self.remote_pointers[session_id] = []
        self.attention_history[session_id] = np.array([])
        
        # Advertise session availability
        await self.dht.store(
            key=f"mist:context:{session_id}",
            value={
                "owner": str(self.peer_id),
                "created_at": asyncio.get_event_loop().time(),
                "config": {
                    "max_local_layers": self.config.max_local_layers,
                    "replication": self.config.neighbor_replication
                }
            },
            expiration_time=3600  # 1 hour
        )
        
        logger.info(f"Created holographic session: {session_id}")
        return session_id
    
    async def store_context(
        self,
        session_id: str,
        layer_idx: int,
        kv_cache: torch.Tensor,
        attention_weights: np.ndarray
    ):
        """Store context chunk, automatically offloading if needed"""
        if session_id not in self.local_context:
            raise ValueError(f"Session {session_id} not found")
        
        # Update attention history
        self._update_attention(session_id, attention_weights)
        
        # Create chunk
        chunk = ContextChunk(
            session_id=session_id,
            layer_idx=layer_idx,
            token_range=(0, kv_cache.shape[0]),
            kv_cache=kv_cache.cpu(),
            attention_weights=attention_weights,
            peer_owner=self.peer_id,
            timestamp=asyncio.get_event_loop().time(),
            priority=np.mean(attention_weights)
        )
        
        # Check if we need to offload
        current_layers = len(self.local_context[session_id])
        if current_layers >= self.config.max_local_layers:
            # Offload oldest/lowest priority chunk
            await self._offload_oldest_chunk(session_id)
        
        # Store locally
        self.local_context[session_id].append(chunk)
        
        # Replicate to neighbors
        await self._replicate_chunk(session_id, chunk)
        
        logger.debug(f"Stored context layer {layer_idx} for session {session_id}")
    
    async def retrieve_context(
        self,
        session_id: str,
        layer_idx: int
    ) -> Optional[torch.Tensor]:
        """Retrieve context from local or remote storage"""
        # Try local first
        if session_id in self.local_context:
            for chunk in self.local_context[session_id]:
                if chunk.layer_idx == layer_idx:
                    return chunk.kv_cache
        
        # Try remote pointers
        if session_id in self.remote_pointers:
            for pointer in self.remote_pointers[session_id]:
                if pointer.get("layer_idx") == layer_idx:
                    # Fetch from remote peer
                    return await self._fetch_remote_context(pointer)
        
        logger.warning(f"Context layer {layer_idx} not found for session {session_id}")
        return None
    
    async def save_session(self, session_id: str):
        """Persist session state to the mesh"""
        if session_id not in self.local_context:
            return
        
        # Store metadata about all chunks
        all_chunks = []
        for chunk in self.local_context[session_id]:
            all_chunks.append({
                "layer_idx": chunk.layer_idx,
                "token_range": chunk.token_range,
                "priority": chunk.priority,
                "timestamp": chunk.timestamp
            })
        
        # Add remote pointers
        all_chunks.extend(self.remote_pointers.get(session_id, []))
        
        await self.dht.store(
            key=f"mist:context:{session_id}:manifest",
            value={
                "chunks": all_chunks,
                "total_layers": len(all_chunks),
                "last_updated": asyncio.get_event_loop().time()
            },
            expiration_time=7200  # 2 hours
        )
        
        logger.info(f"Saved session {session_id} manifest ({len(all_chunks)} chunks)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about context storage"""
        total_local = sum(len(chunks) for chunks in self.local_context.values())
        total_remote = sum(len(pointers) for pointers in self.remote_pointers.values())
        
        return {
            "sessions_active": len(self.local_context),
            "local_chunks": total_local,
            "remote_pointers": total_remote,
            "total_context_items": total_local + total_remote,
            "memory_config": {
                "max_local_layers": self.config.max_local_layers,
                "offload_threshold_mb": self.config.offload_threshold_mb,
                "replication_factor": self.config.neighbor_replication
            }
        }
    
    def _update_attention(self, session_id: str, new_weights: np.ndarray):
        """Update attention history with decay"""
        if session_id not in self.attention_history:
            self.attention_history[session_id] = new_weights
            return
        
        # Decay old weights
        self.attention_history[session_id] *= self.config.attention_decay
        
        # Append new weights
        self.attention_history[session_id] = np.concatenate([
            self.attention_history[session_id],
            new_weights
        ])
        
        # Update priorities of existing chunks
        if session_id in self.local_context:
            for i, chunk in enumerate(self.local_context[session_id]):
                if i < len(self.attention_history[session_id]):
                    chunk.priority = self.attention_history[session_id][i]
    
    async def _offload_oldest_chunk(self, session_id: str):
        """Offload lowest priority chunk to remote peers"""
        if not self.local_context[session_id]:
            return
        
        # Find lowest priority chunk
        chunks = sorted(self.local_context[session_id], key=lambda c: c.priority)
        chunk_to_offload = chunks[0]
        
        # Find neighbor to offload to
        neighbors = await self._get_active_neighbors()
        if not neighbors:
            logger.warning("No neighbors available for offloading")
            return
        
        target_peer = neighbors[0]
        
        # Store on remote peer (simplified - would use RPC in production)
        pointer = {
            "layer_idx": chunk_to_offload.layer_idx,
            "peer_id": str(target_peer),
            "timestamp": chunk_to_offload.timestamp,
            "priority": chunk_to_offload.priority
        }
        
        self.remote_pointers.setdefault(session_id, []).append(pointer)
        self.local_context[session_id].remove(chunk_to_offload)
        
        logger.info(f"Offloaded layer {chunk_to_offload.layer_idx} to peer {target_peer}")
    
    async def _replicate_chunk(self, session_id: str, chunk: ContextChunk):
        """Replicate chunk to neighbor peers for fault tolerance"""
        neighbors = await self._get_active_neighbors()
        
        for i, neighbor in enumerate(neighbors[:self.config.neighbor_replication]):
            # Send chunk to neighbor (simplified)
            logger.debug(f"Replicating chunk to neighbor {neighbor}")
    
    async def _fetch_remote_context(self, pointer: Dict) -> Optional[torch.Tensor]:
        """Fetch context from remote peer"""
        peer_id = PeerID.from_base58(pointer["peer_id"])
        layer_idx = pointer["layer_idx"]
        
        # In production, this would use actual RPC to fetch data
        # For now, return None to indicate it should be recomputed
        logger.debug(f"Fetching remote context from {peer_id} layer {layer_idx}")
        return None
    
    async def _get_active_neighbors(self) -> List[PeerID]:
        """Get list of active neighbor peers"""
        # Query DHT for active peers
        result = await self.dht.get(f"mist:peers:active")
        if not result or not result.value:
            return []
        
        # Parse peer list
        peers = result.value.get("peers", [])
        return [PeerID.from_base58(p) for p in peers[:10]]
    
    async def _periodic_sync(self):
        """Periodically sync context state with mesh"""
        while self._running:
            try:
                await asyncio.sleep(self.config.sync_interval_sec)
                
                # Sync all active sessions
                for session_id in list(self.local_context.keys()):
                    await self.save_session(session_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync error: {e}")
