"""
Contribution Tracking System

Tracks GPU/CPU contributions from participants and calculates token rewards
based on the percentage of compute resources contributed to model inference.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from collections import defaultdict

from hivemind.dht import DHT
from hivemind.utils import get_dht_time, get_logger
from hivemind.moe.expert_uid import ExpertUID

logger = get_logger(__name__)


@dataclass
class ComputeContribution:
    """Records compute contribution from a peer"""
    peer_id: str
    expert_uid: ExpertUID
    layer_start: int
    layer_end: int
    compute_time: float  # seconds spent computing
    tokens_processed: int
    timestamp: float = field(default_factory=get_dht_time)
    device_type: str = "gpu"  # gpu or cpu
    device_specs: Optional[Dict] = None
    
    @property
    def compute_units(self) -> float:
        """
        Calculate compute units based on time and device type.
        GPU compute is weighted higher than CPU.
        """
        weight = 2.0 if self.device_type == "gpu" else 1.0
        return self.compute_time * weight


@dataclass
class SessionContribution:
    """Tracks all contributions for a single inference session"""
    session_id: str
    model_name: str
    total_tokens: int
    contributions: Dict[str, ComputeContribution] = field(default_factory=dict)
    start_time: float = field(default_factory=get_dht_time)
    completed: bool = False
    
    def add_contribution(self, contribution: ComputeContribution):
        """Add a contribution record"""
        self.contributions[contribution.peer_id] = contribution
        
    def get_total_compute_units(self) -> float:
        """Get total compute units from all contributors"""
        return sum(c.compute_units for c in self.contributions.values())
    
    def calculate_rewards(self) -> Dict[str, float]:
        """
        Calculate token reward percentage for each contributor.
        Returns dict mapping peer_id -> percentage of tokens earned (0.0 to 1.0)
        """
        if not self.contributions:
            return {}
            
        total_units = self.get_total_compute_units()
        if total_units == 0:
            return {}
            
        rewards = {}
        for peer_id, contribution in self.contributions.items():
            rewards[peer_id] = contribution.compute_units / total_units
            
        return rewards


class ContributionTracker:
    """
    Tracks compute contributions across the network using DHT.
    
    This enables fair distribution of generated tokens based on 
    actual compute resources contributed to running large models.
    """
    
    def __init__(self, dht: DHT, update_period: float = 10.0):
        self.dht = dht
        self.update_period = update_period
        self._local_contributions: Dict[str, List[ComputeContribution]] = defaultdict(list)
        self._active_sessions: Dict[str, SessionContribution] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        
    def start_session(self, session_id: str, model_name: str, total_expected_tokens: int) -> SessionContribution:
        """Start tracking a new inference session"""
        session = SessionContribution(
            session_id=session_id,
            model_name=model_name,
            total_tokens=total_expected_tokens
        )
        with self._lock:
            self._active_sessions[session_id] = session
        logger.info(f"Started tracking session {session_id} for model {model_name}")
        return session
    
    def record_contribution(
        self,
        session_id: str,
        expert_uid: ExpertUID,
        layer_range: tuple,
        compute_time: float,
        tokens_processed: int,
        device_type: str = "gpu",
        device_specs: Optional[Dict] = None
    ):
        """
        Record a compute contribution from this peer.
        
        :param session_id: ID of the inference session
        :param expert_uid: Expert UID that was executed
        :param layer_range: Tuple of (layer_start, layer_end) this peer handled
        :param compute_time: Time spent computing in seconds
        :param tokens_processed: Number of tokens processed in this contribution
        :param device_type: 'gpu' or 'cpu'
        :param device_specs: Optional dict with device specifications
        """
        contribution = ComputeContribution(
            peer_id=self.dht.peer_id.to_base58(),
            expert_uid=expert_uid,
            layer_start=layer_range[0],
            layer_end=layer_range[1],
            compute_time=compute_time,
            tokens_processed=tokens_processed,
            device_type=device_type,
            device_specs=device_specs
        )
        
        with self._lock:
            self._local_contributions[self.dht.peer_id.to_base58()].append(contribution)
            
            if session_id in self._active_sessions:
                self._active_sessions[session_id].add_contribution(contribution)
        
        # Publish to DHT for global visibility
        self._publish_contribution_to_dht(session_id, contribution)
        
        logger.debug(
            f"Recorded contribution: {compute_time:.3f}s on {device_type} "
            f"for layers {layer_range}, {tokens_processed} tokens"
        )
    
    def _publish_contribution_to_dht(self, session_id: str, contribution: ComputeContribution):
        """Publish contribution record to DHT for transparency"""
        key = f"contribution:{session_id}:{contribution.peer_id}"
        value = {
            "expert_uid": contribution.expert_uid,
            "layer_start": contribution.layer_start,
            "layer_end": contribution.layer_end,
            "compute_time": contribution.compute_time,
            "tokens_processed": contribution.tokens_processed,
            "device_type": contribution.device_type,
            "timestamp": contribution.timestamp
        }
        expiration = get_dht_time() + 3600  # Keep for 1 hour
        self.dht.store(key, value, expiration_time=expiration)
    
    def get_session_rewards(self, session_id: str) -> Optional[Dict[str, float]]:
        """
        Get calculated rewards for a completed session.
        Returns dict mapping peer_id -> token percentage (0.0 to 1.0)
        """
        with self._lock:
            if session_id not in self._active_sessions:
                return None
            session = self._active_sessions[session_id]
            
        return session.calculate_rewards()
    
    def finalize_session(self, session_id: str, actual_tokens: int):
        """Mark a session as completed and calculate final rewards"""
        with self._lock:
            if session_id not in self._active_sessions:
                logger.warning(f"Session {session_id} not found")
                return None
                
            session = self._active_sessions[session_id]
            session.total_tokens = actual_tokens
            session.completed = True
            
        rewards = session.calculate_rewards()
        logger.info(
            f"Session {session_id} finalized: {actual_tokens} tokens, "
            f"{len(session.contributions)} contributors"
        )
        return rewards
    
    def get_my_total_contributions(self) -> Dict[str, float]:
        """Get total compute units contributed by this peer across all sessions"""
        peer_id = self.dht.peer_id.to_base58()
        with self._lock:
            contributions = self._local_contributions.get(peer_id, [])
        
        # Group by model
        model_contributions = defaultdict(float)
        for contrib in contributions:
            # Extract model name from expert_uid (assumes format: model.layer.index)
            model_name = contrib.expert_uid.split('.')[0] if '.' in contrib.expert_uid else "unknown"
            model_contributions[model_name] += contrib.compute_units
            
        return dict(model_contributions)
    
    def run_in_background(self):
        """Start background thread for periodic DHT updates"""
        thread = threading.Thread(target=self._background_loop, daemon=True)
        thread.start()
        return thread
    
    def _background_loop(self):
        """Background loop for maintenance tasks"""
        while not self._stop.is_set():
            try:
                # Clean up old sessions
                current_time = get_dht_time()
                with self._lock:
                    expired_sessions = [
                        sid for sid, session in self._active_sessions.items()
                        if current_time - session.start_time > 7200  # 2 hours
                    ]
                    for sid in expired_sessions:
                        del self._active_sessions[sid]
                
                self._stop.wait(self.update_period)
            except Exception as e:
                logger.error(f"Error in contribution tracker background loop: {e}")
    
    def shutdown(self):
        """Stop the tracker"""
        self._stop.set()


class TokenRewardCalculator:
    """
    Calculates and distributes token rewards to contributors.
    
    This class can be integrated with blockchain or credit systems
    to actually distribute rewards based on contribution percentages.
    """
    
    def __init__(self, contribution_tracker: ContributionTracker):
        self.tracker = contribution_tracker
        self._reward_history: List[Dict] = []
    
    def distribute_rewards(self, session_id: str, total_token_value: float) -> Dict[str, float]:
        """
        Distribute monetary/token rewards based on contribution.
        
        :param session_id: ID of the completed session
        :param total_token_value: Total value (in credits/tokens) to distribute
        :returns: Dict mapping peer_id -> reward amount
        """
        rewards_percentage = self.tracker.get_session_rewards(session_id)
        if not rewards_percentage:
            logger.warning(f"No rewards to distribute for session {session_id}")
            return {}
        
        actual_rewards = {}
        for peer_id, percentage in rewards_percentage.items():
            reward_amount = total_token_value * percentage
            actual_rewards[peer_id] = reward_amount
            
        self._reward_history.append({
            "session_id": session_id,
            "total_value": total_token_value,
            "distribution": actual_rewards,
            "timestamp": get_dht_time()
        })
        
        logger.info(
            f"Distributed {total_token_value} tokens across "
            f"{len(actual_rewards)} contributors"
        )
        
        return actual_rewards
    
    def get_reward_history(self, peer_id: Optional[str] = None) -> List[Dict]:
        """Get reward history, optionally filtered by peer_id"""
        if peer_id is None:
            return self._reward_history.copy()
        
        return [
            record for record in self._reward_history
            if peer_id in record.get("distribution", {})
        ]
