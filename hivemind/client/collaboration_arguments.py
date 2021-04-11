from typing import Optional
from dataclasses import dataclass


@dataclass
class CollaborationArguments:
    initial_peers: str  # one or more peers (comma-separated) that will welcome you into the collaboration
    dht_key_for_averaging: str  # a unique identifier of this experimental run's metadata
    averaging_expiration: float = 5.0  # averaging group will expire after this many seconds
    averaging_step_timeout: float = 30.0  # give up averaging step after this many seconds
    metadata_expiration: float = 30  # peer's metadata will be removed if not updated in this many seconds
    statistics_expiration: float = 1000  # statistics will be removed if not updated in this many seconds
    target_group_size: int = 64      # maximum group size for all-reduce
    target_batch_size: int = 4096  # perform optimizer step after all peers collectively accumulate this many samples
    dht_listen_on: str = '[::]:*'  # network interface used for incoming DHT communication. Default: all ipv6
    listen_on: str = '[::]:*'  # network interface used for incoming averager communication. Default: all ipv6
    endpoint: Optional[str] = None  # this node's public IP, used when the node is behind a proxy
    client_mode: bool = False  # if True, runs training without incoming connections, in a firewall-compatible mode

    min_refresh_period: float = 0.5  # wait for at least this many seconds before fetching new collaboration state
    max_refresh_period: float = 30  # wait for at most this many seconds before fetching new collaboration state
    default_refresh_period: float = 3  # attempt to fetch collaboration state every this often until successful
    expected_collaboration_drift_peers: float = 3  # trainer assumes that this many new peers can join per step
    expected_collaboration_drift_rate = 0.2  # trainer assumes that this fraction of current size can join per step

    bandwidth: float = 1000.0  # available network bandwidth, in mbps (used for load balancing in all-reduce)
    performance_ema_alpha: float = 0.1  # uses this alpha for moving average estimate of samples per second
    trainer_uuid: Optional[str] = None  # this trainer's identifier - used when publishing metadata to DHT
