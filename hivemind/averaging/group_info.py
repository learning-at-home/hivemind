from dataclasses import dataclass
from typing import Tuple

from hivemind.p2p import PeerID


@dataclass(frozen=True)
class GroupInfo:
    """A group of peers assembled through decentralized matchmaking"""

    group_id: bytes  # random unique bytestring that describes the current group, generated by group leader
    peer_ids: Tuple[PeerID, ...]  # an ordered sequence of peer_ids of each groupmate
    gathered: Tuple[bytes, ...]  # binary metadata gathered from all peers by leader, same order as peer_ids

    @property
    def group_size(self):
        return len(self.peer_ids)

    def __contains__(self, peer_id: PeerID):
        return peer_id in self.peer_ids
