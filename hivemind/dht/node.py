from typing import Optional
from kademlia.storage import ForgetfulStorage as Storage
from .protocol import KademliaProtocol
from .routing import DHTID


class DHTNode:
    """
    A low-level class that represents DHT participant.
    Each DHTNode has an identifier, a local storage and access too other nodes via KademliaProtocol.
    Note: unlike Kademlia, nodes in a Hivemind DHT is optimized to store temporary metadata that is regularly updated
     For example, an expert alive timestamp that emitted by the Server responsible for that expert.
     Such metadata does not require maintenance such as ensuring at least k hosts have it or (de)serialization in case
     of node shutdown. Instead, DHTNode is designed to reduce the latency of looking up such data.
    """

    def __init__(self, node_id: Optional[DHTID] = None, storage: Optional[Storage] = None, beam_size=20):
        self.id = node_id or DHTID.generate()
        self.storage = storage or Storage()
        self.protocol = KademliaProtocol(self, bucket_size=beam_size)

# changelog:
# * less files (7 -> 4). mostly because chunks of code were replaced with python system libraries
# * fixed a minor bug that resulted in wrong find_neighbors if k is greater than bucket size
# * acyclic class dependencies: RoutingTable no longer calls KademliaProtocol, KademliaProtocol no longer calls DHTNode
# * DHT node id is now just an integer with helper methods
# * typing and docstrings for (hopefully) all public classes and methods
# * updated to newer asyncio: ensure_future -> create_task, etc
# * our implementaton no longer supports saving/loading a persistent state
# * our implementaton no longer supports pylint
# * large-scale tests for routing (10^4 nodes)

# TODO bmuller's kademlia updated node's bucket:
# * on every rpc_find_node - for the node that is searched for
# * on every welcome_if_new - for the new node
# * on every refresh table - for lonely_buckets
# * on save_state - for bootstrappable neighbors, some reason
# * on server.get/set/set_digest - for a bucket that contains key

# implementation differences
# (1) if a new node is rejected by RoutingTable (bucket is full and could NOT be split),
#   bmuller's implementation would ping the first node in that bucket to *maybe* remove it if it doesn't respond
#   our implementation
# (2)