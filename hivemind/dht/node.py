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


# TODO bmuller's kademlia updated node's bucket:
# * on every rpc_find_node - for the node that is searched for
# * on every welcome_if_new - for the new node
# * on every refresh table - for lonely_buckets
# * on save_state - for bootstrappable neighbors, some reason
# * on server.get/set/set_digest - for a bucket that contains key
