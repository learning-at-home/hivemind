from typing import Optional, Tuple
from kademlia.storage import ForgetfulStorage as Storage
from .protocol import KademliaProtocol
from .routing import DHTID, DHTValue, DHTExpirationTime


class DHTNode:
    """
    A low-level class that represents DHT participant.
    Each DHTNode has an identifier, a local storage and access too other nodes via KademliaProtocol.

    Note: Hivemind DHT is optimized to store temporary metadata that is regularly updated.
     For example, an expert alive timestamp that emitted by the Server responsible for that expert.
     Such metadata does not require maintenance such as ensuring at least k hosts have it or (de)serialization in case
     of node shutdown. Instead, DHTNode is designed to reduce the latency of looking up such data.

    Every (key, value) pair in this DHT has expiration_time - float number computed wth time.monotonic().
    Informally, dht nodes always prefer values with higher expiration_time and may delete any value past its expiration.

    Formally, DHTNode follows this contract:
      - when asked to store(key, value, expiration_time), a node must store (key, value) at least until expiration_time
       unless it already stores that key with greater or equal expiration_time - if so, node must keep the previous key
      - when asked to get(key), a node must return the value with highest expiration time IF that time has not come yet
       if expiration time is greater than current time.monotonic(), DHTNode *may* return None
    """

    def __init__(self, node_id: Optional[DHTID] = None, storage: Optional[Storage] = None, beam_size=20):
        self.id = node_id or DHTID.generate()
        self.storage = storage or Storage()
        self.protocol = KademliaProtocol(self, bucket_size=beam_size)

    def store(self, key: DHTID, value: DHTValue, expiration_time: DHTExpirationTime) -> bool:
        """
        Store a (key, value) pair for storage until expiration_time. See class docstring for details.
        :returns: True if new value was stored, False it was rejected (current value is newer)
        """
        assert isinstance(expiration_time, DHTExpirationTime)  # actually, just float
        raise NotImplementedError()

    def get(self, key: DHTID) -> Tuple[Optional[DHTValue], Optional[DHTExpirationTime]]:
        raise NotImplementedError()


# TODO bmuller's kademlia updated node's bucket:
# * on every rpc_find_node - for the node that is searched for
# * on every welcome_if_new - for the new node
# * on every refresh table - for lonely_buckets
# * on save_state - for bootstrappable neighbors, some reason
# * on server.get/set/set_digest - for a bucket that contains key
