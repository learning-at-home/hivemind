from typing import Optional, Tuple
from .protocol import KademliaProtocol
from .routing import DHTID, DHTValue, DHTExpirationTime


class DHTNode:
    """
    A low-level class that represents DHT participant.
    Each DHTNode has an identifier, a local storage and access too other nodes via KademliaProtocol.

    :param bucket_size: (k) - max number of nodes in one k-bucket. Trying to add {k+1}st node will cause a bucket to
      either split in two buckets along the midpoint or reject the new node (but still save it as a replacement)
      Recommended value: $k$ is chosen s.t. any given k nodes are very unlikely to all fail after staleness_timeout
    :param beam_size: (alpha) - the number of concurrent requests when performing beam search (find node/value)
    :param modulo: (b) - kademlia can split bucket if it contains root OR up to the nearest multiple of :depth_modulo:
    :param staleness_timeout: a bucket is considered stale if no node from that bucket was updated for this many seconds


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

    def __init__(self, node_id: Optional[DHTID] = None, bucket_size=20, beam_size=3, modulo=5, staleness_timeout=600):
        self.id = node_id or DHTID.generate()
        self.beam_size = beam_size
        self.protocol = KademliaProtocol(self, bucket_size=bucket_size, modulo=modulo,
                                         staleness_timeout=staleness_timeout)

    def get(self, key: DHTID, sufficient_time: DHTExpirationTime = -float('inf')) -> \
            Tuple[Optional[DHTValue], Optional[DHTExpirationTime]]:
        """
        :param key: traverse the DHT and find the value for this key
        :param sufficient_time: if the search finds a value that expires after sufficient_time, it can return this
         value right away. By default, return the newest value found after beam search converges.
        :returns: value and its expiration time. If found nothing, returns (None, None)
        """
        raise NotImplementedError()

    def set(self, key: DHTID, value: DHTValue, expiration_time: DHTExpirationTime) -> bool:
        """
        Find beam_size best nodes to store (key, value) and store it there at least until expiration time.
        Also cache (key, value, expiration_time) at all nodes you met along the way (see Section 2.1 end)
        """
        raise NotImplementedError()

    def store(self, key: DHTID, value: DHTValue, expiration_time: DHTExpirationTime) -> bool:
        """
        Store a (key, value) pair locally at least until expiration_time. See class docstring for details.
        :returns: True if new value was stored, False it was rejected (current value is newer)
        """
        raise NotImplementedError()


# TODO bmuller's kademlia updated node's bucket:
# * on every rpc_find_node - for the node that is searched for
# * on every welcome_if_new - for the new node
# * on every refresh table - for lonely_buckets
# * on save_state - for bootstrappable neighbors, some reason
# * on server.get/set/set_digest - for a bucket that contains key
