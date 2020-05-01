import asyncio
import time
from itertools import chain
from functools import partial
from random import random
from typing import Optional, Tuple, List, Dict
from warnings import warn

from .protocol import KademliaProtocol
from .routing import DHTID, DHTValue, DHTExpiration
from ..utils import find_open_port, Endpoint, Port


class DHTNode:
    """
    A low-level class that represents a DHT participant.
    Each DHTNode has an identifier, a local storage and access too other nodes via KademliaProtocol.

    :param node_id: current node's identifier, determines which keys it will store locally, defaults to random id
    :param port: port to which this DHTNode will listen, by default find some open port
    :param initial_peers: connects to these peers to populate routing table, defaults to no peers
    :param bucket_size: (k) - max number of nodes in one k-bucket. Trying to add {k+1}st node will cause a bucket to
      either split in two buckets along the midpoint or reject the new node (but still save it as a replacement)
      Recommended value: $k$ is chosen s.t. any given k nodes are very unlikely to all fail after staleness_timeout
    :param beam_size: (alpha) - the number of concurrent requests when performing beam search (find node/value)
    :param modulo: (b) - kademlia can split bucket if it contains root OR up to the nearest multiple of :depth_modulo:
    :param staleness_timeout: a bucket is considered stale if no node from that bucket was updated for this many seconds
    :param wait_timeout: a kademlia rpc request is deemed lost if we did not recieve a reply in this many seconds
    :param bootstrap_timeout: after one of peers responds, await other peers for at most this many seconds

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

    def __init__(self, node_id: Optional[DHTID] = None, port: Port=None, initial_peers: Tuple[Endpoint, ...] = (),
                 bucket_size=20, beam_size=3, modulo=5, staleness_timeout=600, wait_timeout=5, bootstrap_timeout=None,
                 loop=None):
        self.id = node_id = node_id or DHTID.generate()
        self.port = port = port or find_open_port()
        self.beam_size, self.staleness_timeout = beam_size, staleness_timeout

        loop = loop if loop is not None else asyncio.get_event_loop()
        make_protocol = partial(KademliaProtocol, self.id, bucket_size, modulo, wait_timeout)
        self.transport, self.protocol = loop.create_datagram_endpoint(make_protocol, local_addr=('127.0.0.1', port))

        # bootstrap part 1: ping initial_peers, add each other to the routing table
        bootstrap_timeout = bootstrap_timeout if bootstrap_timeout is not None else wait_timeout
        began_bootstrap_time = time.monotonic()
        ping_tasks = map(self.protocol.call_ping, initial_peers)
        first_finished, remaining_tasks = loop.run_until_complete(
            asyncio.wait(ping_tasks, timeout=wait_timeout, return_when=asyncio.FIRST_COMPLETED))
        time_to_first_response = time.monotonic() - began_bootstrap_time

        # bootstrap part 2: gather all peers who responded within bootstrap_timeout, but at least one peer
        finished_in_time, stragglers = loop.run_until_complete(
            asyncio.wait(remaining_tasks, timeout=bootstrap_timeout - time_to_first_response, loop=loop))
        for straggler in stragglers:
            straggler.cancel()

        peer_ids = [task.result() for task in chain(first_finished, finished_in_time) if task.result() is not None]
        if len(peer_ids) == 0 and len(initial_peers) != 0:
            warn("DHTNode bootstrap failed: none of the initial_peers responded to a ping.")

        # bootstrap part 3: run beam search for my node id to add my own nearest neighbors to the routing table
        self.find_nearest_nodes(query_id=self.id, initial_beam=peer_ids)

    def find_nearest_nodes(self, query_id: DHTID, initial_beam: Optional[List[DHTID]] = None) -> Dict[DHTID, Endpoint]:
        """ TODO """
        raise NotImplementedError()

    def get(self, key: DHTID, sufficient_time: DHTExpiration = -float('inf')) -> \
            Tuple[Optional[DHTValue], Optional[DHTExpiration]]:
        """
        :param key: traverse the DHT and find the value for this key (or None if it does not exist)
        :param sufficient_time: if the search finds a value that expires after sufficient_time, it can return this
         value right away. By default, return the newest value found after beam search converges.
        :returns: value and its expiration time. If found nothing, returns (None, None)
        """
        raise NotImplementedError()

    def set(self, key: DHTID, value: DHTValue, expiration_time: DHTExpiration) -> bool:
        """
        Find beam_size best nodes to store (key, value) and store it there at least until expiration time.
        Also cache (key, value, expiration_time) at all nodes you met along the way (see Section 2.1 end)
        TODO: if we found a newer value in the in the table, terminate immediately and throw a warning
        """
        raise NotImplementedError()

    def refresh_table(self):
        staleness_threshold = time.monotonic() - self.staleness_timeout
        stale_buckets = [bucket for bucket in self.buckets if bucket.last_updated < staleness_threshold]
        staleness_ids = [DHTID(random.randint(bucket.lower, bucket.upper)) for bucket in stale_buckets]

        raise NotImplementedError("TODO")

# TODO bmuller's kademlia updated node's bucket:
# * on every rpc_find_node - for the node that is searched for
# * on every welcome_if_new - for the new node
# * on every refresh table - for lonely_buckets
# * on save_state - for bootstrappable neighbors, some reason
# * on server.get/set/set_digest - for a bucket that contains key

# debt:
# * make sure we ping least-recently-updated node in full bucket if someone else wants to replace him
#   this should happen every time we add new node
