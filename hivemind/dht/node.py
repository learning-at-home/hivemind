import asyncio
import heapq
import time
from collections import OrderedDict
from functools import partial
from random import random
from typing import Optional, Tuple, List, Dict
from warnings import warn

from .protocol import KademliaProtocol
from .routing import DHTID, DHTValue, DHTExpiration
from ..utils import find_open_port, Endpoint, Hostname, Port, LOCALHOST


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
    :param num_replicas: (≈k) - number of nearest nodes that will be asked to store a given key, default = bucket_size
    :param depth_modulo: (b) - kademlia can split bucket if it contains root OR up to the nearest multiple of this value
    :param beam_size: (≈alpha) - beam search will not give up until it exhausts this many best nodes from the heap
    :param wait_timeout: a kademlia rpc request is deemed lost if we did not recieve a reply in this many seconds
    :param staleness_timeout: a bucket is considered stale if no node from that bucket was updated in this many seconds
    :param bootstrap_timeout: after one of peers responds, await other peers for at most this many seconds
    :param interface: provide 0.0.0.0 to operate over ipv4, :: to operate over ipv6, localhost to operate locally, etc.

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

    def __init__(self, node_id: Optional[DHTID] = None, port: Optional[Port] = None, initial_peers: List[Endpoint] = (),
                 bucket_size: int = 20, num_replicas: Optional[int] = None, depth_modulo: int = 5,
                 beam_size: Optional[int] = None, wait_timeout: float = 5, staleness_timeout: Optional[float] = 600,
                 bootstrap_timeout: Optional[float] = None, interface: Hostname = '0.0.0.0', loop=None):
        self.node_id = node_id = node_id if node_id is not None else DHTID.generate()
        self.port = port = port if port is not None else find_open_port()
        self.num_replicas = num_replicas if num_replicas is not None else bucket_size
        self.beam_size = beam_size if beam_size is not None else bucket_size
        self.staleness_timeout = staleness_timeout

        # create kademlia protocol and make it listen to a port
        loop = loop if loop is not None else asyncio.get_event_loop()
        make_protocol = partial(KademliaProtocol, self.node_id, bucket_size, depth_modulo, wait_timeout)
        listener = loop.run_until_complete(loop.create_datagram_endpoint(make_protocol, local_addr=(interface, port)))
        self.transport: asyncio.Transport = listener[0]
        self.protocol: KademliaProtocol = listener[1]

        if initial_peers:
            # bootstrap part 1: ping initial_peers, add each other to the routing table
            bootstrap_timeout = bootstrap_timeout if bootstrap_timeout is not None else wait_timeout
            began_bootstrap_time = time.monotonic()
            ping_tasks = map(self.protocol.call_ping, initial_peers)
            finished_tasks, remaining_tasks = loop.run_until_complete(
                asyncio.wait(ping_tasks, timeout=wait_timeout, return_when=asyncio.FIRST_COMPLETED))
            time_to_first_response = time.monotonic() - began_bootstrap_time
            # bootstrap part 2: gather all peers who responded within bootstrap_timeout, but at least one peer
            if remaining_tasks:
                finished_in_time, stragglers = loop.run_until_complete(
                    asyncio.wait(remaining_tasks, timeout=bootstrap_timeout - time_to_first_response, loop=loop))
                for straggler in stragglers:
                    straggler.cancel()
                finished_tasks |= finished_in_time

            peer_ids = [task.result() for task in finished_tasks if task.result() is not None]
            if len(peer_ids) == 0 and len(initial_peers) != 0:
                warn("DHTNode bootstrap failed: none of the initial_peers responded to a ping.")

            # bootstrap part 3: run beam search for my node id to add my own nearest neighbors to the routing table
            # ... and maybe receive some values that we are meant to store (see protocol.update_routing_table)
            asyncio.ensure_future(self.beam_search(query_id=self.node_id), loop=loop)

    async def get(self, key: DHTID, sufficient_time: DHTExpiration = float('inf')) -> \
            Tuple[Optional[DHTValue], Optional[DHTExpiration]]:
        """
        :param key: traverse the DHT and find the value for this key (or None if it does not exist)
        :param sufficient_time: if the search finds a value that expires after sufficient_time, it can return this
         value right away. By default, return the newest value found after beam search converges.
        :returns: value and its expiration time. If found nothing, returns (None, None)
        """
        beam_search_results = await self.beam_search(key, k_nearest=self.protocol.bucket_size)
        tasks = [self.protocol.call_find_value(endpoint, key)
                 for endpoint in beam_search_results.values()]
        latest_time, latest_value = None, None
        for task in asyncio.as_completed(tasks):
            value, expiration_time, _ = await task
            if expiration_time is None:
                continue
            if latest_time is None or expiration_time > latest_time:
                latest_time, latest_value = expiration_time, value
                if latest_time > sufficient_time:
                    return latest_time, latest_value
        return latest_time, latest_value

    async def store(self, key: DHTID, value: DHTValue, expiration_time: DHTExpiration) -> bool:
        """
        Find beam_size best nodes to store (key, value) and store it there at least until expiration time.
        Also cache (key, value, expiration_time) at all nodes you met along the way (see Section 2.1 end)
        TODO: if we found a newer value in the in the table, terminate immediately and throw a warning
        """
        beam_search_results = await self.beam_search(key, k_nearest=self.num_replicas)
        tasks = [self.protocol.call_store(endpoint, key, value, expiration_time)
                 for endpoint in beam_search_results.values()]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        return any(done)

    async def refresh_stale_buckets(self):
        staleness_threshold = time.monotonic() - self.staleness_timeout
        stale_buckets = [bucket for bucket in self.protocol.routing_table.buckets
                         if bucket.last_updated < staleness_threshold]

        refresh_ids = [DHTID(random.randint(bucket.lower, bucket.upper - 1)) for bucket in stale_buckets]
        # note: we use bucket.upper - 1 because random.randint is inclusive w.r.t. both lower and upper bounds

        raise NotImplementedError("TODO")

    async def beam_search(self, query_id: DHTID, initial_peers: Optional[Dict[DHTID, Endpoint]] = None,
                          k_nearest: Optional[int] = None, beam_size: Optional[int] = None,
                          max_hops: Optional[int] = None, exclude_self: bool = False) -> Dict[DHTID, Endpoint]:
        """ you see a lengthy description of how beam search works """

        # infer defaults
        if beam_size is not None and k_nearest is not None and k_nearest > beam_size:
            warn(f"beam search: beam_size({beam_size}) is too small, setting it equal to k_nearest({k_nearest}).")
        k_nearest = k_nearest if k_nearest is not None else self.protocol.bucket_size
        beam_size = max(self.beam_size, (k_nearest if beam_size is None else beam_size))
        max_hops = float('inf') if max_hops is None else max_hops
        initial_peers = initial_peers or dict(self.protocol.routing_table.get_nearest_neighbors(query_id, k=beam_size))

        # initialize beam search
        ids_to_endpoint = dict(initial_peers)  # all nodes visited by this beam search
        ids_to_endpoint[self.node_id] = (LOCALHOST, self.port)  # add self
        ids_to_distance = dict(zip(initial_peers, query_id.xor_distance(initial_peers)))

        candidates = [(distance, peer_id) for peer_id, distance in
                      ids_to_distance.items() if peer_id != self.node_id]  # unvisited nodes, nearest-first heap
        heapq.heapify(candidates)
        top_results = [(-distance, peer) for distance, peer in
                       heapq.nsmallest(beam_size, candidates)]  # fathest-first heap, at most beam_size elements
        heapq.heapify(top_results)
        if not exclude_self:
            heapq.heappush(top_results, (-query_id.xor_distance(self.node_id), self.node_id))
            while len(top_results) > beam_size:
                heapq.heappop(top_results)

        if len(top_results) == 0:
            return {}

        lower_bound = -heapq.nsmallest(1, top_results)[0][0]
        num_hops = 0

        while candidates:
            dist, node_id = heapq.heappop(candidates)
            if dist > lower_bound:
                break

            neighbor_ids_to_endpoint = await self.protocol.call_find_node(ids_to_endpoint[node_id], query_id)

            # only consider neighbors that have not been visited before
            neighbor_ids_to_endpoint = {node_id: endpoint for node_id, endpoint in neighbor_ids_to_endpoint.items()
                                        if node_id not in ids_to_endpoint}

            neighbor_ids = list(neighbor_ids_to_endpoint.keys())
            if not neighbor_ids:
                continue

            neighbor_ids_to_distance = dict(zip(neighbor_ids, query_id.xor_distance(neighbor_ids)))

            for neighbor_id, distance in neighbor_ids_to_distance.items():
                if distance < lower_bound or len(top_results) < beam_size:
                    heapq.heappush(candidates, (distance, neighbor_id))

                    heapq_add_or_replace = heapq.heappush if len(top_results) < beam_size else heapq.heappushpop
                    heapq_add_or_replace(top_results, (-distance, neighbor_id))

                    lower_bound = -heapq.nsmallest(1, top_results)[0][0]

            ids_to_distance.update(neighbor_ids_to_distance)
            ids_to_endpoint.update(neighbor_ids_to_endpoint)

            num_hops += 1
            if num_hops >= max_hops:
                break

        return OrderedDict((node_id, ids_to_endpoint[node_id]) for _, node_id in heapq.nlargest(k_nearest, top_results))

# TODO bmuller's kademlia updated node's bucket:
# * on every rpc_find_node - for the node that is searched for
# * on every welcome_if_new - for the new node
# * on every refresh table - for lonely_buckets
# * on save_state - for bootstrappable neighbors, some reason
# * on server.get/set/set_digest - for a bucket that contains key

# debt:
# * make sure we ping least-recently-updated node in full bucket if someone else wants to replace him
#   this should happen every time we add new node
