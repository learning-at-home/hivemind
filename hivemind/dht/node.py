import asyncio
import time
from collections import OrderedDict
from functools import partial
from random import random
from typing import Optional, Tuple, List, Dict
from warnings import warn

from .protocol import KademliaProtocol
from .routing import DHTID, DHTValue, DHTExpiration, DHTKey
from .search import beam_search
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
    :param wait_timeout: a kademlia rpc request is deemed lost if we did not recieve a reply in this many seconds
    :param staleness_timeout: a bucket is considered stale if no node from that bucket was updated in this many seconds
    :param bootstrap_timeout: after one of peers responds, await other peers for at most this many seconds
    :param interface: provide 0.0.0.0 to operate over ipv4, :: to operate over ipv6, localhost to operate locally, etc.

    :note: Hivemind DHT is optimized to store temporary metadata that is regularly updated.
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
                 wait_timeout: float = 5, staleness_timeout: Optional[float] = 600,
                 bootstrap_timeout: Optional[float] = None, cache_locally: bool = True, cache_nearest: int = 1,
                 interface: Hostname = '0.0.0.0'):
        self.node_id = node_id = node_id if node_id is not None else DHTID.generate()
        self.port = port = port if port is not None else find_open_port()
        self.num_replicas = num_replicas if num_replicas is not None else bucket_size
        self.cache_locally, self.cache_nearest = cache_locally, cache_nearest
        self.staleness_timeout = staleness_timeout

        # create kademlia protocol and make it listen to a port
        loop = asyncio.get_event_loop()
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
                    asyncio.wait(remaining_tasks, timeout=bootstrap_timeout - time_to_first_response))
                for straggler in stragglers:
                    straggler.cancel()
                finished_tasks |= finished_in_time

            peer_ids = [task.result() for task in finished_tasks if task.result() is not None]
            if len(peer_ids) == 0 and len(initial_peers) != 0:
                warn("DHTNode bootstrap failed: none of the initial_peers responded to a ping.")

            # bootstrap part 3: run beam search for my node id to add my own nearest neighbors to the routing table
            # ... and maybe receive some values that we are meant to store (see protocol.update_routing_table)
            loop.run_until_complete(self.find_nearest_nodes(query_id=self.node_id))

    async def find_nearest_nodes(self, query_id: DHTID, k_nearest: Optional[int] = None,
                                 beam_size: Optional[int] = None, exclude_self: bool = False) -> Dict[DHTID, Endpoint]:
        """
        Traverse the DHT and find :k_nearest: nodes to a given :query_id:, optionally :exclude_self: from the results.
        :note: this is a thin wrapper over dht.search.beam_search, look there for more details
        :returns: an ordered dictionary of [peer DHTID -> network Endpoint], ordered from nearest to farthest neighbor
        """
        k_nearest = k_nearest if k_nearest is not None else self.protocol.bucket_size
        beam_size = beam_size if beam_size is not None else max(self.protocol.bucket_size, k_nearest)
        node_to_addr = dict(
            self.protocol.routing_table.get_nearest_neighbors(query_id, beam_size, exclude=self.node_id))

        async def get_neighbors(node: DHTID) -> Tuple[List[DHTID], bool]:
            peers: Dict[DHTID, Endpoint] = await self.protocol.call_find_node(node_to_addr[node], query_id)
            node_to_addr.update(peers)
            return list(peers.keys()), False  # False means "do not interrupt beam search"

        nearest_nodes, visited_nodes = await beam_search(
            query_id=query_id, initial_nodes=list(node_to_addr), k_nearest=k_nearest, beam_size=beam_size,
            get_neighbors=get_neighbors, visited_nodes=(self.node_id,))

        if not exclude_self:
            nearest_nodes = sorted(nearest_nodes + [self.node_id], key=query_id.xor_distance)[:k_nearest]
            node_to_addr[self.node_id] = (LOCALHOST, self.port)

        return OrderedDict((node, node_to_addr[node]) for node in nearest_nodes)

    async def store(self, key: DHTKey, value: DHTValue, expiration_time: DHTExpiration) -> bool:
        """
        Find beam_size best nodes to store (key, value) and store it there at least until expiration time.
        Also cache (key, value, expiration_time) at all nodes you met along the way (see Section 2.1 end)
        :note: if store finds a newer value in the table, it will propagate this newer value instead of the original
        :return: True if store succeeds, False if it fails (due to no response or newer value)
        """
        key_id = DHTID.generate(key)
        nearest_node_to_addr = await self.find_nearest_nodes(key_id, k_nearest=self.num_replicas, exclude_self=True)
        tasks = [asyncio.create_task(self.protocol.call_store(endpoint, key_id, value, expiration_time))
                 for endpoint in nearest_node_to_addr.values()]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        return any(done)

    async def get(self, key: DHTKey, sufficient_expiration_time: Optional[DHTExpiration] = None,
                  beam_size: Optional[int] = None) -> Tuple[Optional[DHTValue], Optional[DHTExpiration]]:
        """
        :param key: traverse the DHT and find the value for this key (or return None if it does not exist)
        :param sufficient_expiration_time: if the search finds a value that expires after this time,
            default = time of call, find any value that did not expire by the time of call
            If min_expiration_time=float('inf'), this method will find a value with _latest_ expiration
        :returns: value and its expiration time. If nothing is found , returns (None, None).
        :note: in order to check if get returned a value, please check (expiration_time is None)
        """
        key_id = DHTID.generate(key)
        sufficient_expiration_time = sufficient_expiration_time or time.monotonic()
        beam_size = beam_size if beam_size is not None else self.protocol.bucket_size
        latest_value, latest_expiration = None, -float('inf')
        node_to_addr, nodes_checked_for_value = dict(), set()

        # Option A: value can be stored in our local cache
        maybe_value, maybe_expiration = self.protocol.storage.get(key_id)
        if maybe_expiration is not None and maybe_expiration > latest_expiration:
            latest_value, latest_expiration = maybe_value, maybe_expiration
            # TODO(jheuristic) we may want to run background beam search to update our cache
        nodes_checked_for_value.add(self.node_id)

        # Option B: go beam search the DHT
        if latest_expiration < sufficient_expiration_time:
            node_to_addr.update(self.protocol.routing_table.get_nearest_neighbors(
                key_id, self.protocol.bucket_size, exclude=self.node_id))

            async def get_neighbors(node: DHTID) -> Tuple[List[DHTID], bool]:
                nonlocal latest_value, latest_expiration, node_to_addr, nodes_checked_for_value
                maybe_value, maybe_expiration, peers = await self.protocol.call_find_value(node_to_addr[node], key_id)
                nodes_checked_for_value.add(node)
                node_to_addr.update(peers)
                if maybe_expiration is not None and maybe_expiration > latest_expiration:
                    latest_value, latest_expiration = maybe_value, maybe_expiration
                should_interrupt = (latest_expiration >= sufficient_expiration_time)
                return list(peers.keys()), should_interrupt

            nearest_nodes, visited_nodes = await beam_search(
                query_id=key_id, initial_nodes=list(node_to_addr), k_nearest=beam_size, beam_size=beam_size,
                get_neighbors=get_neighbors, visited_nodes=nodes_checked_for_value)
            # normally, by this point we will have found a sufficiently recent value in one of get_neighbors calls

        # Option C: didn't find good-enough value in beam search, make a last-ditch effort to find it in unvisited nodes
        if latest_expiration < sufficient_expiration_time:
            nearest_unvisited = [node_id for node_id in nearest_nodes if node_id not in nodes_checked_for_value]
            tasks = [self.protocol.call_find_value(node_to_addr[node_id], key_id) for node_id in nearest_unvisited]
            pending_tasks = set(tasks)
            for task in asyncio.as_completed(tasks):
                pending_tasks.remove(task)
                maybe_value, maybe_expiration, _ = await task
                if maybe_expiration is not None and maybe_expiration > latest_expiration:
                    latest_value, latest_expiration = maybe_value, maybe_expiration
                    if latest_expiration >= sufficient_expiration_time:
                        break
            for task in pending_tasks:
                task.close()

        # step 4: we have not found entry with sufficient_expiration_time, but we may have found *something* older
        # TODO(jheuristic) cache here once vsevolodpl is done with the storage - both to self and to nearest
        return (latest_value, latest_expiration) if latest_expiration != -float('inf') else (None, None)

    async def refresh_stale_buckets(self):
        staleness_threshold = time.monotonic() - self.staleness_timeout
        stale_buckets = [bucket for bucket in self.protocol.routing_table.buckets
                         if bucket.last_updated < staleness_threshold]

        refresh_ids = [DHTID(random.randint(bucket.lower, bucket.upper - 1)) for bucket in stale_buckets]
        # note: we use bucket.upper - 1 because random.randint is inclusive w.r.t. both lower and upper bounds

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
