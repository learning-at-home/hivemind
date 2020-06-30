from __future__ import annotations
import asyncio
import random
from collections import OrderedDict
from typing import Optional, Tuple, List, Dict
from warnings import warn

from .protocol import DHTProtocol
from .routing import DHTID, BinaryDHTValue, DHTExpiration, DHTKey, get_dht_time, DHTValue
from .search import traverse_dht
from ..utils import Endpoint, LOCALHOST, MSGPackSerializer


class DHTNode:
    """
    A low-level class that represents a DHT participant. Please see DHTNode.create for parameters
    Each DHTNode has an identifier, a local storage and access too other nodes via DHTProtocol.

    :note: Hivemind DHT is optimized to store a lot of temporary metadata that is regularly updated.
     For example, an expert alive timestamp that emitted by the Server responsible for that expert.
     Such metadata does not require regular maintenance by peers, persistence on shutdown.
     Instead, DHTNode is designed to rapidly send bulk data and resolve conflicts.

    Every (key, value) pair in this DHT has an expiration time - float computed as get_dht_time(), UnixTime by default
    DHT nodes always prefer values with higher expiration time and may delete any value past its expiration.

    Compared to Kademlia RPC protocol, hivemind DHT has 3 RPCs:

    * ping - request peer's identifier and update routing table (same as Kademlia PING RPC)
    * store - send several (key, value, expiration) pairs to the same peer (like Kademlia STORE, but in bulk)
    * find - request one or several keys, get values & expiration (if peer finds it locally) and :bucket_size: of
        nearest peers from recipient's routing table (ordered nearest-to-farthest, not including recipient itself)
        This RPC is a mixture between Kademlia FIND_NODE and FIND_VALUE with multiple keys per call.

    Formally, DHTNode follows the following contract:

    - when asked to get(key), a node must find and return a value with highest expiration time that it found across DHT
      IF that time has not come yet. if expiration time is smaller than current get_dht_time(), node may return None;
    - when requested to store(key: value, expiration), a node must store (key => value) at until expiration time
      or until DHTNode gets the same key with greater expiration time. If a node is asked to store a key but it already
      has the same key with newer expiration, the older key will not be stored. Return True if stored, False if refused;
    - when requested to store(key: value, expiration, in_cache=True), stores (key => value) in a separate "cache".
      Cache operates same as regular storage, but it has a limited size and evicts least recently used nodes when full;

    """
    node_id: int; port: int; num_replicas: int; cache_locally: bool; cache_nearest: int; refresh_timeout: float
    protocol: DHTProtocol
    serializer = MSGPackSerializer  # used to pack/unpack DHT Values for transfer over network

    @classmethod
    async def create(
            cls, node_id: Optional[DHTID] = None, initial_peers: List[Endpoint] = (),
            bucket_size: int = 20, num_replicas: Optional[int] = None, depth_modulo: int = 5, parallel_rpc: int = None,
            wait_timeout: float = 5, refresh_timeout: Optional[float] = None, bootstrap_timeout: Optional[float] = None,
            cache_locally: bool = True, cache_nearest: int = 1, cache_size=None,
            listen: bool = True, listen_on: Endpoint = "0.0.0.0:*", **kwargs) -> DHTNode:
        """
        :param node_id: current node's identifier, determines which keys it will store locally, defaults to random id
        :param initial_peers: connects to these peers to populate routing table, defaults to no peers
        :param bucket_size: max number of nodes in one k-bucket (k). Trying to add {k+1}st node will cause a bucket to
          either split in two buckets along the midpoint or reject the new node (but still save it as a replacement)
          Recommended value: k is chosen s.t. any given k nodes are very unlikely to all fail after staleness_timeout
        :param num_replicas: number of nearest nodes that will be asked to store a given key, default = bucket_size (≈k)
        :param depth_modulo: split full k-bucket if it contains root OR up to the nearest multiple of this value (≈b)
        :param parallel_rpc: maximum number of concurrent outgoing RPC requests emitted by DHTProtocol
          Reduce this value if your RPC requests register no response despite the peer sending the response.
        :param wait_timeout: a kademlia rpc request is deemed lost if we did not recieve a reply in this many seconds
        :param refresh_timeout: refresh buckets if no node from that bucket was updated in this many seconds
          if staleness_timeout is None, DHTNode will not refresh stale buckets (which is usually okay)
        :param bootstrap_timeout: after one of peers responds, await other peers for at most this many seconds
        :param cache_locally: if True, caches all values (stored or found) in a node-local cache
        :param cache_nearest: whenever DHTNode finds a value, it will also store (cache) this value on this many
          nodes nearest nodes visited by search algorithm. Prefers nodes that are nearest to :key: but have no value yet
        :param cache_size: if specified, local cache will store up to this many records (as in LRU cache)
        :param listen: if True (default), this node will accept incoming request and otherwise be a DHT "citzen"
          if False, this node will refuse any incoming request, effectively being only a "client"
        :param listen_on: network interface for incoming RPCs, e.g. "0.0.0.0:1337" or "localhost:\*" or "[::]:7654"
        :param channel_options: options for grpc.aio.insecure_channel, e.g. [('grpc.enable_retries', 0)]
          see https://grpc.github.io/grpc/core/group__grpc__arg__keys.html for a list of all options
        :param kwargs: extra parameters used in grpc.aio.server
        """
        self = cls(_initialized_with_create=True)
        self.node_id = node_id = node_id if node_id is not None else DHTID.generate()
        self.num_replicas = num_replicas if num_replicas is not None else bucket_size
        self.cache_locally, self.cache_nearest = cache_locally, cache_nearest
        self.refresh_timeout = refresh_timeout

        self.protocol = await DHTProtocol.create(self.node_id, bucket_size, depth_modulo, num_replicas, wait_timeout,
                                                 parallel_rpc, cache_size, listen, listen_on, **kwargs)
        self.port = self.protocol.port


        if initial_peers:
            # stage 1: ping initial_peers, add each other to the routing table
            bootstrap_timeout = bootstrap_timeout if bootstrap_timeout is not None else wait_timeout
            start_time = get_dht_time()
            ping_tasks = map(self.protocol.call_ping, initial_peers)
            finished_pings, unfinished_pings = await asyncio.wait(ping_tasks, return_when=asyncio.FIRST_COMPLETED)

            # stage 2: gather remaining peers (those who respond within bootstrap_timeout)
            if unfinished_pings:
                finished_in_time, stragglers = await asyncio.wait(
                    unfinished_pings, timeout=bootstrap_timeout - get_dht_time() + start_time)
                for straggler in stragglers:
                    straggler.cancel()
                finished_pings |= finished_in_time

            if not finished_pings:
                warn("DHTNode bootstrap failed: none of the initial_peers responded to a ping.")

            # stage 3: traverse dht to find my own nearest neighbors and populate the routing table
            # ... maybe receive some values that we are meant to store (see protocol.update_routing_table)
            # note: using asyncio.wait instead of wait_for because wait_for cancels task on timeout
            await asyncio.wait([asyncio.create_task(self.find_nearest_nodes(key_id=self.node_id)),
                                asyncio.sleep(bootstrap_timeout - get_dht_time() + start_time)],
                               return_when=asyncio.FIRST_COMPLETED)

        if self.refresh_timeout is not None:
            asyncio.create_task(self._refresh_routing_table(period=self.refresh_timeout))
        return self

    def __init__(self, *, _initialized_with_create=False):
        """ Internal init method. Please use DHTNode.create coroutine to spawn new node instances """
        assert _initialized_with_create, " Please use DHTNode.create coroutine to spawn new node instances "
        super().__init__()

    async def shutdown(self, timeout=None):
        """ Process existing requests, close all connections and stop the server """
        await self.protocol.shutdown(timeout)

    async def find_nearest_nodes(self, key_id: DHTID, k_nearest: Optional[int] = None,
                                 beam_size: Optional[int] = None, exclude_self: bool = False) -> Dict[DHTID, Endpoint]:
        """
        Traverse the DHT and find :k_nearest: nodes to a given :query_id:, optionally :exclude_self: from the results.

        :returns: an ordered dictionary of [peer DHTID -> network Endpoint], ordered from nearest to farthest neighbor
        :note: this is a thin wrapper over dht.search.traverse_dht, look there for more details
        """
        k_nearest = k_nearest if k_nearest is not None else self.protocol.bucket_size
        beam_size = beam_size if beam_size is not None else max(self.protocol.bucket_size, k_nearest)
        node_to_addr = dict(
            self.protocol.routing_table.get_nearest_neighbors(key_id, beam_size, exclude=self.node_id))

        async def get_neighbors(node_id: DHTID) -> Tuple[List[DHTID], bool]:
            response = await self.protocol.call_find(node_to_addr[node_id], [key_id])
            if not response or key_id not in response:
                return [], False  # False means "do not interrupt search"

            peers: Dict[DHTID, Endpoint] = response[key_id][-1]
            node_to_addr.update(peers)
            return list(peers.keys()), False  # False means "do not interrupt search"

        nearest_nodes, visited_nodes = await traverse_dht(
            query_id=key_id, initial_nodes=list(node_to_addr), k_nearest=k_nearest, beam_size=beam_size,
            get_neighbors=get_neighbors, visited_nodes=(self.node_id,))

        if not exclude_self:
            nearest_nodes = sorted(nearest_nodes + [self.node_id], key=key_id.xor_distance)[:k_nearest]
            node_to_addr[self.node_id] = (LOCALHOST, self.port)

        return OrderedDict((node, node_to_addr[node]) for node in nearest_nodes)

    async def store(self, key: DHTKey, value: BinaryDHTValue, expiration_time: DHTExpiration) -> bool:
        """
        Find beam_size best nodes to store (key, value) and store it there at least until expiration time.
        Optionally cache (key, value, expiration) on nodes you met along the way (see Section 2.1 end) TODO(jheuristic)

        :returns: True if store succeeds, False if it fails (due to no response or newer value)
        """
        key_id, value_bytes = DHTID.generate(source=key), self.serializer.dumps(value)
        nearest_node_to_addr = await self.find_nearest_nodes(key_id, k_nearest=self.num_replicas, exclude_self=True)
        tasks = [asyncio.create_task(self.protocol.call_store(endpoint, [key_id], [value_bytes], [expiration_time]))
                 for endpoint in nearest_node_to_addr.values()]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        return any(store_ok for response in done for store_ok in response.result())

    async def get(self, key: DHTKey, sufficient_expiration_time: Optional[DHTExpiration] = None,
                  beam_size: Optional[int] = None) -> Tuple[Optional[DHTValue], Optional[DHTExpiration]]:
        """
        :param key: traverse the DHT and find the value for this key (or return None if it does not exist)
        :param sufficient_expiration_time: if the search finds a value that expires after this time,
            default = time of call, find any value that did not expire by the time of call
            If min_expiration_time=float('inf'), this method will find a value with _latest_ expiration
        :param beam_size: maintains up to this many nearest nodes when crawling dht, default beam_size = bucket_size
        :returns: value and its expiration time. If nothing is found , returns (None, None).
        :note: in order to check if get returned a value, please check (expiration_time is None)
        """
        key_id = DHTID.generate(key)
        sufficient_expiration_time = sufficient_expiration_time or get_dht_time()
        beam_size = beam_size if beam_size is not None else self.protocol.bucket_size
        latest_value_bytes, latest_expiration, latest_node_id = b'', -float('inf'), None
        node_to_addr, nodes_checked_for_value, nearest_nodes = dict(), set(), []
        should_cache = False  # True if found value in DHT that is newer than local value

        # Option A: value can be stored in our local cache
        maybe_value, maybe_expiration = self.protocol.storage.get(key_id)
        if maybe_expiration is None:
            maybe_value, maybe_expiration = self.protocol.cache.get(key_id)
        if maybe_expiration is not None and maybe_expiration > latest_expiration:
            latest_value_bytes, latest_expiration, latest_node_id = maybe_value, maybe_expiration, self.node_id
            # TODO(jheuristic) we may want to run background beam search to update our cache
        nodes_checked_for_value.add(self.node_id)

        # Option B: go beam search the DHT
        if latest_expiration < sufficient_expiration_time:
            node_to_addr.update(self.protocol.routing_table.get_nearest_neighbors(
                key_id, self.protocol.bucket_size, exclude=self.node_id))

            async def get_neighbors(node: DHTID) -> Tuple[List[DHTID], bool]:
                nonlocal latest_value_bytes, latest_expiration, latest_node_id, node_to_addr, nodes_checked_for_value
                response = await self.protocol.call_find(node_to_addr[node], [key_id])
                nodes_checked_for_value.add(node)
                if not response or key_id not in response:
                    return [], False

                maybe_value, maybe_expiration, peers = response[key_id]
                node_to_addr.update(peers)
                if maybe_expiration is not None and maybe_expiration > latest_expiration:
                    latest_value_bytes, latest_expiration, latest_node_id = maybe_value, maybe_expiration, node
                should_interrupt = (latest_expiration >= sufficient_expiration_time)
                return list(peers.keys()), should_interrupt

            nearest_nodes, visited_nodes = await traverse_dht(
                query_id=key_id, initial_nodes=list(node_to_addr), k_nearest=beam_size, beam_size=beam_size,
                get_neighbors=get_neighbors, visited_nodes=nodes_checked_for_value)
            # normally, by this point we will have found a sufficiently recent value in one of get_neighbors calls
            should_cache = latest_expiration >= sufficient_expiration_time  # if we found a newer value, cache it later

        # Option C: didn't find good-enough value in beam search, make a last-ditch effort to find it in unvisited nodes
        if latest_expiration < sufficient_expiration_time:
            nearest_unvisited = [node_id for node_id in nearest_nodes if node_id not in nodes_checked_for_value]
            tasks = [self.protocol.call_find(node_to_addr[node_id], [key_id]) for node_id in nearest_unvisited]
            pending_tasks = set(tasks)
            for task in asyncio.as_completed(tasks):
                pending_tasks.remove(task)
                if not task.result() or key_id not in task.result():
                    maybe_value, maybe_expiration, _ = task.result()[key_id]
                if maybe_expiration is not None and maybe_expiration > latest_expiration:
                    latest_value_bytes, latest_expiration = maybe_value, maybe_expiration
                    if latest_expiration >= sufficient_expiration_time:
                        break
            for task in pending_tasks:
                task.close()
            should_cache = latest_expiration >= sufficient_expiration_time  # if we found a newer value, cache it later

        # step 4: we have not found entry with sufficient_expiration_time, but we may have found *something* older
        if should_cache and self.cache_locally:
            self.protocol.cache.store(key_id, latest_value_bytes, latest_expiration)
        if should_cache and self.cache_nearest:
            num_cached_nodes = 0
            for node_id in nearest_nodes:
                if node_id == latest_node_id:
                    continue
                asyncio.create_task(self.protocol.call_store(
                    node_to_addr[node_id], [key_id], [latest_value_bytes], [latest_expiration], in_cache=True))
                num_cached_nodes += 1
                if num_cached_nodes >= self.cache_nearest:
                    break
        if latest_expiration != -float('inf'):
            return self.serializer.loads(latest_value_bytes), latest_expiration
        else:
            return None, None

    async def _refresh_routing_table(self, *, period: Optional[float]) -> None:
        """ Tries to find new nodes for buckets that were unused for more than self.staleness_timeout """
        while period is not None:  # if None run once, otherwise run forever
            refresh_time = get_dht_time()
            staleness_threshold = refresh_time - self.staleness_timeout
            stale_buckets = [bucket for bucket in self.protocol.routing_table.buckets
                             if bucket.last_updated < staleness_threshold]
            for bucket in stale_buckets:
                refresh_id = DHTID(random.randint(bucket.lower, bucket.upper - 1))
                await self.find_nearest_nodes(refresh_id)

            await asyncio.sleep(max(0.0, period - (get_dht_time() - refresh_time)))
