from __future__ import annotations

import asyncio
import random
from collections import namedtuple
from typing import Optional, Tuple, List, Dict, Collection, Union, Set
from warnings import warn

from hivemind.dht.protocol import DHTProtocol
from hivemind.dht.routing import DHTID, DHTExpiration, DHTKey, get_dht_time, DHTValue
from hivemind.dht.traverse import traverse_dht
from hivemind.utils import Endpoint, LOCALHOST, MSGPackSerializer


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
    * store - send several (key, value, expiration_time) pairs to the same peer (like Kademlia STORE, but in bulk)
    * find - request one or several keys, get values & expiration (if peer finds it locally) and :bucket_size: of
        nearest peers from recipient's routing table (ordered nearest-to-farthest, not including recipient itself)
        This RPC is a mixture between Kademlia FIND_NODE and FIND_VALUE with multiple keys per call.

    Formally, DHTNode follows the following contract:

    - when asked to get(key), a node must find and return a value with highest expiration time that it found across DHT
      IF that time has not come yet. if expiration time is smaller than current get_dht_time(), node may return None;
    - when requested to store(key: value, expiration_time), a node must store (key => value) at until expiration time
      or until DHTNode gets the same key with greater expiration time. If a node is asked to store a key but it already
      has the same key with newer expiration, the older key will not be stored. Return True if stored, False if refused;
    - when requested to store(key: value, expiration_time, in_cache=True), stores (key => value) in a separate "cache".
      Cache operates same as regular storage, but it has a limited size and evicts least recently used nodes when full;

    """
    # fmt:off
    node_id: DHTID; port: int; num_replicas: int; cache_locally: bool; cache_nearest: int; num_workers: int
    refresh_timeout: float; protocol: DHTProtocol
    serializer = MSGPackSerializer  # used to pack/unpack DHT Values for transfer over network
    # fmt:on

    @classmethod
    async def create(
            cls, node_id: Optional[DHTID] = None, initial_peers: List[Endpoint] = (),
            bucket_size: int = 20, num_replicas: int = 5, depth_modulo: int = 5, parallel_rpc: int = None,
            wait_timeout: float = 5, refresh_timeout: Optional[float] = None, bootstrap_timeout: Optional[float] = None,
            num_workers: int = 1, cache_locally: bool = True, cache_nearest: int = 1, cache_size=None,
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
        :param wait_timeout: a kademlia rpc request is deemed lost if we did not receive a reply in this many seconds
        :param refresh_timeout: refresh buckets if no node from that bucket was updated in this many seconds
          if staleness_timeout is None, DHTNode will not refresh stale buckets (which is usually okay)
        :param bootstrap_timeout: after one of peers responds, await other peers for at most this many seconds
        :param num_workers: concurrent workers in traverse_dht (see traverse_dht num_workers param)
        :param cache_locally: if True, caches all values (stored or found) in a node-local cache
        :param cache_nearest: whenever DHTNode finds a value, it will also store (cache) this value on this many
          nodes nearest nodes visited by search algorithm. Prefers nodes that are nearest to :key: but have no value yet
        :param cache_size: if specified, local cache will store up to this many records (as in LRU cache)
        :param listen: if True (default), this node will accept incoming request and otherwise be a DHT "citzen"
          if False, this node will refuse any incoming request, effectively being only a "client"
        :param listen_on: network interface, e.g. "0.0.0.0:1337" or "localhost:*" (* means pick any port) or "[::]:7654"
        :param channel_options: options for grpc.aio.insecure_channel, e.g. [('grpc.enable_retries', 0)]
          see https://grpc.github.io/grpc/core/group__grpc__arg__keys.html for a list of all options
        :param kwargs: extra parameters used in grpc.aio.server
        """
        self = cls(_initialized_with_create=True)
        self.node_id = node_id = node_id if node_id is not None else DHTID.generate()
        self.num_replicas, self.num_workers = num_replicas, num_workers
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
            await asyncio.wait([asyncio.create_task(self.find_nearest_nodes([self.node_id])),
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

    async def find_nearest_nodes(
            self, queries: Collection[DHTID], k_nearest: Optional[int] = None, beam_size: Optional[int] = None,
            num_workers: Optional[int] = None, node_to_endpoint: Optional[Dict[DHTID, Endpoint]] = None,
            exclude_self: bool = False, **kwargs) -> Dict[DHTID, Dict[DHTID, Endpoint]]:
        """
        :param queries: find k nearest nodes for each of these DHTIDs
        :param k_nearest: return this many nearest nodes for every query (if there are enough nodes)
        :param beam_size: replacement for self.beam_size, see traverse_dht beam_size param
        :param num_workers: replacement for self.num_workers, see traverse_dht num_workers param
        :param node_to_endpoint: if specified, uses this dict[node_id => endpoint] as initial peers
        :param exclude_self: if True, nearest nodes will not contain self.node_id (default = use local peers)
        :param kwargs: additional params passed to traverse_dht
        :returns: for every query, return nearest peers ordered dict[peer DHTID -> network Endpoint], nearest-first
        """
        queries = tuple(queries)
        k_nearest = k_nearest if k_nearest is not None else self.protocol.bucket_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        beam_size = beam_size if beam_size is not None else max(self.protocol.bucket_size, k_nearest)
        if k_nearest > beam_size:
            warn("Warning: beam_size is too small, beam search is not guaranteed to find enough nodes")
        if node_to_endpoint is None:
            node_to_endpoint: Dict[DHTID, Endpoint] = dict()
            for query in queries:
                node_to_endpoint.update(
                    self.protocol.routing_table.get_nearest_neighbors(query, beam_size, exclude=self.node_id))

        async def get_neighbors(peer: DHTID, queries: Collection[DHTID]) -> Dict[DHTID, Tuple[List[DHTID], bool]]:
            response = await self.protocol.call_find(node_to_endpoint[peer], queries)
            if not response:
                return {query: ([], False) for query in queries}

            output: Dict[DHTID, Tuple[List[DHTID], bool]] = {}
            for query, (_, _, peers) in response.items():
                node_to_endpoint.update(peers)
                output[query] = list(peers.keys()), False  # False means "do not interrupt search"
            return output

        nearest_nodes_per_query, visited_nodes = await traverse_dht(
            queries, initial_nodes=list(node_to_endpoint), beam_size=beam_size, num_workers=num_workers,
            queries_per_call=int(len(queries) ** 0.5), get_neighbors=get_neighbors,
            visited_nodes={query: {self.node_id} for query in queries}, **kwargs)

        nearest_nodes_with_endpoints = {}
        for query, nearest_nodes in nearest_nodes_per_query.items():
            if not exclude_self:
                nearest_nodes = sorted(nearest_nodes + [self.node_id], key=query.xor_distance)
                node_to_endpoint[self.node_id] = f"{LOCALHOST}:{self.port}"
            nearest_nodes_with_endpoints[query] = {node: node_to_endpoint[node] for node in nearest_nodes[:k_nearest]}
        return nearest_nodes_with_endpoints

    async def store(self, key: DHTKey, value: DHTValue, expiration_time: DHTExpiration, **kwargs) -> bool:
        """
        Find num_replicas best nodes to store (key, value) and store it there at least until expiration time.

        :note: store is a simplified interface to store_many, all kwargs are be forwarded there
        :returns: True if store succeeds, False if it fails (due to no response or newer value)
        """
        store_ok = await self.store_many([key], [value], [expiration_time], **kwargs)
        return store_ok[key]

    async def store_many(self, keys: List[DHTKey], values: List[DHTValue],
                         expiration_time: Union[DHTExpiration, List[DHTExpiration]],
                         exclude_self: bool = False, await_all_replicas=True, **kwargs) -> Dict[DHTKey, bool]:
        """
        Traverse DHT to find up to best nodes to store multiple (key, value, expiration_time) pairs.

        :param keys: arbitrary serializable keys associated with each value
        :param values: serializable "payload" for each key
        :param expiration_time: either one expiration time for all keys or individual expiration times (see class doc)
        :param kwargs: any additional parameters passed to traverse_dht function (e.g. num workers)
        :param exclude_self: if True, never store value locally even if you are one of the nearest nodes
        :note: if exclude_self is True and self.cache_locally == True, value will still be __cached__ locally
        :param await_all_replicas: if False, this function returns after first store_ok and proceeds in background
            if True, the function will wait for num_replicas successful stores or running out of beam_size nodes
        :returns: for each key: True if store succeeds, False if it fails (due to no response or newer value)
        """
        if isinstance(expiration_time, DHTExpiration):
            expiration_time = [expiration_time] * len(keys)
        assert len(keys) == len(values) == len(expiration_time), "Number of keys, values and expiration doesn't match."

        key_ids = list(map(DHTID.generate, keys))
        id_to_original_key = dict(zip(key_ids, keys))
        binary_values_by_key_id = {key_id: self.serializer.dumps(value) for key_id, value in zip(key_ids, values)}
        expiration_by_key_id = {key_id: expiration_time for key_id, expiration_time in zip(key_ids, expiration_time)}
        unfinished_key_ids = set(key_ids)  # we use this set to ensure that each store request is finished

        store_ok = {key: False for key in keys}  # outputs, updated during search
        store_finished_events = {key: asyncio.Event() for key in keys}

        if self.cache_locally:
            for key_id in key_ids:
                self.protocol.cache.store(key_id, binary_values_by_key_id[key_id], expiration_by_key_id[key_id])

        # pre-populate node_to_endpoint
        node_to_endpoint: Dict[DHTID, Endpoint] = dict()
        for key_id in key_ids:
            node_to_endpoint.update(self.protocol.routing_table.get_nearest_neighbors(
                key_id, self.protocol.bucket_size, exclude=self.node_id))

        async def on_found(key_id: DHTID, nearest_nodes: List[DHTID], visited_nodes: Set[DHTID]) -> None:
            """ This will be called once per key when find_nearest_nodes is done for a particular node """
            # note: we use callbacks instead of returned values to call store immediately without waiting for stragglers
            assert key_id in unfinished_key_ids, "Internal error: traverse_dht finished the same query twice"
            assert self.node_id not in nearest_nodes
            unfinished_key_ids.remove(key_id)

            # ensure k nodes stored the value, optionally include self.node_id as a candidate
            num_successful_stores = 0
            pending_store_tasks = set()
            store_candidates = sorted(nearest_nodes + ([] if exclude_self else [self.node_id]),
                                      key=key_id.xor_distance, reverse=True)  # ordered so that .pop() returns nearest

            while num_successful_stores < self.num_replicas and (store_candidates or pending_store_tasks):
                # spawn enough tasks to cover all replicas
                while store_candidates and num_successful_stores + len(pending_store_tasks) < self.num_replicas:
                    node_id: DHTID = store_candidates.pop()  # nearest untried candidate
                    if node_id == self.node_id:
                        self.protocol.storage.store(key_id, binary_values_by_key_id[key_id],
                                                    expiration_by_key_id[key_id])
                        store_ok[id_to_original_key[key_id]] = True
                        num_successful_stores += 1
                        if not await_all_replicas:
                            store_finished_events[id_to_original_key[key_id]].set()

                    else:
                        pending_store_tasks.add(asyncio.create_task(self.protocol.call_store(
                            node_to_endpoint[node_id], [key_id], [binary_values_by_key_id[key_id]],
                            [expiration_by_key_id[key_id]])))

                # await nearest task. If it fails, dispatch more on the next iteration
                if pending_store_tasks:
                    finished_store_tasks, pending_store_tasks = await asyncio.wait(
                        pending_store_tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in finished_store_tasks:
                        if task.result()[0]:  # if store succeeded
                            store_ok[id_to_original_key[key_id]] = True
                            num_successful_stores += 1
                            if not await_all_replicas:
                                store_finished_events[id_to_original_key[key_id]].set()

            store_finished_events[id_to_original_key[key_id]].set()

        store_task = asyncio.create_task(self.find_nearest_nodes(
            queries=set(key_ids), k_nearest=self.num_replicas, node_to_endpoint=node_to_endpoint,
            found_callback=on_found, exclude_self=exclude_self, **kwargs))
        try:
            await asyncio.wait([evt.wait() for evt in store_finished_events.values()])  # wait for items to be stored
            assert len(unfinished_key_ids) == 0, "Internal error: traverse_dht didn't finish search"
            return store_ok
        except asyncio.CancelledError as e:
            store_task.cancel()
            raise e

    async def get(self, key: DHTKey, latest=False, **kwargs) -> Tuple[Optional[DHTValue], Optional[DHTExpiration]]:
        """
        Search for a key across DHT and return either first or latest entry.
        :param key: same key as in node.store(...)
        :param latest: if True, finds the latest value, otherwise finds any non-expired value (which is much faster)
        :param kwargs: parameters forwarded to get_many
        :returns: (value, expiration time); if value was not found, returns (None, None)
        """
        if latest:
            kwargs["sufficient_expiration_time"] = float('inf')
        result = await self.get_many([key])
        return result[key]

    async def get_many(
            self, keys: Collection[DHTKey], sufficient_expiration_time: Optional[DHTExpiration] = None,
            num_workers: Optional[int] = None, beam_size: Optional[int] = None
    ) -> Dict[DHTKey, Tuple[Optional[DHTValue], Optional[DHTExpiration]]]:
        """
        :param keys: traverse the DHT and find the value for each of these keys (or (None, None) if not key found)
        :param sufficient_expiration_time: if the search finds a value that expires after this time,
            default = time of call, find any value that did not expire by the time of call
            If min_expiration_time=float('inf'), this method will find a value with _latest_ expiration
        :param beam_size: maintains up to this many nearest nodes when crawling dht, default beam_size = bucket_size
        :param num_workers: override for default num_workers, see traverse_dht num_workers param
        :returns: for each key: value and its expiration time. If nothing is found , returns (None, None) for that key
        :note: in order to check if get returned a value, please check (expiration_time is None)
        """
        key_ids = [DHTID.generate(key) for key in keys]
        id_to_original_key = dict(zip(key_ids, keys))
        sufficient_expiration_time = sufficient_expiration_time or get_dht_time()
        beam_size = beam_size if beam_size is not None else self.protocol.bucket_size
        num_workers = num_workers if num_workers is not None else self.num_workers

        # search metadata
        unfinished_key_ids = set(key_ids)  # track key ids for which the search is not terminated
        node_to_endpoint: Dict[DHTID, Endpoint] = dict()  # global routing table for all queries

        SearchResult = namedtuple("SearchResult", ["binary_value", "expiration_time", "source_node_id"])
        latest_results = {key_id: SearchResult(b'', -float('inf'), None) for key_id in key_ids}

        # stage 1: value can be stored in our local cache
        for key_id in key_ids:
            maybe_value, maybe_expiration_time = self.protocol.storage.get(key_id)
            if maybe_expiration_time is None:
                maybe_value, maybe_expiration_time = self.protocol.cache.get(key_id)
            if maybe_expiration_time is not None and maybe_expiration_time > latest_results[key_id].expiration_time:
                latest_results[key_id] = SearchResult(maybe_value, maybe_expiration_time, self.node_id)
                if maybe_expiration_time >= sufficient_expiration_time:
                    unfinished_key_ids.remove(key_id)

        # stage 2: traverse the DHT for any unfinished keys
        for key_id in unfinished_key_ids:
            node_to_endpoint.update(self.protocol.routing_table.get_nearest_neighbors(
                key_id, self.protocol.bucket_size, exclude=self.node_id))

        async def get_neighbors(peer: DHTID, queries: Collection[DHTID]) -> Dict[DHTID, Tuple[List[DHTID], bool]]:
            queries = list(queries)
            response = await self.protocol.call_find(node_to_endpoint[peer], queries)
            if not response:
                return {query: ([], False) for query in queries}

            output: Dict[DHTID, Tuple[List[DHTID], bool]] = {}
            for key_id, (maybe_value, maybe_expiration_time, peers) in response.items():
                node_to_endpoint.update(peers)
                if maybe_expiration_time is not None and maybe_expiration_time > latest_results[key_id].expiration_time:
                    latest_results[key_id] = SearchResult(maybe_value, maybe_expiration_time, peer)
                should_interrupt = (latest_results[key_id].expiration_time >= sufficient_expiration_time)
                output[key_id] = list(peers.keys()), should_interrupt
            return output

        nearest_nodes_per_query, visited_nodes = await traverse_dht(
            queries=list(unfinished_key_ids), initial_nodes=list(node_to_endpoint),
            beam_size=beam_size, num_workers=num_workers, queries_per_call=int(len(unfinished_key_ids) ** 0.5),
            get_neighbors=get_neighbors, visited_nodes={key_id: {self.node_id} for key_id in unfinished_key_ids})

        # stage 3: cache any new results depending on caching parameters
        for key_id, nearest_nodes in nearest_nodes_per_query.items():
            latest_value_bytes, latest_expiration_time, latest_node_id = latest_results[key_id]
            should_cache = latest_expiration_time >= sufficient_expiration_time  # if we found a newer value, cache it
            if should_cache and self.cache_locally:
                self.protocol.cache.store(key_id, latest_value_bytes, latest_expiration_time)

            if should_cache and self.cache_nearest:
                num_cached_nodes = 0
                for node_id in nearest_nodes:
                    if node_id == latest_node_id:
                        continue
                    asyncio.create_task(self.protocol.call_store(
                        node_to_endpoint[node_id], [key_id], [latest_value_bytes], [latest_expiration_time],
                        in_cache=True))
                    num_cached_nodes += 1
                    if num_cached_nodes >= self.cache_nearest:
                        break

        # stage 4: deserialize data and assemble function output
        find_result: Dict[DHTKey, Tuple[Optional[DHTValue], Optional[DHTExpiration]]] = {}
        for key_id, (latest_value_bytes, latest_expiration_time, _) in latest_results.items():
            if latest_expiration_time != -float('inf'):
                latest_value = self.serializer.loads(latest_value_bytes)
                find_result[id_to_original_key[key_id]] = (latest_value, latest_expiration_time)
            else:
                find_result[id_to_original_key[key_id]] = None, None
        return find_result

    async def _refresh_routing_table(self, *, period: Optional[float]) -> None:
        """ Tries to find new nodes for buckets that were unused for more than self.staleness_timeout """
        while period is not None:  # if None run once, otherwise run forever
            refresh_time = get_dht_time()
            staleness_threshold = refresh_time - period
            stale_buckets = [bucket for bucket in self.protocol.routing_table.buckets
                             if bucket.last_updated < staleness_threshold]
            for bucket in stale_buckets:
                refresh_id = DHTID(random.randint(bucket.lower, bucket.upper - 1))
                await self.find_nearest_nodes(refresh_id)

            await asyncio.sleep(max(0.0, period - (get_dht_time() - refresh_time)))
