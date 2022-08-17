from __future__ import annotations

import asyncio
import dataclasses
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import (
    Any,
    Awaitable,
    Callable,
    Collection,
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from multiaddr import Multiaddr
from sortedcontainers import SortedSet

from hivemind.dht.crypto import DHTRecord, RecordValidatorBase
from hivemind.dht.protocol import DHTProtocol
from hivemind.dht.routing import DHTID, BinaryDHTValue, DHTKey, DHTValue, Subkey, get_dht_time
from hivemind.dht.storage import DictionaryDHTValue
from hivemind.dht.traverse import traverse_dht
from hivemind.p2p import P2P, PeerID
from hivemind.utils import MSGPackSerializer, SerializerBase, get_logger
from hivemind.utils.auth import AuthorizerBase
from hivemind.utils.timed_storage import DHTExpiration, TimedStorage, ValueWithExpiration

logger = get_logger(__name__)


DEFAULT_NUM_WORKERS = int(os.getenv("HIVEMIND_DHT_NUM_WORKERS", 4))


class DHTNode:
    """
    Asyncio-based class that represents one DHT participant. Created via await DHTNode.create(...)
    Each DHTNode has an identifier, a local storage and access too other nodes via DHTProtocol.

    :note: Hivemind DHT is optimized to store a lot of temporary metadata that is regularly updated.
     For example, expert heartbeat emitted by a hivemind.moe.Server responsible for that expert.
     Such metadata does not require regular maintenance by peers or persistence on shutdown.
     Instead, DHTNode is designed to rapidly send bulk data and resolve conflicts.

    Every (key, value) pair in this DHT has an expiration time - float computed as get_dht_time() (UnixTime by default)
    DHT nodes always prefer values with higher expiration time and may delete any value past its expiration.

    Similar to Kademlia RPC protocol, hivemind DHT has 3 RPCs:

    * ping - request peer's identifier and update routing table (same as Kademlia PING RPC)
    * store - send several (key, value, expiration_time) pairs to the same peer (like Kademlia STORE, but in bulk)
    * find - request one or several keys, get values and expiration (if peer finds it locally) and :bucket_size: of
        nearest peers from recipient's routing table (ordered nearest-to-farthest, not including recipient itself)
        This RPC is a mixture between Kademlia FIND_NODE and FIND_VALUE with multiple keys per call.

    A DHTNode follows the following contract:

    - when asked to get(key), a node must find and return a value with highest expiration time that it found across DHT
      IF that time has not come yet. if expiration time is smaller than current get_dht_time(), node may return None;
    - when requested to store(key: value, expiration_time), a node must store (key => value) at until expiration time
      or until DHTNode gets the same key with greater expiration time. If a node is asked to store a key but it already
      has the same key with newer expiration, store will be rejected. Store returns True if accepted, False if rejected;
    - when requested to store(key: value, expiration_time, subkey=subkey), adds a sub-key to a dictionary value type.
      Dictionary values can have multiple sub-keys stored by different peers with individual expiration times. A subkey
      will be accepted to a dictionary either if there is no such sub-key or if new subkey's expiration is later than
      previous expiration under that subkey. See DHTProtocol.call_store for details.

    DHTNode also features several (optional) caching policies:

    - cache_locally: after GET, store the result in node's own local cache
    - cache_nearest: after GET, send the result to this many nearest nodes that don't have that value yet (see Kademlia)
    - cache_on_store: after STORE, either save or remove that key from node's own cache depending on store status
    - cache_refresh_before_expiry: if a value in cache was used and is about to expire, try to GET it this many seconds
      before expiration. The motivation here is that some frequent keys should be always kept in cache to avoid latency.
    - reuse_get_requests: if there are several concurrent GET requests, when one request finishes, DHTNode will attempt
      to reuse the result of this GET request for other requests with the same key. Useful for batch-parallel requests.

    """

    # fmt:off
    node_id: DHTID; is_alive: bool; peer_id: PeerID; num_replicas: int; num_workers: int; protocol: DHTProtocol
    chunk_size: int; refresh_timeout: float; cache_locally: bool; cache_nearest: int; cache_refresh_before_expiry: float
    cache_on_store: bool; reuse_get_requests: bool; pending_get_requests: DefaultDict[DHTID, SortedSet[_SearchState]]
    cache_refresh_task: Optional[asyncio.Task]; cache_refresh_evt: asyncio.Event; cache_refresh_queue: CacheRefreshQueue
    blacklist: Blacklist
    # fmt:on

    @classmethod
    async def create(
        cls,
        p2p: Optional[P2P] = None,
        node_id: Optional[DHTID] = None,
        initial_peers: Optional[Sequence[Union[Multiaddr, str]]] = None,
        bucket_size: int = 20,
        num_replicas: int = 5,
        depth_modulo: int = 5,
        parallel_rpc: int = None,
        wait_timeout: float = 3,
        refresh_timeout: Optional[float] = None,
        bootstrap_timeout: Optional[float] = None,
        cache_locally: bool = True,
        cache_nearest: int = 1,
        cache_size=None,
        cache_refresh_before_expiry: float = 5,
        cache_on_store: bool = True,
        reuse_get_requests: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
        chunk_size: int = 16,
        blacklist_time: float = 5.0,
        backoff_rate: float = 2.0,
        client_mode: bool = False,
        record_validator: Optional[RecordValidatorBase] = None,
        authorizer: Optional[AuthorizerBase] = None,
        ensure_bootstrap_success: bool = True,
        strict: bool = True,
        **kwargs,
    ) -> DHTNode:
        """
        :param p2p: instance of hivemind.p2p.P2P that will be used for communication.
          If None, DHTNode will create and manage its own P2P instance with given initial_peers and
          parameters from ``kwargs``
        :param node_id: current node's DHTID for hivemind.dht, determines which keys it will store locally,
          defaults to random id
        :param initial_peers: multiaddrs of one or more active DHT peers (if you want to join an existing DHT)
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
        :param cache_locally: if True, caches all values (stored or found) in a node-local cache
        :param cache_on_store: if True, update cache entries for a key after storing a new item for that key
        :param cache_nearest: whenever DHTNode finds a value, it will also store (cache) this value on this many
          nearest nodes visited by search algorithm. Prefers nodes that are nearest to :key: but have no value yet
        :param cache_size: if specified, local cache will store up to this many records (as in LRU cache)
        :param cache_refresh_before_expiry: if nonzero, refreshes locally cached values
          if they are accessed this many seconds before expiration time.
        :param reuse_get_requests: if True, DHTNode allows only one traverse_dht procedure for every key
          all concurrent get requests for the same key will reuse the procedure that is currently in progress
        :param num_workers: concurrent workers in traverse_dht (see traverse_dht num_workers param)
        :param chunk_size: maximum number of concurrent calls in get_many and cache refresh queue
        :param blacklist_time: excludes non-responsive peers from search for this many seconds (set 0 to disable)
        :param backoff_rate: blacklist time will be multiplied by :backoff_rate: for each successive non-response
        :param ensure_bootstrap_success: raise an error if node could not connect to initial peers (or vice versa)
           If False, print a warning instead. It is recommended to keep this flag unless you know what you're doing.
        :param strict: if True, any error encountered in validation will interrupt the creation of DHTNode
        :param client_mode: if False (default), this node will accept incoming requests as a full DHT "citizen"
          if True, this node will refuse any incoming requests, effectively being only a client
        :param record_validator: instance of RecordValidatorBase used for signing and validating stored records
        :param authorizer: instance of AuthorizerBase used for signing and validating requests and response
          for a given authorization protocol
        :param kwargs: extra parameters for an internally created instance of hivemind.p2p.P2P.
          Should be empty if the P2P instance is provided in the constructor
        """
        self = cls(_initialized_with_create=True)
        self.node_id = node_id if node_id is not None else DHTID.generate()
        self.num_replicas, self.num_workers, self.chunk_size = num_replicas, num_workers, chunk_size
        self.is_alive = True  # if set to False, cancels all background jobs such as routing table refresh

        self.reuse_get_requests = reuse_get_requests
        self.pending_get_requests = defaultdict(partial(SortedSet, key=lambda _res: -_res.sufficient_expiration_time))

        # caching policy
        self.refresh_timeout = refresh_timeout
        self.cache_locally, self.cache_nearest, self.cache_on_store = cache_locally, cache_nearest, cache_on_store
        self.cache_refresh_before_expiry = cache_refresh_before_expiry
        self.blacklist = Blacklist(blacklist_time, backoff_rate)
        self.cache_refresh_queue = CacheRefreshQueue()
        self.cache_refresh_evt = asyncio.Event()
        self.cache_refresh_task = None

        if p2p is None:
            if not kwargs.get("use_ipfs"):
                kwargs["initial_peers"] = initial_peers
            if client_mode:
                kwargs.setdefault("dht_mode", "client")
            p2p = await P2P.create(**kwargs)
            self._should_shutdown_p2p = True
        else:
            if kwargs:
                raise ValueError(
                    f"**kwargs in DHTNode.create() should be empty if hivemind.p2p.P2P instance is provided"
                    f"in the constructor. Got kwargs = {kwargs} instead. "
                    f"You may have a typo in a DHTNode.create() parameter name"
                )
            self._should_shutdown_p2p = False
        self.p2p = p2p

        self.protocol = await DHTProtocol.create(
            p2p,
            self.node_id,
            bucket_size,
            depth_modulo,
            num_replicas,
            wait_timeout,
            parallel_rpc,
            cache_size,
            client_mode,
            record_validator,
            authorizer,
        )
        self.peer_id = p2p.peer_id

        if initial_peers:
            initial_peers = {PeerID.from_base58(Multiaddr(item)["p2p"]) for item in initial_peers}

            # stage 1: ping initial_peers, add each other to the routing table
            bootstrap_timeout = bootstrap_timeout if bootstrap_timeout is not None else wait_timeout
            start_time = get_dht_time()
            ping_tasks = set(
                asyncio.create_task(self.protocol.call_ping(peer, validate=ensure_bootstrap_success, strict=strict))
                for peer in initial_peers
            )
            finished_pings, unfinished_pings = await asyncio.wait(ping_tasks, return_when=asyncio.FIRST_COMPLETED)

            # stage 2: gather remaining peers (those who respond within bootstrap_timeout)
            if unfinished_pings:
                finished_in_time, stragglers = await asyncio.wait(
                    unfinished_pings, timeout=bootstrap_timeout - get_dht_time() + start_time
                )
                for straggler in stragglers:
                    straggler.cancel()
                finished_pings |= finished_in_time

            if not finished_pings or all(ping.result() is None for ping in finished_pings):
                message = "DHTNode bootstrap failed: none of the initial_peers responded to a ping."
                if ensure_bootstrap_success:
                    raise RuntimeError(f"{message} (set ensure_bootstrap_success=False to ignore)")
                else:
                    logger.warning(message)

            if strict:
                for task in asyncio.as_completed(finished_pings):
                    await task  # propagate exceptions

            # stage 3: traverse dht to find my own nearest neighbors and populate the routing table
            # ... maybe receive some values that we are meant to store (see protocol.update_routing_table)
            # note: using asyncio.wait instead of wait_for because wait_for cancels task on timeout
            await asyncio.wait(
                [
                    asyncio.create_task(self.find_nearest_nodes([self.node_id])),
                    asyncio.sleep(bootstrap_timeout - get_dht_time() + start_time),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

        if self.refresh_timeout is not None:
            asyncio.create_task(self._refresh_routing_table(period=self.refresh_timeout))
        return self

    def __init__(self, *, _initialized_with_create=False):
        """Internal init method. Please use DHTNode.create coroutine to spawn new node instances"""
        assert _initialized_with_create, " Please use DHTNode.create coroutine to spawn new node instances "
        super().__init__()

    async def shutdown(self):
        """Process existing requests, close all connections and stop the server"""
        self.is_alive = False
        await self.protocol.shutdown()
        if self._should_shutdown_p2p:
            await self.p2p.shutdown()

    async def find_nearest_nodes(
        self,
        queries: Collection[DHTID],
        k_nearest: Optional[int] = None,
        beam_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        node_to_peer_id: Optional[Dict[DHTID, PeerID]] = None,
        exclude_self: bool = False,
        **kwargs,
    ) -> Dict[DHTID, Dict[DHTID, PeerID]]:
        """
        :param queries: find k nearest nodes for each of these DHTIDs
        :param k_nearest: return this many nearest nodes for every query (if there are enough nodes)
        :param beam_size: replacement for self.beam_size, see traverse_dht beam_size param
        :param num_workers: replacement for self.num_workers, see traverse_dht num_workers param
        :param node_to_peer_id: if specified, uses this dict[node_id => peer_id] as initial peers
        :param exclude_self: if True, nearest nodes will not contain self.node_id (default = use local peers)
        :param kwargs: additional params passed to traverse_dht
        :returns: for every query, return nearest peers ordered dict[peer DHTID -> network PeerID], nearest-first
        """
        queries = tuple(queries)
        k_nearest = k_nearest if k_nearest is not None else self.protocol.bucket_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        beam_size = beam_size if beam_size is not None else max(self.protocol.bucket_size, k_nearest)
        if k_nearest > beam_size:
            logger.warning("Warning: beam_size is too small, beam search is not guaranteed to find enough nodes")
        if node_to_peer_id is None:
            node_to_peer_id: Dict[DHTID, PeerID] = dict()
            for query in queries:
                neighbors = self.protocol.routing_table.get_nearest_neighbors(query, beam_size, exclude=self.node_id)
                node_to_peer_id.update(self._filter_blacklisted(dict(neighbors)))

        async def get_neighbors(peer: DHTID, queries: Collection[DHTID]) -> Dict[DHTID, Tuple[Tuple[DHTID], bool]]:
            response = await self._call_find_with_blacklist(node_to_peer_id[peer], queries)
            if not response:
                return {query: ([], False) for query in queries}

            output: Dict[DHTID, Tuple[Tuple[DHTID], bool]] = {}
            for query, (_, peers) in response.items():
                node_to_peer_id.update(peers)
                output[query] = tuple(peers.keys()), False  # False means "do not interrupt search"
            return output

        nearest_nodes_per_query, visited_nodes = await traverse_dht(
            queries,
            initial_nodes=list(node_to_peer_id),
            beam_size=beam_size,
            num_workers=num_workers,
            queries_per_call=int(len(queries) ** 0.5),
            get_neighbors=get_neighbors,
            visited_nodes={query: {self.node_id} for query in queries},
            **kwargs,
        )

        nearest_nodes_with_peer_ids = {}
        for query, nearest_nodes in nearest_nodes_per_query.items():
            if not exclude_self:
                nearest_nodes = sorted(nearest_nodes + [self.node_id], key=query.xor_distance)
                node_to_peer_id[self.node_id] = self.peer_id
            nearest_nodes_with_peer_ids[query] = {node: node_to_peer_id[node] for node in nearest_nodes[:k_nearest]}
        return nearest_nodes_with_peer_ids

    async def store(
        self, key: DHTKey, value: DHTValue, expiration_time: DHTExpiration, subkey: Optional[Subkey] = None, **kwargs
    ) -> bool:
        """
        Find num_replicas best nodes to store (key, value) and store it there at least until expiration time.
        :note: store is a simplified interface to store_many, all kwargs are forwarded there
        :returns: True if store succeeds, False if it fails (due to no response or newer value)
        """
        store_ok = await self.store_many([key], [value], [expiration_time], subkeys=[subkey], **kwargs)
        return store_ok[(key, subkey) if subkey is not None else key]

    async def store_many(
        self,
        keys: List[DHTKey],
        values: List[DHTValue],
        expiration_time: Union[DHTExpiration, List[DHTExpiration]],
        subkeys: Optional[Union[Subkey, List[Optional[Subkey]]]] = None,
        exclude_self: bool = False,
        await_all_replicas=True,
        **kwargs,
    ) -> Dict[DHTKey, bool]:
        """
        Traverse DHT to find up :num_replicas: to best nodes to store multiple (key, value, expiration_time) pairs.

        :param keys: arbitrary serializable keys associated with each value
        :param values: serializable "payload" for each key
        :param expiration_time: either one expiration time for all keys or individual expiration times (see class doc)
        :param subkeys: an optional list of same shape as keys. If specified, this
        :param kwargs: any additional parameters passed to traverse_dht function (e.g. num workers)
        :param exclude_self: if True, never store value locally even if you are one of the nearest nodes
        :note: if exclude_self is True and self.cache_locally == True, value will still be __cached__ locally
        :param await_all_replicas: if False, this function returns after first store_ok and proceeds in background
            if True, the function will wait for num_replicas successful stores or running out of beam_size nodes
        :returns: for each key: True if store succeeds, False if it fails (due to no response or newer value)
        """
        if isinstance(expiration_time, DHTExpiration):
            expiration_time = [expiration_time] * len(keys)
        if subkeys is None:
            subkeys = [None] * len(keys)

        assert (
            len(keys) == len(subkeys) == len(values) == len(expiration_time)
        ), "Either of keys, values, subkeys or expiration timestamps have different sequence lengths."

        key_id_to_data: DefaultDict[DHTID, List[Tuple[DHTKey, Subkey, DHTValue, DHTExpiration]]] = defaultdict(list)
        for key, subkey, value, expiration in zip(keys, subkeys, values, expiration_time):
            key_id_to_data[DHTID.generate(source=key)].append((key, subkey, value, expiration))

        unfinished_key_ids = set(key_id_to_data.keys())  # use this set to ensure that each store request is finished
        store_ok = {(key, subkey): None for key, subkey in zip(keys, subkeys)}  # outputs, updated during search
        store_finished_events = {(key, subkey): asyncio.Event() for key, subkey in zip(keys, subkeys)}

        # pre-populate node_to_peer_id
        node_to_peer_id: Dict[DHTID, PeerID] = dict()
        for key_id in unfinished_key_ids:
            node_to_peer_id.update(
                self.protocol.routing_table.get_nearest_neighbors(
                    key_id, self.protocol.bucket_size, exclude=self.node_id
                )
            )

        async def on_found(key_id: DHTID, nearest_nodes: List[DHTID], visited_nodes: Set[DHTID]) -> None:
            """This will be called once per key when find_nearest_nodes is done for a particular node"""
            # note: we use callbacks instead of returned values to call store immediately without waiting for stragglers
            assert key_id in unfinished_key_ids, "Internal error: traverse_dht finished the same query twice"
            assert self.node_id not in nearest_nodes
            unfinished_key_ids.remove(key_id)

            # ensure k nodes stored the value, optionally include self.node_id as a candidate
            num_successful_stores = 0
            pending_store_tasks = set()
            store_candidates = sorted(
                nearest_nodes + ([] if exclude_self else [self.node_id]), key=key_id.xor_distance, reverse=True
            )  # ordered so that .pop() returns nearest
            [original_key, *_], current_subkeys, current_values, current_expirations = zip(*key_id_to_data[key_id])

            key_bytes = key_id.to_bytes()
            binary_values = []
            stored_records = []
            for subkey, value, expiration_time in zip(current_subkeys, current_values, current_expirations):
                subkey_bytes = self.protocol.serializer.dumps(subkey)
                value_bytes = self.protocol.serializer.dumps(value)
                record = DHTRecord(key_bytes, subkey_bytes, value_bytes, expiration_time)
                if self.protocol.record_validator is not None:
                    value_bytes = self.protocol.record_validator.sign_value(record)
                    record = dataclasses.replace(record, value=value_bytes)
                binary_values.append(value_bytes)
                stored_records.append(record)

            while num_successful_stores < self.num_replicas and (store_candidates or pending_store_tasks):
                while store_candidates and num_successful_stores + len(pending_store_tasks) < self.num_replicas:
                    node_id: DHTID = store_candidates.pop()  # nearest untried candidate

                    if node_id == self.node_id:
                        num_successful_stores += 1
                        for subkey, record in zip(current_subkeys, stored_records):
                            if self.protocol.record_validator is None or self.protocol.record_validator.validate(
                                record
                            ):
                                store_ok[original_key, subkey] = self.protocol.storage.store(
                                    key_id, record.value, record.expiration_time, subkey=subkey
                                )
                            else:
                                store_ok[original_key, subkey] = False
                            if not await_all_replicas:
                                store_finished_events[original_key, subkey].set()
                    else:
                        pending_store_tasks.add(
                            asyncio.create_task(
                                self.protocol.call_store(
                                    node_to_peer_id[node_id],
                                    keys=[key_id] * len(current_values),
                                    values=binary_values,
                                    expiration_time=current_expirations,
                                    subkeys=current_subkeys,
                                )
                            )
                        )

                # await nearest task. If it fails, dispatch more on the next iteration
                if pending_store_tasks:
                    finished_store_tasks, pending_store_tasks = await asyncio.wait(
                        pending_store_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in finished_store_tasks:
                        if task.result() is not None:
                            num_successful_stores += 1
                            for subkey, store_status in zip(current_subkeys, task.result()):
                                store_ok[original_key, subkey] = store_status
                                if not await_all_replicas:
                                    store_finished_events[original_key, subkey].set()

            if self.cache_on_store:
                self._update_cache_on_store(
                    key_id,
                    current_subkeys,
                    binary_values,
                    current_expirations,
                    store_ok=[store_ok[original_key, subkey] for subkey in current_subkeys],
                )

            for subkey, value_bytes, expiration in zip(current_subkeys, binary_values, current_expirations):
                store_finished_events[original_key, subkey].set()

        store_task = asyncio.create_task(
            self.find_nearest_nodes(
                queries=set(unfinished_key_ids),
                k_nearest=self.num_replicas,
                node_to_peer_id=node_to_peer_id,
                found_callback=on_found,
                exclude_self=exclude_self,
                **kwargs,
            )
        )
        try:
            await asyncio.gather(store_task, *(evt.wait() for evt in store_finished_events.values()))
            assert len(unfinished_key_ids) == 0, "Internal error: traverse_dht didn't finish search"
            return {
                (key, subkey) if subkey is not None else key: status or False
                for (key, subkey), status in store_ok.items()
            }
        except asyncio.CancelledError as e:
            store_task.cancel()
            raise e

    def _update_cache_on_store(
        self,
        key_id: DHTID,
        subkeys: List[Subkey],
        binary_values: List[bytes],
        expirations: List[DHTExpiration],
        store_ok: List[bool],
    ):
        """Update local cache after finishing a store for one key (with perhaps several subkeys)"""
        store_succeeded = any(store_ok)
        is_dictionary = any(subkey is not None for subkey in subkeys)
        if store_succeeded and not is_dictionary:  # stored a new regular value, cache it!
            stored_expiration, stored_value_bytes = max(zip(expirations, binary_values))
            self.protocol.cache.store(key_id, stored_value_bytes, stored_expiration)
        elif not store_succeeded and not is_dictionary:  # store rejected, check if local cache is also obsolete
            rejected_expiration, rejected_value = max(zip(expirations, binary_values))
            cached_value = self.protocol.cache.get(key_id)
            if (
                cached_value is not None and cached_value.expiration_time <= rejected_expiration
            ):  # cache would be rejected
                self._schedule_for_refresh(key_id, refresh_time=get_dht_time())  # fetch new key in background (asap)
        elif is_dictionary and key_id in self.protocol.cache:  # there can be other keys and we should update
            for subkey, stored_value_bytes, expiration_time, accepted in zip(
                subkeys, binary_values, expirations, store_ok
            ):
                if accepted:
                    self.protocol.cache.store_subkey(key_id, subkey, stored_value_bytes, expiration_time)
            self._schedule_for_refresh(key_id, refresh_time=get_dht_time())  # fetch new key in background (asap)

    async def get(self, key: DHTKey, latest=False, **kwargs) -> Optional[ValueWithExpiration[DHTValue]]:
        """
        Search for a key across DHT and return either first or latest entry (if found).
        :param key: same key as in node.store(...)
        :param latest: if True, finds the latest value, otherwise finds any non-expired value (which is much faster)
        :param kwargs: parameters forwarded to get_many_by_id
        :returns: (value, expiration time); if value was not found, returns None
        """
        if latest:
            kwargs["sufficient_expiration_time"] = float("inf")
        result = await self.get_many([key], **kwargs)
        return result[key]

    async def get_many(
        self, keys: Collection[DHTKey], sufficient_expiration_time: Optional[DHTExpiration] = None, **kwargs
    ) -> Dict[
        DHTKey, Union[Optional[ValueWithExpiration[DHTValue]], Awaitable[Optional[ValueWithExpiration[DHTValue]]]]
    ]:
        """
        Traverse DHT to find a list of keys. For each key, return latest (value, expiration) or None if not found.

        :param keys: traverse the DHT and find the value for each of these keys (or (None, None) if not key found)
        :param sufficient_expiration_time: if the search finds a value that expires after this time,
            default = time of call, find any value that did not expire by the time of call
            If min_expiration_time=float('inf'), this method will find a value with _latest_ expiration
        :param kwargs: for full list of parameters, see DHTNode.get_many_by_id
        :returns: for each key: value and its expiration time. If nothing is found, returns (None, None) for that key
        :note: in order to check if get returned a value, please check if (expiration_time is None)
        """
        keys = tuple(keys)
        key_ids = [DHTID.generate(key) for key in keys]
        id_to_original_key = dict(zip(key_ids, keys))
        results_by_id = await self.get_many_by_id(key_ids, sufficient_expiration_time, **kwargs)
        return {id_to_original_key[key]: result_or_future for key, result_or_future in results_by_id.items()}

    async def get_many_by_id(
        self,
        key_ids: Collection[DHTID],
        sufficient_expiration_time: Optional[DHTExpiration] = None,
        num_workers: Optional[int] = None,
        beam_size: Optional[int] = None,
        return_futures: bool = False,
        _is_refresh=False,
    ) -> Dict[
        DHTID, Union[Optional[ValueWithExpiration[DHTValue]], Awaitable[Optional[ValueWithExpiration[DHTValue]]]]
    ]:
        """
        Traverse DHT to find a list of DHTIDs. For each key, return latest (value, expiration) or None if not found.

        :param key_ids: traverse the DHT and find the value for each of these keys (or (None, None) if not key found)
        :param sufficient_expiration_time: if the search finds a value that expires after this time,
            default = time of call, find any value that did not expire by the time of call
            If min_expiration_time=float('inf'), this method will find a value with _latest_ expiration
        :param beam_size: maintains up to this many nearest nodes when crawling dht, default beam_size = bucket_size
        :param num_workers: override for default num_workers, see traverse_dht num_workers param
        :param return_futures: if True, immediately return asyncio.Future for every before interacting with the nework.
         The algorithm will populate these futures with (value, expiration) when it finds the corresponding key
         Note: canceling a future will stop search for the corresponding key
        :param _is_refresh: internal flag, set to True by an internal cache refresher (if enabled)
        :returns: for each key: value and its expiration time. If nothing is found, returns (None, None) for that key
        :note: in order to check if get returned a value, please check (expiration_time is None)
        """
        sufficient_expiration_time = sufficient_expiration_time or get_dht_time()
        beam_size = beam_size if beam_size is not None else self.protocol.bucket_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        search_results: Dict[DHTID, _SearchState] = {
            key_id: _SearchState(
                key_id,
                sufficient_expiration_time,
                serializer=self.protocol.serializer,
                record_validator=self.protocol.record_validator,
            )
            for key_id in key_ids
        }

        if not _is_refresh:  # if we're already refreshing cache, there's no need to trigger subsequent refreshes
            for key_id in key_ids:
                search_results[key_id].add_done_callback(self._trigger_cache_refresh)

        # if we have concurrent get request for some of the same keys, subscribe to their results
        if self.reuse_get_requests:
            for key_id, search_result in search_results.items():
                self.pending_get_requests[key_id].add(search_result)
                search_result.add_done_callback(self._reuse_finished_search_result)

        # stage 1: check for value in this node's local storage and cache
        for key_id in key_ids:
            search_results[key_id].add_candidate(self.protocol.storage.get(key_id), source_node_id=self.node_id)
            if not _is_refresh:
                search_results[key_id].add_candidate(self.protocol.cache.get(key_id), source_node_id=self.node_id)

        # stage 2: traverse the DHT to get the remaining keys from remote peers
        unfinished_key_ids = [key_id for key_id in key_ids if not search_results[key_id].finished]
        node_to_peer_id: Dict[DHTID, PeerID] = dict()  # global routing table for all keys
        for key_id in unfinished_key_ids:
            node_to_peer_id.update(
                self.protocol.routing_table.get_nearest_neighbors(
                    key_id, self.protocol.bucket_size, exclude=self.node_id
                )
            )

        # V-- this function will be called every time traverse_dht decides to request neighbors from a remote peer
        async def get_neighbors(peer: DHTID, queries: Collection[DHTID]) -> Dict[DHTID, Tuple[Tuple[DHTID], bool]]:
            queries = list(queries)
            response = await self._call_find_with_blacklist(node_to_peer_id[peer], queries)
            if not response:
                return {query: ([], False) for query in queries}

            output: Dict[DHTID, Tuple[Tuple[DHTID], bool]] = {}
            for key_id, (maybe_value_with_expiration, peers) in response.items():
                node_to_peer_id.update(peers)
                search_results[key_id].add_candidate(maybe_value_with_expiration, source_node_id=peer)
                output[key_id] = tuple(peers.keys()), search_results[key_id].finished
                # note: we interrupt search either if key is either found or finished otherwise (e.g. cancelled by user)
            return output

        # V-- this function will be called exactly once when traverse_dht finishes search for a given key
        async def found_callback(key_id: DHTID, nearest_nodes: List[DHTID], _visited: Set[DHTID]):
            search_results[key_id].finish_search()  # finish search whether or we found something
            self._cache_new_result(search_results[key_id], nearest_nodes, node_to_peer_id, _is_refresh=_is_refresh)

        asyncio.create_task(
            traverse_dht(
                queries=list(unfinished_key_ids),
                initial_nodes=list(node_to_peer_id),
                beam_size=beam_size,
                num_workers=num_workers,
                queries_per_call=min(int(len(unfinished_key_ids) ** 0.5), self.chunk_size),
                get_neighbors=get_neighbors,
                visited_nodes={key_id: {self.node_id} for key_id in unfinished_key_ids},
                found_callback=found_callback,
                await_all_tasks=False,
            )
        )

        if return_futures:
            return {key_id: search_result.future for key_id, search_result in search_results.items()}
        else:
            try:
                # note: this should be first time when we await something, there's no need to "try" the entire function
                return {key_id: await search_result.future for key_id, search_result in search_results.items()}
            except asyncio.CancelledError as e:  # terminate remaining tasks ASAP
                for key_id, search_result in search_results.items():
                    search_result.future.cancel()
                raise e

    def _reuse_finished_search_result(self, finished: _SearchState):
        pending_requests = self.pending_get_requests[finished.key_id]
        if finished.found_something:
            search_result = ValueWithExpiration(finished.binary_value, finished.expiration_time)
            expiration_time_threshold = max(finished.expiration_time, finished.sufficient_expiration_time)
            # note: pending_requests is sorted in the order of descending sufficient_expiration_time
            while pending_requests and expiration_time_threshold >= pending_requests[-1].sufficient_expiration_time:
                pending_requests[-1].add_candidate(search_result, source_node_id=finished.source_node_id)
                pending_requests[-1].finish_search()
                pending_requests.pop()
        else:
            pending_requests.discard(finished)

    async def _call_find_with_blacklist(self, peer_id: PeerID, keys: Collection[DHTID]):
        """same as call_find, but skip if :peer_id: is blacklisted; also exclude blacklisted neighbors from result"""
        if peer_id in self.blacklist:
            return None
        response = await self.protocol.call_find(peer_id, keys)
        if response:
            self.blacklist.register_success(peer_id)
            return {
                key: (maybe_value, self._filter_blacklisted(nearest_peers))
                for key, (maybe_value, nearest_peers) in response.items()
            }
        else:
            self.blacklist.register_failure(peer_id)
            return None

    def _filter_blacklisted(self, peer_ids: Dict[DHTID, PeerID]):
        return {peer: peer_id for peer, peer_id in peer_ids.items() if peer_id not in self.blacklist}

    def _trigger_cache_refresh(self, search: _SearchState):
        """Called after get request is finished (whether it was found, not found, hit cache, cancelled, or reused)"""
        if search.found_something and search.source_node_id == self.node_id:
            if self.cache_refresh_before_expiry and search.key_id in self.protocol.cache:
                self._schedule_for_refresh(search.key_id, search.expiration_time - self.cache_refresh_before_expiry)

    def _schedule_for_refresh(self, key_id: DHTID, refresh_time: DHTExpiration):
        """Add key to a refresh queue, refresh at :refresh_time: or later"""
        if self.cache_refresh_task is None or self.cache_refresh_task.done() or self.cache_refresh_task.cancelled():
            self.cache_refresh_task = asyncio.create_task(self._refresh_stale_cache_entries())
            logger.debug("Spawned cache refresh task")
        earliest_key, earliest_item = self.cache_refresh_queue.top()
        if earliest_item is None or refresh_time < earliest_item.expiration_time:
            self.cache_refresh_evt.set()  # if we new element is now earliest, notify the cache queue
        self.cache_refresh_queue.store(key_id, value=refresh_time, expiration_time=refresh_time)

    async def _refresh_stale_cache_entries(self):
        """periodically refresh keys near-expired keys that were accessed at least once during previous lifetime"""
        while self.is_alive:
            while len(self.cache_refresh_queue) == 0:
                await self.cache_refresh_evt.wait()
                self.cache_refresh_evt.clear()
            key_id, (_, nearest_refresh_time) = self.cache_refresh_queue.top()

            try:
                # step 1: await until :cache_refresh_before_expiry: seconds before earliest first element expires
                time_to_wait = nearest_refresh_time - get_dht_time()
                await asyncio.wait_for(self.cache_refresh_evt.wait(), timeout=time_to_wait)
                # note: the line above will cause TimeoutError when we are ready to refresh cache
                self.cache_refresh_evt.clear()  # no timeout error => someone added new entry to queue and ...
                continue  # ... and this element is earlier than nearest_expiration. we should refresh this entry first

            except asyncio.TimeoutError:  # caught TimeoutError => it is time to refresh the most recent cached entry
                # step 2: find all keys that we should already refresh and remove them from queue
                current_time = get_dht_time()
                keys_to_refresh = {key_id}
                max_expiration_time = nearest_refresh_time
                del self.cache_refresh_queue[key_id]  # we pledge to refresh this key_id in the nearest batch
                while self.cache_refresh_queue and len(keys_to_refresh) < self.chunk_size:
                    key_id, (_, nearest_refresh_time) = self.cache_refresh_queue.top()
                    if nearest_refresh_time > current_time:
                        break
                    del self.cache_refresh_queue[key_id]  # we pledge to refresh this key_id in the nearest batch
                    keys_to_refresh.add(key_id)
                    cached_item = self.protocol.cache.get(key_id)
                    if cached_item is not None and cached_item.expiration_time > max_expiration_time:
                        max_expiration_time = cached_item.expiration_time

                # step 3: search newer versions of these keys, cache them as a side-effect of self.get_many_by_id
                sufficient_expiration_time = max_expiration_time + self.cache_refresh_before_expiry + 1
                await self.get_many_by_id(keys_to_refresh, sufficient_expiration_time, _is_refresh=True)

    def _cache_new_result(
        self,
        search: _SearchState,
        nearest_nodes: List[DHTID],
        node_to_peer_id: Dict[DHTID, PeerID],
        _is_refresh: bool = False,
    ):
        """after key_id is found, update cache according to caching policy. used internally in get and get_many"""
        if search.found_something:
            _, storage_expiration_time = self.protocol.storage.get(search.key_id) or (None, -float("inf"))
            _, cache_expiration_time = self.protocol.cache.get(search.key_id) or (None, -float("inf"))

            if search.expiration_time > max(storage_expiration_time, cache_expiration_time):
                if self.cache_locally or _is_refresh:
                    self.protocol.cache.store(search.key_id, search.binary_value, search.expiration_time)
                if self.cache_nearest:
                    num_cached_nodes = 0
                    for node_id in nearest_nodes:
                        if node_id == search.source_node_id:
                            continue
                        asyncio.create_task(
                            self.protocol.call_store(
                                node_to_peer_id[node_id],
                                [search.key_id],
                                [search.binary_value],
                                [search.expiration_time],
                                in_cache=True,
                            )
                        )
                        num_cached_nodes += 1
                        if num_cached_nodes >= self.cache_nearest:
                            break

    async def _refresh_routing_table(self, *, period: Optional[float]) -> None:
        """Tries to find new nodes for buckets that were unused for more than self.staleness_timeout"""
        while self.is_alive and period is not None:  # if None run once, otherwise run forever
            refresh_time = get_dht_time()
            staleness_threshold = refresh_time - period
            stale_buckets = [
                bucket for bucket in self.protocol.routing_table.buckets if bucket.last_updated < staleness_threshold
            ]
            for bucket in stale_buckets:
                refresh_id = DHTID(random.randint(bucket.lower, bucket.upper - 1))
                await self.find_nearest_nodes(refresh_id)

            await asyncio.sleep(max(0.0, period - (get_dht_time() - refresh_time)))

    async def get_visible_maddrs(self, latest: bool = False) -> List[Multiaddr]:
        return await self.protocol.p2p.get_visible_maddrs(latest=latest)


@dataclass(init=True, repr=True, frozen=False, order=False)
class _SearchState:
    """A helper class that stores current-best GET results with metadata"""

    key_id: DHTID
    sufficient_expiration_time: DHTExpiration
    binary_value: Optional[Union[BinaryDHTValue, DictionaryDHTValue]] = None
    expiration_time: Optional[DHTExpiration] = None  # best expiration time so far
    source_node_id: Optional[DHTID] = None  # node that gave us the value
    future: asyncio.Future[Optional[ValueWithExpiration[DHTValue]]] = field(default_factory=asyncio.Future)
    serializer: Type[SerializerBase] = MSGPackSerializer
    record_validator: Optional[RecordValidatorBase] = None

    def add_candidate(
        self,
        candidate: Optional[ValueWithExpiration[Union[BinaryDHTValue, DictionaryDHTValue]]],
        source_node_id: Optional[DHTID],
    ):
        if self.finished or candidate is None:
            return
        elif isinstance(candidate.value, DictionaryDHTValue) and isinstance(self.binary_value, DictionaryDHTValue):
            self.binary_value.maxsize = max(self.binary_value.maxsize, candidate.value.maxsize)
            for subkey, subentry in candidate.value.items():
                self.binary_value.store(subkey, subentry.value, subentry.expiration_time)
        elif candidate.expiration_time > (self.expiration_time or float("-inf")):
            self.binary_value = candidate.value

        if candidate.expiration_time > (self.expiration_time or float("-inf")):
            self.expiration_time = candidate.expiration_time
            self.source_node_id = source_node_id
            if self.expiration_time >= self.sufficient_expiration_time:
                self.finish_search()

    def add_done_callback(self, callback: Callable[[_SearchState], Any]):
        """Add callback that will be called when _SearchState is done (found OR cancelled by user)"""
        self.future.add_done_callback(lambda _future: callback(self))

    def finish_search(self):
        if self.future.done():
            return  # either user cancelled our search or someone sent it before us. Nothing more to do here.
        elif not self.found_something:
            self.future.set_result(None)
        elif isinstance(self.binary_value, BinaryDHTValue):
            value_bytes = self.binary_value
            if self.record_validator is not None:
                record = DHTRecord(
                    self.key_id.to_bytes(), DHTProtocol.IS_REGULAR_VALUE, value_bytes, self.expiration_time
                )
                value_bytes = self.record_validator.strip_value(record)

            self.future.set_result(ValueWithExpiration(self.serializer.loads(value_bytes), self.expiration_time))
        elif isinstance(self.binary_value, DictionaryDHTValue):
            dict_with_subkeys = {}
            for subkey, (value_bytes, item_expiration_time) in self.binary_value.items():
                if self.record_validator is not None:
                    subkey_bytes = self.serializer.dumps(subkey)
                    record = DHTRecord(self.key_id.to_bytes(), subkey_bytes, value_bytes, item_expiration_time)
                    value_bytes = self.record_validator.strip_value(record)

                dict_with_subkeys[subkey] = ValueWithExpiration(
                    self.serializer.loads(value_bytes), item_expiration_time
                )
            self.future.set_result(ValueWithExpiration(dict_with_subkeys, self.expiration_time))
        else:
            logger.error(f"Invalid value type: {type(self.binary_value)}")

    @property
    def found_something(self) -> bool:
        """Whether or not we have found at least some value, regardless of its expiration time"""
        return self.expiration_time is not None

    @property
    def finished(self) -> bool:
        return self.future.done()

    def __lt__(self, other: _SearchState):
        """_SearchState instances will be sorted by their target expiration time"""
        return self.sufficient_expiration_time < other.sufficient_expiration_time

    def __hash__(self):
        return hash(self.key_id)


class Blacklist:
    """
    A temporary blacklist of non-responding peers with exponential backoff policy
    :param base_time: peers are suspended for this many seconds by default
    :param backoff_rate: suspension time increases by this factor after each successive failure
    """

    def __init__(self, base_time: float, backoff_rate: float, **kwargs):
        self.base_time, self.backoff = base_time, backoff_rate
        self.banned_peers = TimedStorage[PeerID, int](**kwargs)
        self.ban_counter = Counter()

    def register_failure(self, peer: PeerID):
        """peer failed to respond, add him to blacklist or increase his downtime"""
        if peer not in self.banned_peers and self.base_time > 0:
            ban_duration = self.base_time * self.backoff ** self.ban_counter[peer]
            self.banned_peers.store(peer, self.ban_counter[peer], expiration_time=get_dht_time() + ban_duration)
            self.ban_counter[peer] += 1

    def register_success(self, peer):
        """peer responded successfully, remove him from blacklist and reset his ban time"""
        del self.banned_peers[peer], self.ban_counter[peer]

    def __contains__(self, peer: PeerID) -> bool:
        return peer in self.banned_peers

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(base_time={self.base_time}, backoff={self.backoff}, "
            f"banned_peers={len(self.banned_peers)})"
        )

    def clear(self):
        self.banned_peers.clear()
        self.ban_counter.clear()


class CacheRefreshQueue(TimedStorage[DHTID, DHTExpiration]):
    """a queue of keys scheduled for refresh in future, used in DHTNode"""

    frozen = True
