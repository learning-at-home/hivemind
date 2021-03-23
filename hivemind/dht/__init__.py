"""
This is a Distributed Hash Table optimized for rapidly accessing a lot of lightweight metadata.
Hivemind DHT is based on Kademlia [1] with added support for improved bulk store/get operations and caching.

The code is organized as follows:

 * **class DHT (__init__.py)** - high-level class for model training. Runs DHTNode in a background process.
 * **class DHTNode (node.py)** - an asyncio implementation of dht server, stores AND gets keys.
 * **class DHTProtocol (protocol.py)** - an RPC protocol to request data from dht nodes.
 * **async def traverse_dht (traverse.py)** - a search algorithm that crawls DHT peers.

- [1] Maymounkov P., Mazieres D. (2002) Kademlia: A Peer-to-Peer Information System Based on the XOR Metric.
- [2] https://github.com/bmuller/kademlia , Brian, if you're reading this: THANK YOU! you're awesome :)
"""
from __future__ import annotations
import asyncio
import ctypes
import heapq
import multiprocessing as mp
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Sequence, Union, Dict, Deque, Iterator, Set, Callable, Awaitable, TypeVar
from deprecated import deprecated

from hivemind.client import RemoteExpert
from hivemind.dht.node import DHTNode, DHTID, DHTExpiration
from hivemind.dht.routing import get_dht_time, DHTValue, DHTKey, Subkey
from hivemind.utils.networking import Hostname, Endpoint, strip_port
from hivemind.utils import MPFuture, get_logger, switch_to_uvloop, ValueWithExpiration, await_cancelled
from hivemind.client.expert_uid import * #TODO remove

logger = get_logger(__name__)

ReturnType = TypeVar('ReturnType')


class DHT(mp.Process):
    """
    High-level interface to hivemind.dht that is designed to allow RemoteMixtureOfExperts to select best experts.
    * hivemind servers periodically announce their experts via declare_experts (dht_handler.py)
    * trainers find most suitable experts via RemoteMixtureOfExperts (beam_saerch.py)

    :param initial_peers: one or multiple endpoints pointing to active DHT peers. Similar format to listen_on.
    :param listen_on: an interface for incoming connections, e.g. "127.0.0.1:*", "0.0.0.0:1234" or "ipv6:[::]:*"
    :param start: if True, automatically starts the background process on creation. Otherwise await manual start
    :param daemon: if True, the background process is marked as daemon and automatically terminated after main process
    :param max_workers: declare_experts and get_experts will use up to this many parallel workers
        (but no more than one per key)
    :param expiration: experts declared from this node expire after this many seconds (default = 5 minutes)
    :param receiver_threads: uses this many threads to await on input pipe. Default = 1 should be enough in most cases
    :param negative_caching: if True, whenever DHT is unable to find an expert or prefix, it will cache the "no key"
      result inside the DHT for :expiration: seconds. Caching only affects beam search and has three main effects:

      1. Faster beam search under node failures: if there are inconsistencies in DHT keys, such as a prefix pointing to
         a now-defunct expert, these inconsistencies will be overwritten by the first peer that stumbles upon them. As a
         result, beam search will not have to wait for non-existent experts until the expiration of their DHT entries;
      2. Delayed expert availability: Without negative cache, new experts are always immediately available for beam
         search after they are published to the DHT. With negative cache, there are rare cases (e.g. when adding new
         experts in place of recently defunct ones) when new experts will be initially invisible, but gradually become
         visible to more peers as those peers refresh their cache. This process takes at most :expiration: seconds;
      3. Faster beam search in very sparse grids: there is one edge case where negative cache will improve beam search
         performance; If an expert grid is very sparse, there can be empty indices in the first grid dimension (i.e.
         indices {i} such that _no_ experts that start with "{prefix}.{i}.*"). If so, the default beam search will
         be very slow due to the way it forms initial beam. Beam search with negative cache enabled will run normally.
         Though, this is a pathological case (e.g. only 90 experts in an oversized 100x100 grid) that should be avoided.

    :param kwargs: any other params will be forwarded to DHTNode upon creation

    Each expert has an identifier in the form of {prefix}.{i}.{j}.{...}, e.g. "ffn_expert.98.76.54.32.10"
    An expert identifier consists of:

        * optional prefix that determines expert role, experiment name, etc.
        * one or more integers that determine that expert's position in an N-dimensional grid

    A hivemind.Server can ``DHT.declare_experts(expert_uids: List[str])`` to make its experts visible to everyone.
    When declaring experts, DHT will store each expert's uid and all its prefixes until :expiration: (specified at init)
    For instance, declaring "ffn_expert.98.76.54.32.10" will store the following keys in a DHT:
    ``"ffn_expert.98", "ffn_expert.98.76", "ffn_expert.98.76.54", ..., "ffn_expert.98.76.54.32.10"``

    In order to enable fast beam search, DHT maintains dictionaries of all active suffixes for every prefix
    (e.g. "ffn_expert.98": {76: ffn_expert.98.76...., 123: ffn_expert.98.123..., 225: ffn_expert.98.225....}))

    RemoteMixtureOfExperts can use these prefixes to find top-k most suitable experts with a left-to-right beam search.
    For instance, consider RemoteMixtureOfExperts with prefix "ffn_expert" and grid size [100, 100, 100, 100, 100].
    This MoE can query all experts with that prefix and arbitrary indices in 0...99 along each dimension.
    However, not every expert in such 100^5 grid can be alive at a given moment of time (the grid size is redundant).
    In order to find k best "alive" experts, MoE first ranks indices along the first dimension with its gating function.
    It can then check which of those indices correspond to "alive" experts by querying keys such as "ffn_expert.98".

    After selecting k best indices along first dimension, MoE moves to the second dimension.
    It can find top-k index pairs (e.g. "expert.98.76") that use one of k best indices from the previous step.
    This beam search explores one additional dimension per step and finds k best experts from across the DHT
    in O(k * num_dimensions * dimension_size) time depending on the chosen grid dimensions.
    """

    def __init__(self, listen_on: Endpoint = "0.0.0.0:*", initial_peers: Sequence[Endpoint] = (), *, start: bool,
                 daemon: bool = True, max_workers: Optional[int] = None, parallel_rpc: Optional[int] = None,
                 receiver_threads: int = 1, negative_caching: bool = True, expiration: float = 300, **kwargs):
        super().__init__()
        assert not isinstance(initial_peers, str), "please specify a list/tuple of initial peers (even if there's one)"
        self.listen_on, self.initial_peers, self.kwargs = listen_on, initial_peers, kwargs
        self.receiver_threads, self.max_workers, self.parallel_rpc = receiver_threads, max_workers, parallel_rpc
        self.expiration, self.negative_caching = expiration, negative_caching
        self._port = mp.Value(ctypes.c_int32, 0)  # initialized after dht starts
        self._pipe, self.pipe = mp.Pipe(duplex=True)
        self.ready = mp.Event()
        self.daemon = daemon
        if start:
            self.run_in_background(await_ready=True)

    def run(self) -> None:
        """ Serve DHT forever. This function will not return until DHT node is shut down """
        loop = switch_to_uvloop()
        pipe_awaiter = ThreadPoolExecutor(self.receiver_threads)

        async def _run():
            node = await DHTNode.create(
                initial_peers=list(self.initial_peers), listen_on=self.listen_on, parallel_rpc=self.parallel_rpc,
                num_workers=self.max_workers or 1, **self.kwargs)
            if node.port is not None:
                self._port.value = node.port
            self.ready.set()

            while True:
                method, args, kwargs = await loop.run_in_executor(pipe_awaiter, self._pipe.recv)
                asyncio.create_task(getattr(self, method)(node, *args, **kwargs))

        try:
            loop.run_until_complete(_run())
        except KeyboardInterrupt:
            logger.debug("Caught KeyboardInterrupt, shutting down")

    def run_in_background(self, await_ready=True, timeout=None):
        """
        Starts DHT in a background process. if await_ready, this method will wait until background dht
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError("Server didn't notify .ready in {timeout} seconds")

    def shutdown(self) -> None:
        """ Shut down a running dht process """
        if self.is_alive():
            self.terminate()
        else:
            logger.warning("DHT shutdown has no effect: dht process is already not alive")

    @property
    def port(self) -> Optional[int]:
        return self._port.value if self._port.value != 0 else None

    def get(self, key: DHTKey, latest: bool = False, return_future: bool = False, **kwargs
            ) -> Union[Optional[ValueWithExpiration[DHTValue]], MPFuture]:
        """
        Search for a key across DHT and return either first or latest entry (if found).
        :param key: same key as in node.store(...)
        :param latest: if True, finds the latest value, otherwise finds any non-expired value (which is much faster)
        :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
        :param kwargs: parameters forwarded to DHTNode.get_many_by_id
        :returns: (value, expiration time); if value was not found, returns None
        """
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_get', [], dict(key=key, latest=latest, future=_future, **kwargs)))
        return future if return_future else future.result()

    async def _get(self, node: DHTNode, key: DHTKey, latest: bool, future: MPFuture, **kwargs):
        try:
            result = await node.get(key, latest=latest, **kwargs)
            if not future.done():
                future.set_result(result)
        except BaseException as e:
            if not future.done():
                future.set_exception(e)
            raise

    def store(self, key: DHTKey, value: DHTValue, expiration_time: DHTExpiration,
              subkey: Optional[Subkey] = None, return_future: bool = False, **kwargs) -> Union[bool, MPFuture]:
        """
        Find num_replicas best nodes to store (key, value) and store it there until expiration time.

        :param key: msgpack-serializable key to be associated with value until expiration.
        :param value: msgpack-serializable value to be stored under a given key until expiration.
        :param expiration_time: absolute time when the entry should expire, based on hivemind.get_dht_time()
        :param subkey: if specified, add a value under that subkey instead of overwriting key (see DHTNode.store_many)
        :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
        :returns: True if store succeeds, False if it fails (due to no response or newer value)
        """
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_store', [], dict(key=key, value=value, expiration_time=expiration_time, subkey=subkey,
                                           future=_future, **kwargs)))
        return future if return_future else future.result()

    async def _store(self, node: DHTNode, key: DHTKey, value: DHTValue, expiration_time: DHTExpiration,
                     subkey: Optional[Subkey], future: MPFuture, **kwargs):
        try:
            result = await node.store(key, value, expiration_time, subkey=subkey, **kwargs)
            if not future.done():
                future.set_result(result)
        except BaseException as e:
            if not future.done():
                future.set_exception(e)
            raise

    def run_coroutine(self, coro: Callable[[DHT, DHTNode], Awaitable[ReturnType]],
                      return_future: bool = False) -> Union[ReturnType, MPFuture[ReturnType]]:
        """
        Execute an asynchronous function on a DHT participant and return results. This is meant as an interface
         for running custom functions DHT for special cases (e.g. declare experts, beam search)

        :param coro: async function to be executed. Receives 2 arguments: this DHT daemon and a running DHTNode
        :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
        :returns: coroutine outputs or MPFuture for these outputs
        :note: the coroutine will be executed inside the DHT process. As such, any changes to global variables or
          DHT fields made by this coroutine will not be accessible from the host process.
        :note: all time-consuming operations in coro should be asynchronous (e.g. asyncio.sleep instead of time.sleep)
          or use asyncio.get_event_loop().run_in_executor(...) to prevent coroutine from blocking background DHT tasks
        :note: when run_coroutine is called with wait=False, MPFuture can be cancelled to interrupt the task.
        """
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_run_coroutine', [], dict(coro=coro, future=_future)))
        return future if return_future else future.result()

    async def _run_coroutine(self, node: DHTNode, coro: Callable[[DHT, DHTNode], Awaitable[ReturnType]],
                             future: MPFuture[ReturnType]):
        main_task = asyncio.create_task(coro(self, node))
        cancel_task = asyncio.create_task(await_cancelled(future))
        try:
            await asyncio.wait({main_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
            if future.cancelled():
                main_task.cancel()
            else:
                future.set_result(await main_task)
        except BaseException as e:
            if not future.done():
                future.set_exception(e)

    def get_visible_address(self, num_peers: Optional[int] = None, peers: Sequence[Endpoint] = ()) -> Hostname:
        """
        Get this machine's visible address by requesting other peers or using pre-specified network addresses.
        If no parameters are specified, this function will check for manual endpoint; if unavailable, ask 1 random peer.

        :param num_peers: if specified, ask multiple peers and check that they perceive the same endpoint
        :param peers: if specified, ask these exact peers instead of choosing random known peers
        :note: if this node has no known peers in routing table, one must specify :peers: manually
        """
        assert num_peers is None or peers == (), "please specify either a num_peers or the list of peers, not both"
        assert not isinstance(peers, str) and isinstance(peers, Sequence), "Please send a list / tuple of endpoints"
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_get_visible_address', [], dict(num_peers=num_peers, peers=peers, future=_future)))
        return future.result()

    async def _get_visible_address(self, node: DHTNode, num_peers: Optional[int], peers: Sequence[Endpoint],
                                   future: Optional[MPFuture]):
        if not peers and (num_peers or not node.protocol.node_info.endpoint):
            # if we can't resolve the endpoint locally, ask one random peer
            peers_and_endpoints = node.protocol.routing_table.get_nearest_neighbors(
                DHTID.generate(), num_peers or 1, exclude=node.node_id)
            peers = tuple(endpoint for node_id, endpoint in peers_and_endpoints)

        chosen_address = None
        if peers:
            possible_endpoints: Sequence[Optional[Endpoint]] = await asyncio.gather(*(
                node.protocol.get_outgoing_request_endpoint(peer) for peer in peers))

            for endpoint in possible_endpoints:
                if endpoint is None:
                    continue
                address = strip_port(endpoint)
                if chosen_address is not None and address != chosen_address:
                    logger.warning("At least two peers returned different visible addresses for this node:"
                                   f"{address} and {chosen_address} (keeping the former one)")
                else:
                    chosen_address = address

            if chosen_address is None:
                logger.warning(f"None of the selected peers responded with an address ({peers})")

        if node.protocol.node_info.endpoint:
            address = strip_port(node.protocol.node_info.endpoint)
            if chosen_address is not None and address != chosen_address:
                logger.warning(f"Node was manually given endpoint {address} , but other peers report {chosen_address}")
            chosen_address = chosen_address or address

        if chosen_address:
            future.set_result(chosen_address)
        else:
            future.set_exception(ValueError(f"Can't get address: DHT node has no peers and no public endpoint."
                                            f" Please ensure the node is connected or specify peers=... manually."))

    @deprecated(version='0.9.5', reason="dht.declare_experts is deprecated, please use hivemind.declare_experts.")
    def declare_experts(self, uids, endpoint, wait: bool = True):
        from hivemind.client.dht_ops import declare_experts
        return declare_experts(self, uids, endpoint, wait=wait)

    @deprecated(version='0.9.5', reason="dht.get_experts is deprecated, please use hivemind.get_experts.")
    def get_experts(self, uids: List[ExpertUID], expiration_time: Optional[DHTExpiration] = None,
                    return_future: bool = False) -> List[Optional[RemoteExpert]]:
        from hivemind.client.dht_ops import get_experts
        return get_experts(self, uids, expiration_time, return_future)

    def get_initial_beam(self, prefix: ExpertPrefix, scores: Sequence[float], beam_size: int,
                         num_workers: Optional[int] = None, return_future: bool = False
                         ) -> List[Tuple[Score, ExpertPrefix, Dict[Coordinate, UidEndpoint]]]:
        """
        :param prefix: search for experts whose uids start with this prefix
        :param scores: prefer suffix coordinates that have highest scores
        :param beam_size: select this many active suffixes with highest scores
        :param num_workers: maintain up to this many concurrent DHT searches
        :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
        :returns: a list of up to beam_size tuples of (prefix score, prefix itself, dict{suffix: example expert})
        """
        assert is_valid_prefix(prefix), f"prefix '{prefix}' is invalid, it must follow {PREFIX_PATTERN.pattern}"
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_get_initial_beam', [], dict(prefix=prefix, scores=tuple(scores), beam_size=beam_size,
                                                      num_workers=num_workers, future=_future)))
        return future if return_future else future.result()

    async def _get_initial_beam(self, node, prefix: ExpertPrefix, beam_size: int, scores: Tuple[float, ...],
                                num_workers: Optional[int] = None, future: Optional[MPFuture] = None
                                ) -> List[Tuple[Score, ExpertPrefix, Dict[Coordinate, UidEndpoint]]]:
        num_workers = num_workers or self.max_workers or beam_size
        beam: List[Tuple[Score, ExpertPrefix, Dict[Coordinate, UidEndpoint]]] = []
        unattempted_indices: List[Coordinate] = sorted(range(len(scores)), key=scores.__getitem__)  # from worst to best
        pending_tasks: Deque[Tuple[Coordinate, ExpertPrefix, asyncio.Task]] = deque()

        while len(beam) < beam_size and (unattempted_indices or pending_tasks):
            # dispatch additional tasks
            while unattempted_indices and len(pending_tasks) < num_workers:
                next_index = unattempted_indices.pop()  # note: this is best unattempted index because of sort order
                next_best_prefix = f"{prefix}{next_index}{UID_DELIMITER}"
                pending_tasks.append((next_index, next_best_prefix, asyncio.create_task(node.get(next_best_prefix))))

            # await the next best prefix to be fetched
            pending_best_index, pending_best_prefix, pending_task = pending_tasks.popleft()
            try:
                maybe_prefix_data = await pending_task
                if maybe_prefix_data is not None and isinstance(maybe_prefix_data.value, dict):
                    successors = {coord: UidEndpoint(*match.value) for coord, match in maybe_prefix_data.value.items()
                                  if isinstance(coord, Coordinate) and isinstance(getattr(match, 'value', None), list)
                                  and len(match.value) == 2}
                    if successors:
                        beam.append((scores[pending_best_index], pending_best_prefix, successors))
                elif maybe_prefix_data is None and self.negative_caching:
                    logger.debug(f"DHT negative caching: storing a 'no prefix' entry for {pending_best_prefix}")
                    asyncio.create_task(node.store(pending_best_prefix, subkey=-1, value=None,
                                                   expiration_time=get_dht_time() + self.expiration))

            except asyncio.CancelledError:
                for _, pending_task in pending_tasks:
                    pending_task.cancel()
                raise
        if future:
            future.set_result(beam)
        return beam

    def get_active_successors(self, prefixes: List[ExpertPrefix], grid_size: Optional[int] = None,
                              num_workers: Optional[int] = None, return_future: bool = False
                              ) -> Dict[ExpertPrefix, Dict[Coordinate, UidEndpoint]]:
        """
        :param prefixes: a list of prefix for which to find active successor uids
        :param grid_size: if specified, only return successors if ther are in range [0, grid_size)
        :param num_workers: how many parallel workers to use for DHTNode.get_many
        :param return_future: if False (default), find and return successors. Otherwise return MPFuture and fill later.
        :returns: for every expert, return a dict{active_next_coordinate: (matching_expert_uid, matching_endpoint)}
        :note: if a prefix is not found, get_active_successors will return an empty dictionary for that prefix
        """
        assert not isinstance(prefixes, str), "Please send a list / tuple of expert prefixes."
        for prefix in prefixes:
            assert is_valid_prefix(prefix), f"prefix '{prefix}' is invalid, it must follow {PREFIX_PATTERN.pattern}"
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_get_active_successors', [], dict(
            prefixes=list(prefixes), grid_size=grid_size, num_workers=num_workers, future=_future)))
        return future if return_future else future.result()

    async def _get_active_successors(self, node: DHTNode, prefixes: List[ExpertPrefix], grid_size: Optional[int] = None,
                                     num_workers: Optional[int] = None, future: Optional[MPFuture] = None
                                     ) -> Dict[ExpertPrefix, Dict[Coordinate, UidEndpoint]]:
        grid_size = grid_size or float('inf')
        num_workers = num_workers or min(len(prefixes), self.max_workers or len(prefixes))
        dht_responses = await node.get_many(keys=prefixes, num_workers=num_workers)
        successors: Dict[ExpertPrefix, Dict[Coordinate, UidEndpoint]] = {}
        for prefix, found in dht_responses.items():
            if found and isinstance(found.value, dict):
                successors[prefix] = {coord: UidEndpoint(*match.value) for coord, match in found.value.items()
                                      if isinstance(coord, Coordinate) and 0 <= coord < grid_size
                                      and isinstance(getattr(match, 'value', None), list) and len(match.value) == 2}
            else:
                successors[prefix] = {}
                if found is None and self.negative_caching:
                    logger.debug(f"DHT negative caching: storing a 'no prefix' entry for {prefix}")
                    asyncio.create_task(node.store(prefix, subkey=-1, value=None,
                                                   expiration_time=get_dht_time() + self.expiration))
        if future:
            future.set_result(successors)
        return successors

    def find_best_experts(self, prefix: ExpertPrefix, grid_scores: Sequence[Sequence[float]], beam_size: int,
                          num_workers: Optional[int] = None, return_future: bool = False
                          ) -> Union[List[RemoteExpert], MPFuture]:
        """
        Find and return :beam_size: active experts with highest scores, use both local cache and DHT

        :param prefix: common prefix for all expert uids in grid
        :param grid_scores: scores predicted for each dimension in the grid,
        :type grid_scores: model scores for each grid dimension, list of arrays of shape grid_size[i]
        :param beam_size: how many best experts should beam search return
         After time_budget is reached, beam search won't search for more experts and instead fall back on local cache
         Please note that any queries that fall outside the budget will still be performed in background and cached
         for subsequent iterations as long as DHTNode.cache_locally is True
        :param num_workers: use up to this many concurrent workers to search DHT
        :param return_future: if set to True, returns MPFuture that can be awaited to get the actual result
        :returns: a list that contains *up to* k_best RemoteExpert instances
        """
        assert len(grid_scores) > 0 and beam_size > 0
        assert is_valid_prefix(prefix), f"prefix '{prefix}' is invalid, it must follow {PREFIX_PATTERN.pattern}"
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_find_best_experts', [], dict(prefix=prefix, grid_scores=list(map(tuple, grid_scores)),
                                                       beam_size=beam_size, num_workers=num_workers, future=_future)))
        return future if return_future else future.result()

    async def _find_best_experts(
            self, node: DHTNode, prefix: str, grid_scores: List[Tuple[float]], beam_size: int,
            num_workers: Optional[int] = None, future: Optional[MPFuture] = None, **kwargs) -> List[RemoteExpert]:
        num_workers = num_workers or min(beam_size, self.max_workers or beam_size)

        # form initial beam from top-k active L1 prefixes, each row is (score, uid prefix, possible suffixes)
        beam: List[Tuple[Score, ExpertPrefix, Dict[Coordinate, UidEndpoint]]] = await self._get_initial_beam(
            node, prefix, beam_size, grid_scores[0], min(beam_size, num_workers))

        best_experts_heap: List[Tuple[Score, UidEndpoint]] = []  # max-heap of expert uids/endpoints ordered by scores
        unique_experts: Set[ExpertUID] = set()

        for dim_index in range(1, len(grid_scores) - 1):
            for score, uid_endpoint in self._iterate_matching_experts(beam, grid_scores):
                if uid_endpoint.uid not in unique_experts:
                    push_and_maybe_pop = heapq.heappush if len(best_experts_heap) < beam_size else heapq.heappushpop
                    push_and_maybe_pop(best_experts_heap, (score, uid_endpoint))
                    unique_experts.add(uid_endpoint.uid)

            # form new beam using successors from the current beam
            dim_scores = grid_scores[dim_index]
            best_active_pairs: List[Tuple[Score, ExpertPrefix]] = heapq.nlargest(beam_size, (
                (prefix_score + dim_scores[next_coord], f"{prefix}{next_coord}{UID_DELIMITER}")
                for prefix_score, prefix, suffixes in beam for next_coord in suffixes.keys()
                if isinstance(next_coord, int) and 0 <= next_coord < len(dim_scores)))
            _, best_uid_prefixes = zip(*best_active_pairs)

            # search DHT for next step suffixes
            successors = await self._get_active_successors(node, best_uid_prefixes, num_workers=num_workers)
            beam = [(score, prefix, successors[prefix]) for score, prefix in best_active_pairs if successors[prefix]]
            if not beam:
                logger.warning(f"Beam search had to terminate prematurely because of empty beam (dim 0)")
                break

        # add best experts from the final beam
        for score, uid_endpoint in self._iterate_matching_experts(beam, grid_scores):
            if uid_endpoint.uid not in unique_experts:
                push_and_maybe_pop = heapq.heappush if len(best_experts_heap) < beam_size else heapq.heappushpop
                push_and_maybe_pop(best_experts_heap, (score, uid_endpoint))
                unique_experts.add(uid_endpoint.uid)

        best_experts = [RemoteExpert(*uid_endpoint) for score, uid_endpoint in sorted(best_experts_heap, reverse=True)]
        if future is not None:
            future.set_result(best_experts)
        return best_experts

    @staticmethod
    def _iterate_matching_experts(beam: List[Tuple[Score, ExpertPrefix, Dict[Coordinate, UidEndpoint]]],
                                  grid_scores: Sequence[Sequence[float]]) -> Iterator[Tuple[Score, UidEndpoint]]:
        """ iterate over all exemplar experts attached to current beam """
        for score, prefix, suffixes in beam:
            for next_coord, match in suffixes.items():
                if len(grid_scores) == 1 and next_coord == FLAT_EXPERT:
                    yield score, match
                elif isinstance(match.uid, ExpertUID) and match.uid.count(UID_DELIMITER) == len(grid_scores):
                    expert_coords = match.uid.split(UID_DELIMITER)[1:]
                    if all(coord.isdigit() and 0 <= int(coord) < len(grid_scores[i])
                           for i, coord in enumerate(expert_coords)):
                        expert_score = sum(scores[coord] for scores, coord in zip(grid_scores, map(int, expert_coords)))
                        yield expert_score, match
                    else:
                        logger.warning(f"Found incompatible expert coordinates: {expert_coords}")
                else:
                    logger.warning(f"Found incompatible expert UID: {match.uid}")

    def batch_find_best_experts(
            self, prefix: str, batch_grid_scores: Sequence[Sequence[Sequence[float]]], beam_size: int, *,
            workers_per_sample: Optional[int] = None, return_future=False) -> Union[List[List[RemoteExpert]], MPFuture]:
        """
        Find and return :beam_size: active experts with highest scores, use both local cache and DHT

        :param prefix: common prefix for all expert uids in grid
        :param batch_grid_scores: scores predicted for each batch example and each dimension in the grid,
        :type batch_grid_scores: list of arrays of shape (batch_size, grid_size[i])
        :param beam_size: how many best experts should beam search return
         After time_budget is reached, beam search won't search for more experts and instead fall back on local cache
         Please note that any queries that fall outside the budget will still be performed in background and cached
         for subsequent iterations as long as DHTNode.cache_locally is True
        :param workers_per_sample: use up to this many concurrent workers for every sample in batch
        :param return_future: if set to True, returns MPFuture that can be awaited to get the actual result
        :returns: a list that contains *up to* k_best RemoteExpert instances
        """
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_batch_find_best_experts', [], dict(prefix=prefix, batch_grid_scores=batch_grid_scores,
                                                             beam_size=beam_size, workers_per_sample=workers_per_sample,
                                                             future=_future)))
        return future if return_future else future.result()

    async def _batch_find_best_experts(
            self, node: DHTNode, prefix: str, batch_grid_scores: Sequence[Sequence[Tuple[float]]], beam_size: int,
            workers_per_sample: Optional[int] = None, future: Optional[MPFuture] = None) -> List[List[RemoteExpert]]:

        batch_grid_scores = [[tuple(grid_score[i]) for grid_score in batch_grid_scores]
                             for i in range(len(batch_grid_scores[0]))]
        coros = [self._find_best_experts(node, prefix, grid_scores, beam_size, workers_per_sample)
                 for grid_scores in batch_grid_scores]

        best_experts_batch = await asyncio.gather(*coros)
        if future is not None:
            future.set_result(best_experts_batch)
        return best_experts_batch
