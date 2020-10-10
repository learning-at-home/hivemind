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
import multiprocessing as mp
import warnings
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Sequence, OrderedDict as TOrderedDict, Union, Awaitable, Dict, Deque, Set

import uvloop
import heapq

from hivemind.client import RemoteExpert
from hivemind.dht.node import DHTNode, DHTID, DHTExpiration
from hivemind.dht.routing import get_dht_time
from hivemind.utils import MPFuture, Endpoint, get_logger

logger = get_logger(__name__)


class DHT(mp.Process):
    """
    High-level interface to hivemind.dht that is designed to allow RemoteMixtureOfExperts to select best experts.

    :param initial_peers: one or multiple endpoints pointing to active DHT peers. Similar format to listen_on.
    :param listen_on: an interface for incoming connections, e.g. "127.0.0.1:*", "0.0.0.0:1234" or "ipv6:[::]:*"
    :param start: if True, automatically starts the background process on creation. Otherwise await manual start
    :param daemon: if True, the background process is marked as daemon and automatically terminated after main process
    :param max_workers: declare_experts and get_experts will use up to this many parallel workers
        (but no more than one per key)
    :param expiration: experts declared from this node expire after this many seconds (default = 5 minutes)
    :param receiver_threads: uses this many threads to await on input pipe. Default = 1 should be enough in most cases
    :param kwargs: any other params will be forwarded to DHTNode upon creation

    Each expert has an identifier in the form of {prefix}.{i}.{j}.{...}, e.g. "ffn_expert.98.76.54.32.10"
    An expert identifier consists of:

        * optional prefix that determines expert role, experiment name, etc.
        * one or more integers that determine that expert's position in an N-dimensional grid

    A hivemind.Server can ``DHT.declare_experts(expert_uids: List[str])`` to make its experts visible to everyone.
    When declaring experts, DHT will store each expert's uid and all its prefixes until :expiration: (specified at init)
    For instance, declaring "ffn_expert.98.76.54.32.10" will store the following keys in a DHT:
    ``"ffn_expert", "ffn_expert.98", "ffn_expert.98.76", ..., "ffn_expert.98.76.54.32.10"``

    RemoteMixtureOfExperts can use these prefixes to find top-k most suitable experts with a left-to-right beam search.
    For instance, consider RemoteMixtureOfExperts with prefix "ffn_expert" and grid size [100, 100, 100, 100, 100].
    This MoE can query all experts with that prefix and arbitrary indices in 0...99 along each dimension.
    However, not every expert in such 100^5 grid can be alive at a given moment of time (the grid size is redundant).
    In order to find k best "alive" experts, MoE first ranks indices along the first dimension with its gating function.
    It can then check which of those indices correspond to "alive" experts by querying keys such as "ffn_expert.98".
    This is done using DHT.first_k_active function. After selecting k best indices along first dimension, MoE moves
    to the second dimension. It can find top-k pairs of indices (e.g. "expert.98.76") that start with one of k first
    indices from the previous step. Finally, MoE will use DHT.get_experts(uids: List[str]) search for specific experts.
    This beam search explores one additional dimension per step and finds k best experts from across the DHT
    in O(k / s * log(N)) average time where s is grid sparsity rate and N is the total number of experts.
    """
    UID_DELIMITER = '.'  # when declaring experts, DHT store all prefixes of that expert's uid, split over this prefix
    #  formally, prefixes = {uid.split(UID_DELIMITER)[:length] for length in range(1, uid.count(UID_DELIMITER) + 2)}

    def __init__(self, listen_on: Endpoint = "0.0.0.0:*", initial_peers: Sequence[Endpoint] = (), *, start: bool,
                 daemon: bool = True, max_workers: Optional[int] = None, parallel_rpc: Optional[int] = None,
                 receiver_threads: int = 1, expiration: float = 300, **kwargs):
        super().__init__()
        self.listen_on, self.initial_peers, self.kwargs = listen_on, initial_peers, kwargs
        self.receiver_threads, self.max_workers, self.parallel_rpc = receiver_threads, max_workers, parallel_rpc
        self.expiration = expiration
        self._port = mp.Value(ctypes.c_int32, 0)  # initialized after dht starts
        self._pipe, self.pipe = mp.Pipe(duplex=True)
        self.ready = mp.Event()
        self.daemon = daemon
        if start:
            self.run_in_background(await_ready=True)

    def run(self) -> None:
        """ Serve DHT forever. This function will not return until DHT node is shut down """
        if asyncio.get_event_loop().is_running():
            asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop
        uvloop.install()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
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

        loop.run_until_complete(_run())

    def run_in_background(self, await_ready=True, timeout=None):
        """
        Starts DHT in a background process. if await_ready, this method will wait until background dht
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError("Server didn't notify .ready in {timeout} seconds")

    def shutdown(self) -> None:
        """ Shuts down the dht process """
        if self.is_alive():
            self.terminate()
        else:
            warnings.warn("DHT shutdown has no effect: dht process is already not alive")

    @property
    def port(self) -> Optional[int]:
        return self._port.value if self._port.value != 0 else None

    def get_experts(self, uids: List[str], expiration_time: Optional[DHTExpiration] = None,
                    return_future=False) -> List[Optional[RemoteExpert]]:
        """
        :param uids: find experts with these ids from across the DHT
        :param expiration_time: if specified, return experts that expire no sooner than this (based on get_dht_time)
        :param return_future: if False (default), return when experts are returned. Otherwise return MPFuture.
        :returns: a list of [RemoteExpert if found else None]
        """
        assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_get_experts', [], dict(uids=uids, expiration_time=expiration_time, future=_future)))
        return future if return_future else future.result()

    async def _get_experts(
            self, node: DHTNode, uids: List[str], expiration_time: Optional[DHTExpiration], future: MPFuture):
        if expiration_time is None:
            expiration_time = get_dht_time()
        num_workers = len(uids) if self.max_workers is None else min(len(uids), self.max_workers)
        response = await node.get_many(uids, expiration_time, num_workers=num_workers)
        future.set_result([RemoteExpert(*expert_data['expert'][0])
                           if maybe_expiration_time else None and expert_data['expert'][1] is not None
                           for uid, (expert_data, maybe_expiration_time) in response.items()])

    def declare_experts(self, uids: List[str], endpoint: Endpoint, wait=True, timeout=None) -> Optional[List[bool]]:
        """
        Make experts visible to all DHT peers; update timestamps if declared previously.

        :param uids: a list of expert ids to update
        :param endpoint: endpoint that serves these experts, usually your server endpoint (e.g. "201.111.222.333:1337")
        :param wait: if True, awaits for declaration to finish, otherwise runs in background
        :param timeout: waits for the procedure to finish for up to this long, None means wait indefinitely
        :returns: if wait, returns a list of booleans, (True = store succeeded, False = store rejected)
        """
        assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
        future, _future = MPFuture.make_pair() if wait else (None, None)
        self.pipe.send(('_declare_experts', [], dict(uids=list(uids), endpoint=endpoint, future=_future)))
        if wait:
            return future.result(timeout)

    async def _declare_experts(self, node: DHTNode, uids: List[str], endpoint: Endpoint, future: Optional[MPFuture]):
        num_workers = len(uids) if self.max_workers is None else min(len(uids), self.max_workers)
        expiration_time = get_dht_time() + self.expiration
        #                 prefix---v next_dim     uid  endpoint
        unique_entries: Set[Tuple[str, str]] = set()
        data_to_store: List[Tuple[str, str, List[str, Endpoint]]] = []
        for uid in uids:  # first k entries are expert uids themselves
            data_to_store.append((uid, "expert", [uid, endpoint]))
        for uid in uids:  # and then, add all prefixes
            uid_parts = uid.split(self.UID_DELIMITER)
            for i in range(len(uid_parts) - 1):
                uid_prefix_i = self.UID_DELIMITER.join(uid_parts[:i + 1])
                if (uid_prefix_i, uid_parts[i + 1]) in unique_entries:
                    continue
                unique_entries.add((uid_prefix_i, uid_parts[i + 1]))
                data_to_store.append((uid_prefix_i, uid_parts[i + 1], [uid, endpoint]))

        keys, subkeys, values = map(list, zip(*data_to_store))
        store_ok = await node.store_many(keys, values, expiration_time, subkeys=subkeys, num_workers=num_workers)
        if future is not None:
            future.set_result([store_ok[key, subkey] for key, subkey in zip(keys, subkeys)])

    def find_best_experts(self, prefix: str, grid_scores: Sequence[Sequence[float]], beam_size: int, *,
                          return_future=False, **kwargs) -> Union[List[RemoteExpert], MPFuture]:
        """
        Find and return :beam_size: active experts with highest scores, use both local cache and DHT

        :param prefix: common prefix for all expert uids in grid
        :param grid_scores: scores predicted for each dimension in the grid,
        :type grid_scores: model scores for each grid dimension, list of arrays of shape grid_size[i]
        :param beam_size: how many best experts should beam search return
         After time_budget is reached, beam search won't search for more experts and instead fall back on local cache
         Please note that any queries that fall outside the budget will still be performed in background and cached
         for subsequent iterations as long as DHTNode.cache_locally is True
        :param return_future: if set to True, returns MPFuture that can be awaited to get the actual result
        :param kwargs: extra keyword parameters passed to DHTNode.get_many
        :returns: a list that contains *up to* k_best RemoteExpert instances
        """
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_find_best_experts', [], dict(prefix=prefix, grid_scores=list(map(tuple, grid_scores)),
                                                       beam_size=beam_size, future=_future, **kwargs)))
        return future if return_future else future.result()

    async def _find_best_experts(
            self, node: DHTNode, prefix: str, grid_scores: List[Tuple[float]], beam_size: int,
            max_workers: Optional[int] = None, future: Optional[MPFuture] = None, **kwargs) -> List[RemoteExpert]:
        max_workers: Optional[int] = max_workers or self.max_workers or beam_size

        # form initial beam from top-k active L1 prefixes, each row is (score, uid prefix, possible suffixes)
        beam: List[Tuple[float, str, Dict[str, List[str, Endpoint]]]] = await self._get_initial_beam(
            node, prefix, beam_size, grid_scores[0], num_workers=min(beam_size, max_workers))
        if not beam:
            logger.warning(f"Beam search had to terminate prematurely because of empty beam (dim 0)")
            return []

        for dim_index in range(1, len(grid_scores) - 1):
            # select beam_size best suffixes from current beam
            dim_scores = grid_scores[dim_index]
            best_active_pairs: List[Tuple[float, str]] = heapq.nlargest(beam_size, (
                (prefix_score + dim_scores[int(suffix_i)], f"{prefix}{self.UID_DELIMITER}{suffix_i}")
                for prefix_score, prefix, suffixes in beam for suffix_i in suffixes.keys()
                if str.isdecimal(suffix_i) and 0 <= int(suffix_i) < len(dim_scores)))

            # search DHT for next step suffixes
            _, best_uid_prefixes = zip(*best_active_pairs)
            dht_responses: Dict[str, Tuple[Dict[str, List[str, Endpoint]], DHTExpiration]] = await node.get_many(
                keys=best_uid_prefixes, num_workers=min(len(best_uid_prefixes), max_workers), **kwargs)
            if all(expiration is None for key, (_, expiration) in dht_responses.items()):
                logger.warning(f"Beam search had to terminate prematurely because of empty beam (dim {dim_index})")
                break
            beam: List[Tuple[float, str, Dict[str, List[str, Endpoint]]]] = [
                (prefix_score, prefix, dht_responses[prefix][0])  # add suffix dict if it is found
                for prefix_score, prefix in best_active_pairs if dht_responses[prefix][1] is not None]

        # select best experts from the final beam
        dim_scores = grid_scores[-1]
        final_best_pairs: List[Tuple[float, str, Endpoint]] = heapq.nlargest(beam_size, (
            (prefix_score + dim_scores[int(suffix_i)], uid, endpoint)
            for prefix_score, prefix, suffixes in beam for suffix_i, ((uid, endpoint), _) in suffixes.items()
            if str.isdecimal(suffix_i) and 0 <= int(suffix_i) < len(dim_scores)
        ))
        best_experts = [RemoteExpert(uid, endpoint) for score, uid, endpoint in final_best_pairs]
        if future is not None:
            future.set_result(best_experts)
        return best_experts

    def batch_find_best_experts(self, prefix: str, batch_grid_scores: Sequence[Sequence[Sequence[float]]], beam_size: int, *,
                                return_future=False, **kwargs) -> Union[List[RemoteExpert], MPFuture]:
        """
        Find and return :beam_size: active experts with highest scores, use both local cache and DHT

        :param prefix: common prefix for all expert uids in grid
        :param batch_grid_scores: scores predicted for each batch example and each dimension in the grid,
        :type batch_grid_scores: model scores for each example and each grid dimension,  list of arrays of shape (batch_size, grid_size[i])
        :param beam_size: how many best experts should beam search return
         After time_budget is reached, beam search won't search for more experts and instead fall back on local cache
         Please note that any queries that fall outside the budget will still be performed in background and cached
         for subsequent iterations as long as DHTNode.cache_locally is True
        :param return_future: if set to True, returns MPFuture that can be awaited to get the actual result
        :param kwargs: extra keyword parameters passed to DHTNode.get_many
        :returns: a list that contains *up to* k_best RemoteExpert instances
        """
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_batch_find_best_experts', [], dict(prefix=prefix, batch_grid_scores=batch_grid_scores,
                                                             beam_size=beam_size, future=_future, **kwargs)))
        return future if return_future else future.result()

    async def _batch_find_best_experts(
            self, node: DHTNode, prefix: str, batch_grid_scores: Sequence[Sequence[Tuple[float]]], beam_size: int,
            max_workers: Optional[int] = None, future: Optional[MPFuture] = None, **kwargs) -> List[List[RemoteExpert]]:

        batch_grid_scores = [[tuple(grid_score[i]) for grid_score in batch_grid_scores] for i in range(len(batch_grid_scores[0]))]
        coros = [self._find_best_experts(node, prefix, grid_scores, beam_size, max_workers, **kwargs) for grid_scores in batch_grid_scores]

        best_experts_batch = await asyncio.gather(*coros)
        if future is not None:
            future.set_result(best_experts_batch)
        return best_experts_batch

    async def _get_initial_beam(self, node, prefix: str, beam_size: int, scores: Tuple[float, ...], num_workers: int
                                ) -> List[Tuple[float, str, Dict[str, List[str, Endpoint]]]]:
        """ Fetch a list of all active level-one prefixes of a given prefix. Used for beam search """
        beam: List[Tuple[float, str, Dict[str, List[str, Endpoint]]]] = []  # results will be stored here
        unattempted_indices: List[int] = sorted(range(len(scores)), key=scores.__getitem__)  # order: worst to best
        pending_tasks: Deque[Tuple[int, str, asyncio.Task]] = deque()  # up to num_workers concurrent get tasks

        while len(beam) < beam_size and (unattempted_indices or pending_tasks):
            # dispatch additional tasks
            while unattempted_indices and len(pending_tasks) < num_workers:
                next_index = unattempted_indices.pop()  # note: this is best unattempted index because of sort order
                next_best_prefix = f"{prefix}{self.UID_DELIMITER}{next_index}"
                pending_tasks.append((next_index, next_best_prefix, asyncio.create_task(node.get(next_best_prefix))))

            # await the next best prefix to be fetched
            pending_best_index, pending_best_prefix, pending_task = pending_tasks.popleft()
            try:
                maybe_prefix_data, maybe_expiration_time = await pending_task
                if maybe_expiration_time is not None:
                    beam.append((scores[pending_best_index], pending_best_prefix, maybe_prefix_data))
            except asyncio.CancelledError:
                for _, pending_task in pending_tasks:
                    pending_task.cancel()
                raise
        return beam

    def first_k_active(
            self, uid_prefixes: List[str], k: int, max_prefetch: int = 1, chunk_size: Optional[int] = None,
            return_future=False) -> Union[TOrderedDict[str, RemoteExpert], Awaitable[TOrderedDict[str, RemoteExpert]]]:
        """
        Find k prefixes with active experts; may return less if there aren't enough; used for DMoE beam search

        :param uid_prefixes: a list of uid prefixes ordered from highest to lowest priority
        :param k: return at most *this many* active prefixes
        :param max_prefetch: pre-dispatch up to *this many* tasks (each for chunk_size experts)
        :param chunk_size: dispatch this many requests in one task
        :param return_future: if False (default), return when experts are returned. Otherwise return MPFuture.
        :returns: a ordered dict{uid_prefix -> RemoteExpert} mapping at most :k: prefixes to matching experts
            The keys in the returned dict are ordered same as in uid_prefixes.
        """
        logger.warning("first_k_active is deprecated and will be removed in 0.8.6")
        assert not isinstance(uid_prefixes, str), "please provide a list/tuple of prefixes as the first argument"
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_first_k_active', [],
                        dict(uid_prefixes=uid_prefixes, k=k, max_prefetch=max_prefetch,
                             chunk_size=chunk_size or k, future=_future)))
        return future if return_future else future.result()

    async def _first_k_active(
            self, node: DHTNode, uid_prefixes: List[str], k: int, max_prefetch: int, chunk_size: int, future: MPFuture):
        num_workers_per_chunk = min(chunk_size, self.max_workers or chunk_size)
        total_chunks = (len(uid_prefixes) - 1) // chunk_size + 1
        found: List[Tuple[str, RemoteExpert]] = []

        pending_tasks = deque(
            asyncio.create_task(node.get_many(uid_prefixes[chunk_i * chunk_size: (chunk_i + 1) * chunk_size],
                                              num_workers=num_workers_per_chunk))
            for chunk_i in range(min(max_prefetch + 1, total_chunks))
        )  # pre-dispatch first task and up to max_prefetch additional tasks

        for chunk_i in range(total_chunks):
            # parse task results in chronological order, launch additional tasks on demand
            response = await pending_tasks.popleft()
            for uid_prefix in uid_prefixes[chunk_i * chunk_size: (chunk_i + 1) * chunk_size]:
                maybe_expert_data, maybe_expiration_time = response[uid_prefix]
                if maybe_expiration_time is not None and len(maybe_expert_data) > 0:  # found active peer
                    found.append((uid_prefix, RemoteExpert(*next(iter(maybe_expert_data.values()))[0])))
                    # if we found enough active experts, finish immediately
                    if len(found) >= k:
                        break
            if len(found) >= k:
                break

            pre_dispatch_chunk_i = chunk_i + len(pending_tasks) + 1
            if pre_dispatch_chunk_i < total_chunks:
                pending_tasks.append(asyncio.create_task(node.get_many(
                    uid_prefixes[pre_dispatch_chunk_i * chunk_size: (pre_dispatch_chunk_i + 1) * chunk_size],
                    num_workers=num_workers_per_chunk)))

        for task in pending_tasks:
            task.cancel()

        # return k active prefixes or as many as we could find
        future.set_result(OrderedDict(found))
