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
import asyncio
import ctypes
import heapq
import multiprocessing as mp
import re
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import List, Tuple, Optional, Sequence, Union, Dict, Deque, Set, NamedTuple, Any

import uvloop

from hivemind.client import RemoteExpert
from hivemind.dht.node import DHTNode, DHTID, DHTExpiration
from hivemind.dht.routing import get_dht_time, DHTValue
from hivemind.dht.storage import ValueWithExpiration
from hivemind.utils import MPFuture, Endpoint, get_logger

logger = get_logger(__name__)

ExpertUID, ExpertPrefix, NextCoordinate = str, str, int
UID_PATTERN = re.compile('^[0-9a-zA-Z_-]+(\.[0-9]+)+')  # e.g. ffn_expert.98.76.54 - prefix and coordinates
UID_DELIMITER = '.'  # when declaring experts, DHT store all prefixes of that expert's uid, split over this prefix
#  formally, prefixes = {uid.split(UID_DELIMITER)[:length] for length in range(1, uid.count(UID_DELIMITER) + 2)}


def is_valid_uid(maybe_uid: str) -> bool:
    return bool(UID_PATTERN.search(maybe_uid))


def split_uid(uid: ExpertUID) -> Tuple[ExpertUID, NextCoordinate]:
    pivot = uid.rindex(UID_DELIMITER)
    return uid[:pivot], int(uid[pivot + 1:])


class DHT(mp.Process):
    """
    High-level interface to hivemind.dht that is designed to allow RemoteMixtureOfExperts to select best experts.

    * hivemind servers periodically announce their experts via DHT.declare_experts
    * trainers find most suitable experts via DHT.find_best_experts

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

    def get_experts(self, uids: List[ExpertUID], expiration_time: Optional[DHTExpiration] = None,
                    return_future=False) -> List[Optional[RemoteExpert]]:
        """
        :param uids: find experts with these ids from across the DHT
        :param expiration_time: if specified, return experts that expire no sooner than this (based on get_dht_time)
        :param return_future: if False (default), return when experts are returned. Otherwise return MPFuture.
        :returns: a list of [RemoteExpert if found else None]
        """
        assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_get_experts', [], dict(uids=list(uids), expiration_time=expiration_time, future=_future)))
        return future if return_future else future.result()

    async def _get_experts(self, node: DHTNode, uids: List[ExpertUID], expiration_time: Optional[DHTExpiration],
                           future: MPFuture) -> List[Optional[RemoteExpert]]:
        if expiration_time is None:
            expiration_time = get_dht_time()
        num_workers = len(uids) if self.max_workers is None else min(len(uids), self.max_workers)
        found: Dict[ExpertUID, DHTValue] = await node.get_many(uids, expiration_time, num_workers=num_workers)

        experts: List[Optional[RemoteExpert]] = [None] * len(uids)
        for i, uid in enumerate(uids):
            if found[uid] is not None and isinstance(found[uid].value, Endpoint):
                experts[i] = RemoteExpert(uid, found[uid].value)

        future.set_result(experts)

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
        for uid in uids:
            assert is_valid_uid(uid), f"{uid} is not a valid expert uid. All uids must follow {UID_PATTERN.pattern}"
        future, _future = MPFuture.make_pair() if wait else (None, None)
        self.pipe.send(('_declare_experts', [], dict(uids=list(uids), endpoint=endpoint, future=_future)))
        if wait:
            return future.result(timeout)

    async def _declare_experts(
            self, node: DHTNode, uids: List[ExpertUID], endpoint: Endpoint, future: Optional[MPFuture]):
        num_workers = len(uids) if self.max_workers is None else min(len(uids), self.max_workers)
        expiration_time = get_dht_time() + self.expiration
        data_to_store: Dict[Tuple[ExpertPrefix, Optional[NextCoordinate]], Tuple[ExpertUID, Endpoint]] = {}
        for uid in uids:
            data_to_store[uid, None] = endpoint
            prefix = uid
            for i in range(prefix.count(UID_DELIMITER) - 1):
                prefix, next_coord = split_uid(prefix)
                data_to_store[prefix, next_coord] = uid, endpoint

        keys, maybe_subkeys, values = zip(*((key, subkey, value) for (key, subkey), value in data_to_store.items()))
        store_ok = await node.store_many(keys, values, expiration_time, subkeys=maybe_subkeys, num_workers=num_workers)
        if future is not None:
            future.set_result([store_ok[(key, subkey) if subkey else key] for key, subkey in zip(keys, maybe_subkeys)])

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
        # TODO warn user if indices are out of range on the _last_ level! (rationale: beam search may return <k results)

        for dim_index in range(1, len(grid_scores) - 1):
            # select beam_size best suffixes from current beam
            dim_scores = grid_scores[dim_index]
            best_active_pairs: List[Tuple[float, str]] = heapq.nlargest(beam_size, (
                (prefix_score + dim_scores[int(suffix_i)], f"{prefix}{self.UID_DELIMITER}{suffix_i}")
                for prefix_score, prefix, suffixes in beam for suffix_i in suffixes.keys()
                # TODO get rid of str.isdecimal
                if str.isdecimal(suffix_i) and 0 <= int(suffix_i) < len(dim_scores)))

            # search DHT for next step suffixes
            _, best_uid_prefixes = zip(*best_active_pairs)
            # TODO Tuple[Dict[str, List[str, Endpoint]], DHTExpiration] -> namedtuple
            dht_responses: Dict[str, Tuple[Dict[str, List[str, Endpoint]], DHTExpiration]] = await node.get_many(
                keys=best_uid_prefixes, num_workers=min(len(best_uid_prefixes), max_workers), **kwargs)
            if all(expiration is None for key, (_, expiration) in dht_responses.items()):
                logger.warning(f"Beam search had to terminate prematurely because of empty beam (dim {dim_index})")
                break
            beam = [(prefix_score, prefix, dht_responses[prefix][0])  # add suffix dict if it is found
                    for prefix_score, prefix in best_active_pairs if dht_responses[prefix][1] is not None]

        # select best experts from the final beam
        dim_scores = grid_scores[-1]
        # TODO use heap to harness all results, get rid of five-line expression
        final_best_pairs: List[Tuple[float, str, Endpoint]] = heapq.nlargest(beam_size, chain((
            (prefix_score + dim_scores[int(suffix_i)], uid, endpoint)
            for prefix_score, prefix, suffixes in beam for suffix_i, ((uid, endpoint), _) in suffixes.items()
            if str.isdecimal(suffix_i) and 0 <= int(suffix_i) < len(dim_scores)
        ), ((score, *suffixes['expert']) for score, _, suffixes in beam if 'expert' in suffixes)))
        best_experts = [RemoteExpert(uid, endpoint) for score, uid, endpoint in final_best_pairs]
        if future is not None:
            future.set_result(best_experts)
        return best_experts

    def batch_find_best_experts(
            self, prefix: str, batch_grid_scores: Sequence[Sequence[Sequence[float]]], beam_size: int, *,
            return_future=False, **kwargs) -> Union[List[List[RemoteExpert]], MPFuture]:
        """
        Find and return :beam_size: active experts with highest scores, use both local cache and DHT

        :param prefix: common prefix for all expert uids in grid
        :param batch_grid_scores: scores predicted for each batch example and each dimension in the grid,
        :type batch_grid_scores: list of arrays of shape (batch_size, grid_size[i])
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

        batch_grid_scores = [[tuple(grid_score[i]) for grid_score in batch_grid_scores]
                             for i in range(len(batch_grid_scores[0]))]
        coros = [self._find_best_experts(node, prefix, grid_scores, beam_size, max_workers, **kwargs)
                 for grid_scores in batch_grid_scores]

        best_experts_batch = await asyncio.gather(*coros)
        if future is not None:
            future.set_result(best_experts_batch)
        return best_experts_batch

    async def _get_initial_beam(self, node, prefix: str, beam_size: int, scores: Tuple[float, ...], num_workers: int
                                ) -> List[Tuple[float, str, Dict[str, List[str]]]]:
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
                maybe_prefix_data = await pending_task
                if maybe_prefix_data is not None:
                    beam.append((scores[pending_best_index], pending_best_prefix, maybe_prefix_data.value))
            except asyncio.CancelledError:
                for _, pending_task in pending_tasks:
                    pending_task.cancel()
                raise
        return beam
