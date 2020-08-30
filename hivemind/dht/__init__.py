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
import multiprocessing as mp
import time
import warnings
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Sequence, OrderedDict as TOrderedDict

import uvloop
import torch

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
        future.set_result([RemoteExpert(**expert_data) if maybe_expiration_time else None
                           for uid, (expert_data, maybe_expiration_time) in response.items()])

    def declare_experts(self, uids: List[str], endpoint: Endpoint, wait=True, timeout=None) -> Optional[List[bool]]:
        """
        Make experts visible to all DHT peers; update timestamps if declared previously.

        :param uids: a list of expert ids to update
        :param endpoint: endpoint that serves these experts, usually your server endpoint (e.g. "201.111.222.333:1337")
        :param wait: if True, awaits for declaration to finish, otherwise runs in background
        :param timeout: waits for the procedure to finish, None means wait indeninitely
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

        data_to_store = {}
        for uid in uids:
            uid_parts = uid.split(self.UID_DELIMITER)
            for i in range(len(uid_parts)):
                uid_prefix_i = self.UID_DELIMITER.join(uid_parts[:i + 1])
                data_to_store[uid_prefix_i] = {'uid': uid, 'endpoint': endpoint}

        store_keys, store_values = zip(*data_to_store.items())
        store_ok = await node.store_many(store_keys, store_values, expiration_time, num_workers=num_workers)
        if future is not None:
            future.set_result([store_ok[key] for key in data_to_store.keys()])

    def first_k_active(self, uid_prefixes: List[str], k: int, max_prefetch: int = 1, chunk_size: Optional[int] = None,
                       time_budget: float = float('inf'), return_future=False) -> TOrderedDict[str, RemoteExpert]:
        """
        Find k prefixes with active experts; may return less if there aren't enough; used for DMoE beam search

        :param uid_prefixes: a list of uid prefixes ordered from highest to lowest priority
        :param k: return at most *this many* active prefixes
        :param max_prefetch: pre-dispatch up to *this many* tasks (each for chunk_size experts)
        :param chunk_size: dispatch this many requests in one task (default = k)
        :param time_budget: if specified, search for at most this much time and consider straggler experts inactive
        :param return_future: if set to True, returns MPFuture that can be awaited to get the actual result
        :returns: a ordered dict{uid_prefix -> RemoteExpert} mapping at most :k: prefixes to matching experts
            The keys in the returned dict are ordered same as in uid_prefixes.
        """
        assert not isinstance(uid_prefixes, str), "please provide a list/tuple of prefixes as the first argument"
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_first_k_active', [],
                        dict(uid_prefixes=uid_prefixes, k=k, max_prefetch=max_prefetch,
                             chunk_size=chunk_size or k, time_budget=time_budget, future=_future)))
        return future if return_future else future.result()

    async def _first_k_active(
            self, node: DHTNode, uid_prefixes: List[str], k: int, *, max_prefetch: int = 1,
            chunk_size: Optional[int] = None, time_budget: float = float('inf'),
            future: Optional[MPFuture] = None) -> TOrderedDict[str, RemoteExpert]:
        #TODO respect time_budget here
        chunk_size = chunk_size if chunk_size is not None else k
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
                if maybe_expiration_time is not None:  # found active peer
                    found.append((uid_prefix, RemoteExpert(**maybe_expert_data)))
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
        results: OrderedDict[str, RemoteExpert] = OrderedDict(found)
        if future:
            future.set_result(results)
        return results

    def find_best_experts(self, prefix: str, grid_scores: List[torch.Tensor], k_best: int, *,
                          time_budget: float = float('inf'), grid_indices: Optional[List[torch.Tensor]] = None,
                          return_future=False, **kwargs) -> List[RemoteExpert]:
        """
        Find and return k active experts with highest scores, use both local cache and DHT

        :param prefix: common prefix for all expert uids in grid
        :param grid_scores: scores predicted for each dimension in the grid,
        :type grid_scores: a sequence of tensors of shape[batch_size, grid_size[i]]
        :param grid_indices: optional, indices for each grid dimension. Default = 0, 1, ... len(grid_scores[i]) - 1

        :param k_best: how many best experts should beam search return
        :param time_budget: how much time beam_search is can spend on queries to other peers (default = unlimited)
         After time_budget is reached, beam search won't search for more experts and instead fall back on local cache
         Please note that any queries that fall outside the budget will still be performed in background and cached
         for subsequent iterations as long as DHTNode.cache_locally is True
        :param return_future: if set to True, returns MPFuture that can be awaited to get the actual result
        :param kwargs: extra keyword parameters passed to self.dht.first_k_active
        :returns: a list that contains *up to* k_best RemoteExpert instances
        """
        if grid_indices is None:
            grid_indices = [torch.arange(len(dim_scores)) for dim_scores in grid_scores]
        grid_scores = [dim_scores.cpu().detach() for dim_scores in grid_scores]
        grid_indices = [dim_indices.cpu() for dim_indices in grid_indices]
        assert len(grid_indices) == len(grid_scores), "grid_indices (if provided) must be of same length as grid_scores"
        assert all(dim_scores.ndim == 1 and dim_scores.shape == dim_indices.shape
                   for dim, dim_scores, dim_indices in enumerate(zip(grid_scores, grid_indices)))

        future, _future = MPFuture.make_pair()
        self.pipe.send(('_find_best_experts', [],
                        dict(prefix=prefix, grid_scores=grid_scores, grid_indices=grid_indices, k_best=k_best,
                             time_budget=time_budget, future=_future, **kwargs)))
        return future if return_future else future.result()

    async def _find_best_experts(self, node: DHTNode, prefix: str, grid_scores: List[torch.Tensor],
                                 grid_indices: List[torch.Tensor], k_best: int, time_budget: float = float('inf'),
                                 future: Optional[MPFuture] = None, **kwargs) -> List[RemoteExpert]:
        deadline_time = time.perf_counter() + time_budget
        beam_experts: List[RemoteExpert] = []
        beam: List[str] = [prefix]
        beam_scores = torch.zeros(1)

        for dim_index, dim_scores, dim_indices in enumerate(zip(grid_scores, grid_indices)):
            # create all possible successors from current beam and sort them by total score
            expanded_scores = beam_scores[:, None] + dim_scores[None, :]
            sorted_indices = [(flat_i // len(dim_scores), flat_i % len(dim_scores))
                              for flat_i in (-expanded_scores).flatten().argsort().numpy()]

            sorted_candidates = [f"{beam[row]}{self.UID_DELIMITER}{dim_indices[col]:d}" for row, col in sorted_indices]
            candidate_to_sorted_indices = dict(zip(sorted_candidates, sorted_indices))

            # select k best candidates according to scores but only those that are still active
            best_alive_prefixes: TOrderedDict[str, RemoteExpert] = await self._first_k_active(
                node, sorted_candidates, k=k_best, time_budget=deadline_time - time.perf_counter(), **kwargs)

            if not best_alive_prefixes:
                logger.warning(f"Grid is empty: found neither of {sorted_candidates}")
                break

            beam = list(best_alive_prefixes.keys())
            beam_scores = expanded_scores[tuple(zip(*map(candidate_to_sorted_indices.get, beam)))]
            beam_experts = list(best_alive_prefixes.values())

        if future:
            future.set_result(beam_experts)
        return beam_experts

    def batch_find_best_experts(self, prefix: str, grid_scores: List[torch.Tensor], k_best: int, *,
                                time_budget: float = float('inf'), grid_indices: Optional[List[torch.Tensor]],
                                return_future=False, **kwargs) -> List[RemoteExpert]:
        """
        Batch-parallel version of find_best_experts (see find_best_experts docstring for details)
        The only exception is that grid_scores must now be a list of 2d tensors [batch_size, grid_size[i]]
        :returns: a list of batch_size lists, each contains *up to* k_best RemoteExpert instances
        """
        if grid_indices is None:
            grid_indices = [torch.arange(len(dim_scores)) for dim_scores in grid_scores]
        grid_scores = [dim_scores.cpu().detach() for dim_scores in grid_scores]
        grid_indices = [dim_indices.cpu() for dim_indices in grid_indices]
        assert len(grid_indices) == len(grid_scores), "grid_indices (if provided) must be of same length as grid_scores"
        batch_size = len(grid_scores[0])
        assert all(dim_scores.ndim == 2 and dim_indices.ndim == 1 and len(dim_scores) == len(dim_indices) == batch_size
                   for dim, dim_scores, dim_indices in enumerate(zip(grid_scores, grid_indices)))
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_batch_find_best_experts', [],
                        dict(prefix=prefix, grid_scores=grid_scores, grid_indices=grid_indices, k_best=k_best,
                             time_budget=time_budget, future=_future, **kwargs)))
        return future if return_future else future.result()

    async def _batch_find_best_experts(self, node: DHTNode, prefix: str, grid_scores: List[torch.Tensor],
                                       grid_indices: List[torch.Tensor], k_best: int, time_budget: float = float('inf'),
                                       future: Optional[MPFuture] = None, **kwargs) -> List[List[RemoteExpert]]:
        batch_size = grid_scores[0].shape[0]
        results = await asyncio.gather(*[
            asyncio.create_task(
                self._find_best_experts(node, prefix, grid_scores=grid_scores_i, grid_indices=grid_indices,
                                        k_best=k_best, time_budget=time_budget, **kwargs)
            ) for grid_scores_i in map(list, zip(*grid_scores))
        ])
        if future:
            future.set_result(results)
        return results


