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
import warnings
from collections import deque
from typing import List, Optional, Sequence

import uvloop

from hivemind.client import RemoteExpert
from hivemind.dht.node import DHTNode, DHTID, DHTExpiration
from hivemind.dht.routing import get_dht_time
from hivemind.utils import MPFuture, Endpoint, run_in_background


class DHT(mp.Process):
    """
    High-level interface to hivemind.dht that is designed to allow RemoteMixtureOfExperts select best experts.

    :param initial_peers: one or multiple endpoints pointing to active DHT peers. Similar format to listen_on.
    :param listen_on: an interface for incoming connections, e.g. "127.0.0.1:*", "0.0.0.0:1234" or "ipv6:[::]:*"
    :param start: if True, automatically starts the background process on creation. Otherwise await manual start
    :param daemon: if True, the background process is marked as daemon and automatically terminated after main process
    :param max_workers: declare_experts and get_experts will use up to this many parallel workers
        (but no more than one per key)
    :param uid_delimiter: when declaring experts, DHT will also declare all prefixes of that expert's uid, defined as
        {uid.split(uid_delimiter)[:prefix_length] for prefix_length in range(1, len(uid.split(uid_delimiter) + 1))}
    :param expiration: experts declared from this node expire after this many seconds (default = 5 minutes)
    :param kwargs: any other params will be forwarded to DHTNode upon creation

    Each expert has an identifier in the form of {prefix}.{i}.{j}.{...}, e.g. "ffn_expert.98.76.54.32.10"
    An expert identifier consists of:

        * optional prefix that determines expert role, experiment name, etc.
        * one or more integers that determine that expert's position in an N-dimensional grid
        * delimiters, e.g. "." (uid_delimiter param) over which expert uid will be split to get prefixes (see below)

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

    def __init__(self, listen_on: Endpoint = "0.0.0.0:*", initial_peers: Sequence[Endpoint] = (), *, start: bool,
                 daemon: bool = True, max_workers: Optional[int] = None, parallel_rpc: Optional[int] = None,
                 uid_delimiter='.', expiration=300, **kwargs):
        super().__init__()
        self.listen_on, self.initial_peers, self.kwargs = listen_on, initial_peers, kwargs
        self.max_workers, self.parallel_rpc = max_workers, parallel_rpc
        self.uid_delimiter, self.expiration = uid_delimiter, expiration
        self._port = mp.Value(ctypes.c_int32, 0)  # initialized after dht starts
        self.node: Optional[DHTNode] = None  # initialized inside self.run only
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
        self.node: DHTNode = loop.run_until_complete(DHTNode.create(
            initial_peers=list(self.initial_peers), listen_on=self.listen_on, parallel_rpc=self.parallel_rpc,
            num_workers=self.max_workers or 1, **self.kwargs))
        self._port.value = self.node.port
        run_in_background(loop.run_forever)
        self.ready.set()

        while True:
            method, args, kwargs = self._pipe.recv()
            getattr(self, method)(*args, **kwargs)

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
            self.kill()
        else:
            warnings.warn("DHT shutdown has no effect: dht process is already not alive")

    @property
    def port(self) -> Optional[int]:
        return self._port.value if self._port.value != 0 else None

    def get_experts(self, uids: List[str], expiration=None) -> List[Optional[RemoteExpert]]:
        """
        :param uids: find experts with these ids from across the DHT
        :param expiration: returns experts that expire no sooner than this (based on get_dht_time), default = now
        :returns: a list of [RemoteExpert if found else None]
        """
        assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_get_experts', [], dict(uids=uids, expiration=expiration, future=_future)))
        return future.result()

    def _get_experts(self, uids: List[str], expiration: Optional[DHTExpiration], future: MPFuture):
        loop = asyncio.get_event_loop()
        expiration = expiration or get_dht_time()
        num_workers = len(uids) if self.max_workers is None else min(len(uids), self.max_workers)

        response = asyncio.run_coroutine_threadsafe(
            self.node.get_many(uids, expiration, num_workers=num_workers), loop).result()

        future.set_result([RemoteExpert(uid, maybe_endpoint) if maybe_expiration else None
                           for uid, (maybe_endpoint, maybe_expiration) in response.items()])

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

    def _declare_experts(self, uids: List[str], endpoint: Endpoint, future: Optional[MPFuture]):
        assert self.node is not None, "This method should only be accessed from inside .run method"
        num_workers = len(uids) if self.max_workers is None else min(len(uids), self.max_workers)
        loop = asyncio.get_event_loop()
        expiration_time = get_dht_time() + self.expiration

        data_to_store = {}
        for uid in uids:
            uid_parts = uid.split(self.uid_delimiter)
            for i in range(len(uid_parts)):
                uid_prefix_i = self.uid_delimiter.join(uid_parts[:i + 1])
                data_to_store[uid_prefix_i] = endpoint

        store_ok = asyncio.run_coroutine_threadsafe(
            self.node.store_many(*zip(*data_to_store.items()), expiration=expiration_time, num_workers=num_workers), loop
        ).result()
        if future is not None:
            future.set_result([store_ok[key] for key in data_to_store.keys()])

    def first_k_active(self, uid_prefixes: List[str], k: int, max_prefetch: int = 1, chunk_size: Optional[int] = None):
        """
        Find k prefixes with active experts; may return less if there aren't enough; used for DMoE beam search

        :param uid_prefixes: a list of uid prefixes ordered from highest to lowest priority
        :param k: return at most *this many* active prefixes
        :param max_prefetch: pre-dispatch up to *this many* tasks (each for chunk_size experts)
        :param chunk_size: dispatch this many requests in one task
        :returns: a list of at most :k: prefixes that have at least one active expert each;
        """
        assert not isinstance(uid_prefixes, str), "please provide a list/tuple of prefixes as the first argument"
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_first_k_active', [],
                        dict(uid_prefixes=uid_prefixes, k=k, max_prefetch=max_prefetch,
                             chunk_size=chunk_size or k, future=_future)))
        return future.result()

    def _first_k_active(self, uid_prefixes: List[str], k: int, max_prefetch: int, chunk_size: int, future: MPFuture):
        assert self.node is not None, "This method should only be accessed from inside .run method"
        loop = asyncio.get_event_loop()
        workers_per_chunk = min(chunk_size, self.max_workers or chunk_size)
        total_chunks = (len(uid_prefixes) - 1) // chunk_size + 1
        active_prefixes = []

        pending_tasks = deque(
            asyncio.run_coroutine_threadsafe(self.node.get_many(
                uid_prefixes[chunk_i * chunk_size: (chunk_i + 1) * chunk_size], num_workers=workers_per_chunk), loop)
            for chunk_i in range(min(max_prefetch + 1, total_chunks))
        )  # pre-dispatch first task and up to max_prefetch additional tasks

        for chunk_i in range(total_chunks):
            # parse task results in chronological order, launch additional tasks on demand
            response = pending_tasks.popleft().result()
            for uid_prefix in uid_prefixes[chunk_i * chunk_size: (chunk_i + 1) * chunk_size]:
                if response[uid_prefix][1] is not None:  # found active peer
                    active_prefixes.append(uid_prefix)
                    # if we found enough active experts, finish immediately
                    if len(active_prefixes) >= k:
                        break
            if len(active_prefixes) >= k:
                for task in pending_tasks:
                    task.cancel()
                break

            pre_dispatch_chunk_i = chunk_i + len(pending_tasks) + 1
            if pre_dispatch_chunk_i < total_chunks:
                pending_tasks.append(asyncio.run_coroutine_threadsafe(self.node.get_many(
                    uid_prefixes[pre_dispatch_chunk_i * chunk_size: (pre_dispatch_chunk_i + 1) * chunk_size],
                    num_workers=workers_per_chunk), loop))

        # return k active prefixes or as many as we could find
        future.set_result(active_prefixes)
