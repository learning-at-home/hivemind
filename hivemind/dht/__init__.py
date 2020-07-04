"""
This sub-module implements a node in a Kademlia-based DHT. The code is organized as follows:

 * **class DHT (__init__.py)** - high-level class for model training. Runs DHTNode in a background process.
 * **class DHTNode (node.py)** - an asyncio implementation of dht server, stores AND gets keys.
 * **class DHTProtocol (protocol.py)** - an RPC protocol to request data from dht nodes.
 * **async def traverse_dht (traverse.py)** - a search algorithm that crawls DHT peers.

The code in this module is a modified version of Kademlia DHT  https://github.com/bmuller/kademlia
Brian, if you're reading this: THANK YOU! you're awesome :)
"""
import asyncio
import ctypes
import multiprocessing as mp
import warnings
from typing import List, Optional
import uvloop

from .node import DHTNode, DHTID, DHTExpiration
from .routing import get_dht_time

from ..client import RemoteExpert
from ..utils import SharedFuture, find_open_port, Endpoint, Port, run_in_background, LOCALHOST


class DHT(mp.Process):
    """
    A high-level interface to hivemind DHT. Runs a dht node in a background process.

    :param initial_peers: one or multiple pairs of (host, port) pointing to active DHT peers. Default: no peers
    :param listen_on: an interface for incoming connections, e.g. "127.0.0.1:*", "0.0.0.0:1234" or "ipv6:[::]:*"
    :param start: if True, automatically starts the background process on creation. Otherwise await manual start
    :param daemon: if True, the background process is marked as daemon and automatically terminated after main process
    :param max_workers: declare_experts and get_experts will use up to this many parallel workers
        (but no more than one per key)
    :param kwargs: any other params will be forwarded to DHTNode upon creation
    """
    UID_DELIMETER = '.'  # splits expert uids over this delimeter
    EXPIRATION = 120  # anything written to DHT is considered expired after this many seconds
    make_key = "{}::{}".format

    def __init__(self, *initial_peers: Endpoint, listen_on: Endpoint = "0.0.0.0:*", start: bool, daemon: bool = True,
                 max_workers: Optional[int] = None, parallel_rpc: Optional[int] = None, **kwargs):
        super().__init__()
        self.listen_on, self.initial_peers, self.kwargs = listen_on, initial_peers, kwargs
        self.max_workers, self.parallel_rpc = max_workers, parallel_rpc
        self._port = mp.Value(ctypes.c_int32, 0)  # initialized after server starts
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
        future, _future = SharedFuture.make_pair()
        self.pipe.send(('_get_experts', [], dict(uids=uids, expiration=expiration, future=_future)))
        return future.result()

    def _get_experts(self, uids: List[str], expiration: Optional[DHTExpiration], future: SharedFuture):
        loop = asyncio.get_event_loop()
        expiration = expiration or get_dht_time()
        num_workers = len(uids) if self.max_workers is None else min(len(uids), self.max_workers)
        keys = [self.make_key('expert', uid) for uid in uids]

        response = asyncio.run_coroutine_threadsafe(
            self.node.get_many(keys, expiration, num_workers=num_workers), loop).result()

        experts: List[Optional[RemoteExpert]] = [None] * len(uids)
        for i, (key, uid) in enumerate(zip(keys, uids)):
            maybe_result, maybe_expiration = response[key]
            if maybe_expiration is not None:  # if we found a value
                experts[i] = RemoteExpert(uid=uid, host=maybe_result[0], port=maybe_result[1])

        future.set_result(experts)

    def declare_experts(self, uids: List[str], addr, port, wait=True, timeout=None) -> Optional[List[bool]]:
        """
        Make experts available to DHT; update timestamps if already available

        :param uids: a list of expert ids to update
        :param addr: hostname that can be used to call this expert
        :param port: port that can be used to call this expert
        :param wait: if True, awaits for declaration to finish, otherwise runs in background
        :param timeout: waits for the procedure to finish, None means wait indeninitely
        :returns: if wait, returns a list of booleans, (True = store succeeded, False = store rejected)
        """
        future, _future = SharedFuture.make_pair() if wait else (None, None)
        self.pipe.send(('_declare_experts', [], dict(uids=list(uids), addr=addr, port=port, future=_future)))
        if wait:
            return future.result(timeout)

    def _declare_experts(self, uids: List[str], addr: str, port: int, future: Optional[SharedFuture]):
        assert self.node is not None, "This method should only be accessed from inside .run method"
        num_workers = len(uids) if self.max_workers is None else min(len(uids), self.max_workers)
        loop = asyncio.get_event_loop()
        expiration_time = get_dht_time() + self.EXPIRATION
        unique_prefixes = set()
        coroutines = []

        keys, values = [], []
        for uid in uids:
            uid_parts = uid.split(self.UID_DELIMETER)
            keys.append(self.make_key('expert', uid))
            values.append((addr, port))
            unique_prefixes.update([self.UID_DELIMETER.join(uid_parts[:i + 1]) for i in range(len(uid_parts))])

        for prefix in unique_prefixes:
            keys.append(self.make_key('prefix', prefix))
            values.append(True)

        store_ok = asyncio.run_coroutine_threadsafe(
            self.node.store_many(keys, values, expiration_time, num_workers=num_workers), loop
        ).result()
        if future is not None:
            future.set_result([store_ok[key] for key in keys])

    def first_k_active(self, prefixes: List[str], k: int, max_prefetch=None):
        """
        Find k prefixes with active experts; may return less if there aren't enough; used for DMoE beam search

        :param prefixes: a list of uid prefixes ordered from highest to lowest priority
        :param k: return at most *this many* active prefixes
        :param max_prefetch: pre-dispatch up to *this many* asynchronous expert requests, defaults to pre-dispatch = k
        :returns: a list of at most :k: prefixes that have at least one active expert each;
        """
        assert isinstance(prefixes, (list, tuple)), "please provide a list/tuple of prefixes as the first argument"
        future, _future = SharedFuture.make_pair()
        self.pipe.send(('_first_k_active', [],
                        dict(prefixes=prefixes, k=k, max_prefetch=max_prefetch or k, future=_future)))
        return future.result()

    def _first_k_active(self, prefixes: List[str], k: int, max_prefetch: Optional[int], future: SharedFuture):
        assert self.node is not None, "This method should only be accessed from inside .run method"
        max_prefetch = max_prefetch or len(prefixes)
        loop = asyncio.get_event_loop()
        lookup_prefetch = [asyncio.run_coroutine_threadsafe(self.node.get(self.make_key('prefix', prefix)), loop)
                           for prefix in prefixes[:max_prefetch]]
        active_prefixes = []

        for i, prefix in enumerate(prefixes):
            _, maybe_expiration = lookup_prefetch[i].result()

            if maybe_expiration is not None:
                active_prefixes.append(prefix)
                if len(active_prefixes) >= k:
                    future.set_result(active_prefixes)
                    for task in lookup_prefetch[i:]:
                        task.cancel()
                    return

            # pre-dispatch the next request in line
            if len(lookup_prefetch) < len(prefixes):
                lookup_prefetch.append(
                    asyncio.run_coroutine_threadsafe(
                        self.node.get(self.make_key('prefix', prefixes[len(lookup_prefetch)])), loop))

        # could not find enough active prefixes; return what we can
        future.set_result(active_prefixes)
