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
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Iterable, Optional, Sequence, Union, Callable, Awaitable, TypeVar

from multiaddr import Multiaddr

from hivemind.dht.node import DHTNode, DHTID
from hivemind.dht.routing import DHTValue, DHTKey, Subkey
from hivemind.dht.validation import CompositeValidator, RecordValidatorBase
from hivemind.utils import MPFuture, get_logger, switch_to_uvloop, ValueWithExpiration, await_cancelled, DHTExpiration
from hivemind.utils.networking import Hostname, Endpoint, strip_port

logger = get_logger(__name__)

ReturnType = TypeVar('ReturnType')


class DHT(mp.Process):
    """
    A high-level interface to a hivemind DHT that runs a single DHT node in a background process.
    * hivemind servers periodically announce their experts via declare_experts (dht_handler.py)
    * trainers find most suitable experts via RemoteMixtureOfExperts (beam_search.py)

    :param p2p: instance of hivemind.p2p.P2P that will be used for communication.
      If None, DHTNode will create and manage its own P2P instance with given initial_peers and
      parameters from ``kwargs``
    :param initial_peers: multiaddrs of one or more active DHT peers (if you want to join an existing DHT)
    :param start: if True, automatically starts the background process on creation. Otherwise await manual start
    :param daemon: if True, the background process is marked as daemon and automatically terminated after main process
    :param max_workers: declare_experts and get_experts will use up to this many parallel workers
      (but no more than one per key)
    :param expiration: experts declared from this node expire after this many seconds (default = 5 minutes)
    :param shutdown_timeout: when calling .shutdown, wait for up to this many seconds before terminating
    :param kwargs: any other params will be forwarded to DHTNode and hivemind.p2p.P2P upon creation
    """
    _node: DHTNode

    def __init__(self, p2p: Optional[P2P] = None,
                 initial_peers: Optional[Sequence[Union[Multiaddr, str]]] = None,
                 *, start: bool, daemon: bool = True, max_workers: Optional[int] = None,
                 parallel_rpc: Optional[int] = None, record_validators: Iterable[RecordValidatorBase] = (),
                 shutdown_timeout: float = 3, **kwargs):
        super().__init__()

        self.p2p = p2p
        if not (initial_peers is None or (isinstance(initial_peers, Sequence) and
                                          all(isinstance(item, (Multiaddr, str)) for item in initial_peers))):
            raise TypeError('initial_peers should be of type Optional[Sequence[Union[Multiaddr, str]]]')
        self.initial_peers = initial_peers
        self.kwargs = kwargs
        self.max_workers = max_workers
        self.parallel_rpc = parallel_rpc

        self._record_validator = CompositeValidator(record_validators)
        self._inner_pipe, self._outer_pipe = mp.Pipe(duplex=True)
        self.shutdown_timeout = shutdown_timeout
        self.ready = mp.Event()
        self.daemon = daemon
        if start:
            self.run_in_background(await_ready=True)

    def run(self) -> None:
        """ Serve DHT forever. This function will not return until DHT node is shut down """
        loop = switch_to_uvloop()

        with ThreadPoolExecutor(max_workers=1) as pipe_awaiter:
            async def _run():
                self._node = await DHTNode.create(
                    p2p=self.p2p, initial_peers=self.initial_peers, parallel_rpc=self.parallel_rpc,
                    num_workers=self.max_workers or 1, record_validator=self._record_validator,
                    **self.kwargs)
                self.ready.set()

                while True:
                    method, args, kwargs = await loop.run_in_executor(pipe_awaiter, self._inner_pipe.recv)
                    task = asyncio.create_task(getattr(self, method)(*args, **kwargs))
                    if method == '_shutdown':
                        await task
                        break

            coro = _run()
            loop.run_until_complete(coro)

    def run_in_background(self, await_ready=True, timeout=None):
        """
        Starts DHT in a background process. if await_ready, this method will wait until background dht
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError(f"DHT didn't notify .ready in {timeout} seconds")

    def shutdown(self) -> None:
        """ Shut down a running dht process """
        if self.is_alive():
            self._outer_pipe.send(('_shutdown', [], {}))
            self.join(self.shutdown_timeout)
            if self.is_alive():
                logger.warning("DHT did not shut down within the grace period; terminating it the hard way.")
                self.terminate()

    async def _shutdown(self):
        await self._node.shutdown()

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
        future = MPFuture()
        self._outer_pipe.send(('_get', [], dict(key=key, latest=latest, future=future, **kwargs)))
        return future if return_future else future.result()

    async def _get(self, key: DHTKey, latest: bool, future: MPFuture, **kwargs):
        try:
            result = await self._node.get(key, latest=latest, **kwargs)
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
        future = MPFuture()
        self._outer_pipe.send(('_store', [], dict(key=key, value=value, expiration_time=expiration_time, subkey=subkey,
                                                  future=future, **kwargs)))
        return future if return_future else future.result()

    async def _store(self, key: DHTKey, value: DHTValue, expiration_time: DHTExpiration,
                     subkey: Optional[Subkey], future: MPFuture, **kwargs):
        try:
            result = await self._node.store(key, value, expiration_time, subkey=subkey, **kwargs)
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
        future = MPFuture()
        self._outer_pipe.send(('_run_coroutine', [], dict(coro=coro, future=future)))
        return future if return_future else future.result()

    async def _run_coroutine(self, coro: Callable[[DHT, DHTNode], Awaitable[ReturnType]],
                             future: MPFuture[ReturnType]):
        main_task = asyncio.create_task(coro(self, self._node))
        cancel_task = asyncio.create_task(await_cancelled(future))
        try:
            await asyncio.wait({main_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
            if future.cancelled():
                main_task.cancel()
            else:
                future.set_result(await main_task)
        except BaseException as e:
            logger.exception(f'Caught an exception when running a coroutine: {e}')
            if not future.done():
                future.set_exception(e)

    def add_validators(self, record_validators: Iterable[RecordValidatorBase]) -> None:
        if not self.ready.is_set():
            raise RuntimeError(
                "Can't append new validators before the DHT process has started. "
                "Consider adding them to the initial list via DHT.__init__(record_validators=...)")

        self.run_coroutine(partial(DHT._add_validators, record_validators=record_validators))

    async def _add_validators(
            self, node: DHTNode, record_validators: Iterable[RecordValidatorBase]) -> None:
        node.protocol.record_validator.extend(record_validators)

    def get_visible_maddrs(self, latest: bool = False) -> List[Multiaddr]:
        """
        Get multiaddrs of the current DHT node that should be accessible by other peers.

        :param latest: ask the P2P daemon to refresh the visible multiaddrs
        """

        return self.run_coroutine(partial(DHT._get_visible_maddrs, latest=latest))

    async def _get_visible_maddrs(self, node: DHTNode, latest: bool = False) -> List[Multiaddr]:
        return await node.get_visible_maddrs(latest=latest)

    def __del__(self):
        if self._parent_pid == os.getpid() and self.is_alive():
            self.shutdown()
