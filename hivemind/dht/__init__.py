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
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Iterable, List, Optional, Sequence, Union, Callable, Awaitable, TypeVar

import hivemind
from hivemind.client import RemoteExpert
from hivemind.dht.node import DHTNode, DHTID, DHTExpiration
from hivemind.dht.routing import DHTValue, DHTKey, Subkey
from hivemind.dht.validation import CompositeValidator, RecordValidatorBase
from hivemind.utils.networking import Hostname, Endpoint, strip_port
from hivemind.utils import MPFuture, get_logger, switch_to_uvloop, ValueWithExpiration, await_cancelled, get_dht_time

logger = get_logger(__name__)

ReturnType = TypeVar('ReturnType')


class DHT(mp.Process):
    """
    A high-level interface to a hivemind DHT that runs a single DHT node in a background process.
    * hivemind servers periodically announce their experts via declare_experts (dht_handler.py)
    * trainers find most suitable experts via RemoteMixtureOfExperts (beam_search.py)

    :param initial_peers: one or multiple endpoints pointing to active DHT peers. Similar format to listen_on.
    :param listen_on: an interface for incoming connections, e.g. "127.0.0.1:*", "0.0.0.0:1234" or "ipv6:[::]:*"
    :param start: if True, automatically starts the background process on creation. Otherwise await manual start
    :param daemon: if True, the background process is marked as daemon and automatically terminated after main process
    :param max_workers: declare_experts and get_experts will use up to this many parallel workers
        (but no more than one per key)
    :param expiration: experts declared from this node expire after this many seconds (default = 5 minutes)
    :param kwargs: any other params will be forwarded to DHTNode upon creation
    """

    def __init__(self, listen_on: Endpoint = "0.0.0.0:*", initial_peers: Sequence[Endpoint] = (), *, start: bool,
                 daemon: bool = True, max_workers: Optional[int] = None, parallel_rpc: Optional[int] = None,
                 record_validators: Iterable[RecordValidatorBase] = (), **kwargs):
        super().__init__()
        assert not isinstance(initial_peers, str), "please specify a list/tuple of initial peers (even if there's one)"
        self.listen_on, self.initial_peers, self.kwargs = listen_on, initial_peers, kwargs
        self.max_workers, self.parallel_rpc = max_workers, parallel_rpc
        self._record_validator = CompositeValidator(record_validators)
        self._port = mp.Value(ctypes.c_int32, 0)  # initialized after dht starts
        self._pipe, self.pipe = mp.Pipe(duplex=True)
        self.ready = mp.Event()
        self.daemon = daemon
        if start:
            self.run_in_background(await_ready=True)

    def run(self) -> None:
        """ Serve DHT forever. This function will not return until DHT node is shut down """
        loop = switch_to_uvloop()

        with ThreadPoolExecutor(max_workers=1) as pipe_awaiter:
            async def _run():
                node = await DHTNode.create(
                    initial_peers=list(self.initial_peers), listen_on=self.listen_on, parallel_rpc=self.parallel_rpc,
                    num_workers=self.max_workers or 1, record_validator=self._record_validator,
                    **self.kwargs)
                if node.port is not None:
                    self._port.value = node.port
                self.ready.set()

                while True:
                    method, args, kwargs = await loop.run_in_executor(pipe_awaiter, self._pipe.recv)
                    asyncio.create_task(getattr(self, method)(node, *args, **kwargs))

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
