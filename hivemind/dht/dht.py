from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import signal
from functools import partial
from typing import Awaitable, Callable, Iterable, List, Optional, Sequence, TypeVar, Union

from multiaddr import Multiaddr

from hivemind.dht.node import DEFAULT_NUM_WORKERS, DHTNode
from hivemind.dht.routing import DHTKey, DHTValue, Subkey
from hivemind.dht.validation import CompositeValidator, RecordValidatorBase
from hivemind.p2p import P2P, PeerID
from hivemind.utils import MPFuture, get_logger, switch_to_uvloop
from hivemind.utils.timed_storage import DHTExpiration, ValueWithExpiration

logger = get_logger(__name__)
ReturnType = TypeVar("ReturnType")


class DHT(mp.Process):
    """
    A high-level interface to a hivemind DHT that runs a single DHT node in a background process.
    * hivemind servers periodically announce their experts via declare_experts (dht_handler.py)
    * trainers find most suitable experts via RemoteMixtureOfExperts (beam_search.py)

    :param initial_peers: multiaddrs of one or more active DHT peers (if you want to join an existing DHT)
    :param start: if True, automatically starts the background process on creation. Otherwise await manual start
    :param daemon: if True, the background process is marked as daemon and automatically terminated after main process
    :param num_workers: declare_experts and get_experts will use up to this many parallel workers
      (but no more than one per key)
    :param expiration: experts declared from this node expire after this many seconds (default = 5 minutes)
    :param record_validators: instances of RecordValidatorBase used for signing and validating stored records.
      The validators will be combined using the CompositeValidator class. It merges them when possible
      (according to their `.merge_with()` policies) and orders them according to the `.priority` properties.
    :param shutdown_timeout: when calling .shutdown, wait for up to this many seconds before terminating
    :param await_ready: if True, the constructor waits until the DHT process is ready to process incoming requests
    :param kwargs: any other params will be forwarded to DHTNode and hivemind.p2p.P2P upon creation
    """

    _node: DHTNode

    def __init__(
        self,
        initial_peers: Optional[Sequence[Union[Multiaddr, str]]] = None,
        *,
        start: bool,
        p2p: Optional[P2P] = None,
        daemon: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
        record_validators: Iterable[RecordValidatorBase] = (),
        shutdown_timeout: float = 3,
        await_ready: bool = True,
        **kwargs,
    ):
        self._parent_pid = os.getpid()
        self._origin_pid = os.getpid()
        super().__init__()

        if not (
            initial_peers is None
            or (
                isinstance(initial_peers, Sequence)
                and all(isinstance(item, (Multiaddr, str)) for item in initial_peers)
            )
        ):
            raise TypeError("initial_peers should be of type Optional[Sequence[Union[Multiaddr, str]]]")
        self.initial_peers = initial_peers
        self.kwargs = kwargs
        self.num_workers = num_workers

        self._record_validator = CompositeValidator(record_validators)
        self._inner_pipe, self._outer_pipe = mp.Pipe(duplex=True)
        self.shutdown_timeout = shutdown_timeout
        self._ready = MPFuture()
        self.daemon = daemon

        # These values will be fetched from the child process when requested
        self._peer_id = None
        self._client_mode = None
        self._p2p_replica = None

        self._daemon_listen_maddr = p2p.daemon_listen_maddr if p2p is not None else None

        if start:
            self.run_in_background(await_ready=await_ready)

    def run(self) -> None:
        """Serve DHT forever. This function will not return until DHT node is shut down"""

        loop = switch_to_uvloop()
        pipe_semaphore = asyncio.Semaphore(value=0)
        loop.add_reader(self._inner_pipe.fileno(), pipe_semaphore.release)

        async def _run():
            # Set SIG_IGN handler to SIGINT
            signal.signal(signal.SIGINT, signal.SIG_IGN)

            try:
                if self._daemon_listen_maddr is not None:
                    replicated_p2p = await P2P.replicate(self._daemon_listen_maddr)
                else:
                    replicated_p2p = None

                self._node = await DHTNode.create(
                    initial_peers=self.initial_peers,
                    num_workers=self.num_workers,
                    record_validator=self._record_validator,
                    p2p=replicated_p2p,
                    **self.kwargs,
                )
            except Exception as e:
                # Loglevel is DEBUG since normally the exception is propagated to the caller
                logger.debug(e, exc_info=True)
                self._ready.set_exception(e)
                return
            self._ready.set_result(None)

            while True:
                try:
                    await asyncio.wait_for(pipe_semaphore.acquire(), timeout=self._node.protocol.wait_timeout)
                except asyncio.TimeoutError:
                    pass
                if not self._inner_pipe.poll():
                    continue
                try:
                    method, args, kwargs = self._inner_pipe.recv()
                except (OSError, ConnectionError, RuntimeError) as e:
                    logger.exception(e)
                    await asyncio.sleep(self._node.protocol.wait_timeout)
                    continue
                task = asyncio.create_task(getattr(self, method)(*args, **kwargs))
                if method == "_shutdown":
                    await task
                    break

        loop.run_until_complete(_run())

    def run_in_background(self, await_ready: bool = True, timeout: Optional[float] = None) -> None:
        """
        Starts DHT in a background process. if await_ready, this method will wait until background dht
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready:
            self.wait_until_ready(timeout)

    def wait_until_ready(self, timeout: Optional[float] = None) -> None:
        self._ready.result(timeout=timeout)

    def shutdown(self) -> None:
        """Shut down a running dht process"""
        if self.is_alive():
            self._outer_pipe.send(("_shutdown", [], {}))
            self.join(self.shutdown_timeout)
            if self.is_alive():
                logger.warning("DHT did not shut down within the grace period; terminating it the hard way")
                self.terminate()

    async def _shutdown(self):
        await self._node.shutdown()

    def get(
        self, key: DHTKey, latest: bool = False, return_future: bool = False, **kwargs
    ) -> Union[Optional[ValueWithExpiration[DHTValue]], MPFuture]:
        """
        Search for a key across DHT and return either first or latest entry (if found).
        :param key: same key as in node.store(...)
        :param latest: if True, finds the latest value, otherwise finds any non-expired value (which is much faster)
        :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
        :param kwargs: parameters forwarded to DHTNode.get_many_by_id
        :returns: (value, expiration time); if value was not found, returns None
        """
        assert os.getpid() != self.pid, "calling *external* DHT interface from inside DHT will result in a deadlock"
        future = MPFuture()
        self._outer_pipe.send(("_get", [], dict(key=key, latest=latest, future=future, **kwargs)))
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

    def store(
        self,
        key: DHTKey,
        value: DHTValue,
        expiration_time: DHTExpiration,
        subkey: Optional[Subkey] = None,
        return_future: bool = False,
        **kwargs,
    ) -> Union[bool, MPFuture]:
        """
        Find num_replicas best nodes to store (key, value) and store it there until expiration time.

        :param key: msgpack-serializable key to be associated with value until expiration.
        :param value: msgpack-serializable value to be stored under a given key until expiration.
        :param expiration_time: absolute time when the entry should expire, based on hivemind.get_dht_time()
        :param subkey: if specified, add a value under that subkey instead of overwriting key (see DHTNode.store_many)
        :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
        :returns: True if store succeeds, False if it fails (due to no response or newer value)
        """
        assert os.getpid() != self.pid, "calling *external* DHT interface from inside DHT will result in a deadlock"
        future = MPFuture()
        self._outer_pipe.send(
            (
                "_store",
                [],
                dict(key=key, value=value, expiration_time=expiration_time, subkey=subkey, future=future, **kwargs),
            )
        )
        return future if return_future else future.result()

    async def _store(
        self,
        key: DHTKey,
        value: DHTValue,
        expiration_time: DHTExpiration,
        subkey: Optional[Subkey],
        future: MPFuture,
        **kwargs,
    ):
        try:
            result = await self._node.store(key, value, expiration_time, subkey=subkey, **kwargs)
            if not future.done():
                future.set_result(result)
        except BaseException as e:
            if not future.done():
                future.set_exception(e)
            raise

    def run_coroutine(
        self, coro: Callable[[DHT, DHTNode], Awaitable[ReturnType]], return_future: bool = False
    ) -> Union[ReturnType, MPFuture[ReturnType]]:
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
        :note: when run_coroutine is called with return_future=False, MPFuture can be cancelled to interrupt the task.
        """
        assert os.getpid() != self.pid, "calling *external* DHT interface from inside DHT will result in a deadlock"
        future = MPFuture()
        self._outer_pipe.send(("_run_coroutine", [], dict(coro=coro, future=future)))
        return future if return_future else future.result()

    async def _run_coroutine(
        self, coro: Callable[[DHT, DHTNode], Awaitable[ReturnType]], future: MPFuture[ReturnType]
    ):
        try:
            future.set_result(await coro(self, self._node))
        except BaseException as e:
            logger.exception("Caught an exception when running a coroutine:")
            future.set_exception(e)

    def add_validators(self, record_validators: Iterable[RecordValidatorBase]) -> None:
        if not self._ready.done():
            raise RuntimeError(
                "Can't append new validators before the DHT process has started. "
                "Consider adding them to the initial list via DHT.__init__(record_validators=...)"
            )

        self.run_coroutine(partial(DHT._add_validators, record_validators=record_validators))

    @staticmethod
    async def _add_validators(_dht: DHT, node: DHTNode, record_validators: Iterable[RecordValidatorBase]) -> None:
        node.protocol.record_validator.extend(record_validators)

    @property
    def peer_id(self) -> PeerID:
        if self._peer_id is None:
            if os.getpid() == self.pid:
                self._peer_id = self._node.peer_id
            else:
                # note: we cannot run_coroutine from the same pid because it would deadlock the event loop
                self._peer_id = self.run_coroutine(DHT._get_peer_id)
        return self._peer_id

    @staticmethod
    async def _get_peer_id(_dht: DHT, node: DHTNode) -> PeerID:
        return node.peer_id

    @property
    def client_mode(self) -> bool:
        if self._client_mode is None:
            self._client_mode = self.run_coroutine(DHT._get_client_mode)
        return self._client_mode

    @staticmethod
    async def _get_client_mode(_dht: DHT, node: DHTNode) -> bool:
        return node.protocol.client_mode

    def get_visible_maddrs(self, latest: bool = False) -> List[Multiaddr]:
        """
        Get multiaddrs of the current DHT node that should be accessible by other peers.

        :param latest: ask the P2P daemon to refresh the visible multiaddrs
        """

        return self.run_coroutine(partial(DHT._get_visible_maddrs, latest=latest))

    @staticmethod
    async def _get_visible_maddrs(_dht: DHT, node: DHTNode, latest: bool = False) -> List[Multiaddr]:
        return await node.get_visible_maddrs(latest=latest)

    async def replicate_p2p(self) -> P2P:
        """
        Get a replica of a P2P instance used in the DHT process internally.
        The replica uses the same P2P daemon as the DHT and only works while DHT is alive.
        """
        if self._p2p_replica is None or self._origin_pid != os.getpid():
            self._origin_pid = os.getpid()
            daemon_listen_maddr = self.run_coroutine(DHT._get_p2p_daemon_listen_maddr)
            self._p2p_replica = await P2P.replicate(daemon_listen_maddr)
        return self._p2p_replica

    @staticmethod
    async def _get_p2p_daemon_listen_maddr(_dht: DHT, node: DHTNode) -> Multiaddr:
        return node.p2p.daemon_listen_maddr

    def __del__(self):
        if self._parent_pid == os.getpid() and self.is_alive():
            self.shutdown()
