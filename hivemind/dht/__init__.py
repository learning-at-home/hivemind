"""
This sub-module implements a node in a Kademlia-based DHT. The code is organized as follows:
 * class DHT (below) - high-level class for model training. Runs DHTNode in a background process.
 * class DHTNode (node.py) - an asyncio implementation of dht server, stores AND gets keys. Asyncio-based.
 * class KademliaProtocol (protocol.py) - an rpc protocol to request data from dht nodes. Asyncio-based.

The code in this module is a modified version of https://github.com/bmuller/kademlia
Brian, if you're reading this: THANK YOU! you're awesome :)
"""
import asyncio
import datetime
import multiprocessing as mp
import warnings
from typing import Tuple, List, Optional

from kademlia.network import Server

from hivemind.client import RemoteExpert
from hivemind.utils import (
    run_forever,
    SharedFuture,
    PickleSerializer,
    find_open_port,
    Hostname,
    Port,
)


class DHT(mp.Process):
    """
    A high-level interface to hivemind DHT. Runs a dht node in a background process.
    :param initial_peers: one or multiple pairs of (host, port) pointing to active DHT peers. Default: no peers
    :param port: a port where DHT will listen to incoming connections. Defaults to hivemind.utils.find_open_port
    :param start: if True, automatically starts the background process on creation. Otherwise await manual start
    :param daemon: if True, the background process is marked as daemon and automatically terminated after main process
    """

    UID_DELIMETER = "."  # splits expert uids over this delimeter
    # expert is inactive iff it fails to post timestamp for *this many seconds*
    HEARTBEAT_EXPIRATION = 120
    make_key = "{}::{}".format

    def __init__(
        self,
        *initial_peers: Tuple[Hostname, Port],
        port: Optional[Port] = None,
        start: bool,
        daemon: bool = True
    ):
        super().__init__()
        port = find_open_port() if port is None else port
        self.port, self.initial_peers = port, initial_peers
        self._pipe, self.pipe = mp.Pipe(duplex=False)
        self.ready = mp.Event()
        self.server = Server()
        self.daemon = daemon
        if start:
            self.run_in_background(await_ready=True)

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.server.listen(self.port))
        loop.run_until_complete(self.server.bootstrap(self.initial_peers))
        run_forever(loop.run_forever)
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
            warnings.warn(
                "DHT shutdown has no effect: dht process is already not alive"
            )

    def get_experts(
        self, uids: List[str], heartbeat_expiration=HEARTBEAT_EXPIRATION
    ) -> List[Optional[RemoteExpert]]:
        """ Find experts across DHT using their ids; Return a list of [RemoteExpert if found else None]"""
        future, _future = SharedFuture.make_pair()
        self.pipe.send(
            (
                "_get_experts",
                [],
                dict(
                    uids=uids, heartbeat_expiration=heartbeat_expiration, future=_future
                ),
            )
        )
        return future.result()

    def _get_experts(
        self, uids: List[str], heartbeat_expiration: float, future: SharedFuture
    ):
        loop = asyncio.get_event_loop()
        lookup_futures = [
            asyncio.run_coroutine_threadsafe(
                self.server.get(self.make_key("expert", uid)), loop
            )
            for uid in uids
        ]
        current_time = datetime.datetime.now()

        experts = [None] * len(uids)
        for i, (uid, lookup) in enumerate(zip(uids, lookup_futures)):
            if lookup.result() is not None:
                (host, port), timestamp = PickleSerializer.loads(lookup.result())
                if (current_time - timestamp).total_seconds() <= heartbeat_expiration:
                    experts[i] = RemoteExpert(uid=uid, host=host, port=port)

        future.set_result(experts)

    def declare_experts(self, uids: List[str], addr, port, wait_timeout=0):
        """
        Make experts available to DHT; update timestamps if already available
        :param uids: a list of expert ids to update
        :param addr: hostname that can be used to call this expert
        :param port: port that can be used to call this expert
        :param wait_timeout: if wait_timeout > 0, waits for the procedure to finish
        """
        done_event = mp.Event() if wait_timeout else None
        self.pipe.send(
            (
                "_declare_experts",
                [],
                dict(uids=list(uids), addr=addr, port=port, done_event=done_event),
            )
        )
        if done_event is not None:
            done_event.wait(wait_timeout)

    def _declare_experts(
        self, uids: List[str], addr: str, port: int, done_event: Optional[mp.Event]
    ):
        loop = asyncio.get_event_loop()
        timestamp = datetime.datetime.now()
        expert_metadata = PickleSerializer.dumps(((addr, port), timestamp))
        prefix_metadata = PickleSerializer.dumps(timestamp)

        unique_prefixes = set()

        for uid in uids:
            asyncio.run_coroutine_threadsafe(
                self.server.set(self.make_key("expert", uid), expert_metadata), loop
            )
            uid_parts = uid.split(self.UID_DELIMETER)
            unique_prefixes.update(
                [
                    self.UID_DELIMETER.join(uid_parts[: i + 1])
                    for i in range(len(uid_parts))
                ]
            )

        for prefix in unique_prefixes:
            asyncio.run_coroutine_threadsafe(
                self.server.set(self.make_key("prefix", prefix), prefix_metadata), loop
            )

        if done_event is not None:
            done_event.set()

    def first_k_active(
        self,
        prefixes: List[str],
        k: int,
        heartbeat_expiration=HEARTBEAT_EXPIRATION,
        max_prefetch=None,
    ):
        """
        Find k prefixes with active experts; may return less if there aren't enough; used for DMoE beam search
        :param prefixes: a list of uid prefixes ordered from highest to lowest priority
        :param k: return at most *this many* active prefixes
        :param heartbeat_expiration: consider expert active if his last heartbeat was sent at most this many seconds ago
        :param max_prefetch: pre-dispatch up to *this many* asynchronous expert requests, defaults to pre-dispatch = k
        :returns: a list of at most :k: prefixes that have at least one active expert each;
        """
        future, _future = SharedFuture.make_pair()
        self.pipe.send(
            (
                "_first_k_active",
                [],
                dict(
                    prefixes=prefixes,
                    k=k,
                    heartbeat_expiration=heartbeat_expiration,
                    max_prefetch=max_prefetch or k,
                    future=_future,
                ),
            )
        )
        return future.result()

    def _first_k_active(
        self,
        prefixes: List[str],
        k,
        heartbeat_expiration,
        max_prefetch,
        future: SharedFuture,
    ):
        loop = asyncio.get_event_loop()
        lookup_prefetch = [
            asyncio.run_coroutine_threadsafe(
                self.server.get(self.make_key("prefix", prefix)), loop
            )
            for prefix in prefixes[:max_prefetch]
        ]
        current_time = datetime.datetime.now()

        active_prefixes = []

        for i, prefix in enumerate(prefixes):
            lookup = lookup_prefetch[i]

            if lookup.result() is not None:
                timestamp = PickleSerializer.loads(lookup.result())
                if (current_time - timestamp).total_seconds() <= heartbeat_expiration:
                    active_prefixes.append(prefix)
                    if len(active_prefixes) >= k:
                        future.set_result(active_prefixes)
                        return

            # pre-dispatch the next request in line
            if len(lookup_prefetch) < len(prefixes):
                lookup_prefetch.append(
                    asyncio.run_coroutine_threadsafe(
                        self.server.get(
                            self.make_key("prefix", prefixes[len(lookup_prefetch)])
                        ),
                        loop,
                    )
                )

        # could not find enough active prefixes; return what we can
        future.set_result(active_prefixes)
