""" A background process that averages your tensors with peers """

from __future__ import annotations

import heapq
import os
import random
import ctypes
from dataclasses import asdict
from typing import Sequence, Optional, Tuple, Any, Union, Awaitable, Dict, AsyncIterator, List, Set
from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing as mp
import asyncio

import torch
import uvloop
import grpc

import hivemind
from hivemind.client.averaging.allreduce import GroupAllReduce
from hivemind.client.averaging.matchmaking import Matchmaking
from hivemind.dht import DHTID, DHTExpiration, get_dht_time
from hivemind.utils import get_logger, Endpoint, Port, MPFuture, TensorDescriptor, MSGPackSerializer
from hivemind.utils.grpc import ChannelCache, serialize_torch_tensor, deserialize_torch_tensor
from hivemind.proto import averaging_pb2, averaging_pb2_grpc, runtime_pb2


# flavour types
GroupID = bytes
StreamCallToLeader = grpc.aio.UnaryStreamCall[averaging_pb2.JoinRequest, averaging_pb2.MessageFromLeader]


class AllreduceException(Exception):
    """ A special exception that is raised when allreduce can't continue normally (e.g. disbanded/bad request/etc) """


GROUP_NBITS_INTERVAL = 3
logger = get_logger(__file__)


class DecentralizedAverager(mp.Process, averaging_pb2_grpc.DecentralizedAveragingServicer):
    """
    **Warning!** Decentralized averager is in active development, some critical functionality is still underway

    Gating function averaging service. A trainer can run this service in background to periodically average his gating
    function with other trainers. The averaging pattern is chosen so that (1) you only need to average with a small
    group of peers at a time, but (2) all trainers will converge to global average in a logarithmic number of steps.
    Why averaging is valid: see https://github.com/learning-at-home/hivemind/issues/95#issuecomment-688806705
    On global convergence: see https://github.com/learning-at-home/hivemind/issues/95#issuecomment-717719400

    :param averaged_tensors: a sequence of pytorch tensors that will be averaged in each all-reduce
    :param dht: a DHT node that will be used to find groups
    :param start: if True, starts the background process immediately
    :param initial_group_bits: TODO (also all other examples)
    :param target_group_size:
    :param averaging_expiration: attempt to find a group for this many seconds, otherwise try again
      note - this expiration time only applies to looking for group, passing tensors in allreduce may take more time
    :param allreduce_timeout: spend at most this many seconds for allreduce (after group is formed)
    :param listen: if True (default), this averager will accept incoming requests from other peers and perform allreduce
            if False, the averager will register as a freeloader and attempt to fetch vectors from other averagers
    :param listen_on: network interface, e.g. "0.0.0.0:1337" or "localhost:*" (* means pick any port) or "[::]:7654"
    :param receiver_threads: uses this many threads to await on input pipe. Default = 1 should be enough in most cases
    :param channel_options: options for grpc.aio.insecure_channel, e.g. [('grpc.enable_retries', 0)]
          see https://grpc.github.io/grpc/core/group__grpc__arg__keys.html for a list of all options
    :param kwargs: extra parameters forwarded to in grpc.aio.server
    You can perform averaging using DecentralizedOptimizer (see below) or by manually running each step as such:

    >> TODO add a working example
    # TODO option: use a special key, group_prefix.SOMETHING to state the current number of dimensions
    # each group leader should write this key (to a dict with subkeys) on a sucessful form_group; reader should pick key
    # problem: this may cause excessive load on the peers that store that one special key.
    Alternative: use group_prefix.NDIM{group_nbits[:min_group_nbits]} aka "balance the load between peers 2**min-group-nbits" peers
    Alternative: add such keys to every smaller dimension group down to {GROUP_NBITS_INTERVAL}, check for such keys in your current highest dimension.
        If found, jump to the specified dimension.
    """
    _lock_looking_for_group: asyncio.Lock()
    _matchmaking: Matchmaking

    def __init__(self, averaged_tensors: Sequence[torch.Tensor], dht: hivemind.dht.DHT, *, start: bool,
                 prefix: str, target_group_size: int, min_group_size: int = 1, initial_group_bits: Optional[str] = None,
                 averaging_expiration: float = 15, allreduce_timeout: float = float('inf'),
                 compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
                 listen_on: Endpoint = '0.0.0.0:*', receiver_threads: int = 1,
                 channel_options: Optional[Sequence[Tuple[str, Any]]] = None, **kwargs):
        assert '.' not in prefix, "group prefix must be a string without ."
        if target_group_size != 2 ** target_group_size.bit_length():
            logger.warning("It is recommended to set target_group_size to a power of 2.")
        if initial_group_bits is None:
            initial_group_bits = ''.join(random.choices('01', k=GROUP_NBITS_INTERVAL))
            logger.debug(f"Initializing with random {GROUP_NBITS_INTERVAL}-bit group index: {initial_group_bits}")
        assert len(initial_group_bits) >= GROUP_NBITS_INTERVAL

        super().__init__()
        self.dht = dht
        self.listen_on, self.receiver_threads, self.kwargs = listen_on, receiver_threads, kwargs
        self.channel_options = channel_options
        self.matchmaking_kwargs = dict(prefix=prefix, initial_group_bits=initial_group_bits,
                                       target_group_size=target_group_size, min_group_size=min_group_size,
                                       averaging_expiration=averaging_expiration, allreduce_timeout=allreduce_timeout,
                                       compression_type=compression_type)

        self.averaged_tensors = tuple(averaged_tensors)
        # TODO use mp.Lock to prevent someone from modifying tensors before we copy them! maybe.
        for tensor in self.averaged_tensors:
            assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
            tensor.share_memory_()

        self._pipe, self.pipe = mp.Pipe(duplex=True)  # a control pipe used to communicate with a background process
        self._port = mp.Value(ctypes.c_uint32, 0)  # assigned when averager starts, accessible via self.port
        self._averager_endpoint: Optional[Endpoint] = None

        self.ready = mp.Event()  # whether the averager process has started (and ready for incoming requests)

        self._looking_for_group = False  # whether i am currently willing to allreduce with someone
        self._current_leader: Optional[Endpoint] = None  # iff i am a follower, this is a link to my current leader
        self._current_followers: Set[Endpoint] = set()  # iff i am a leader, this contains my followers excluding myself
        self._declared_expiration_time = -float('inf')  # iff i am looking for group, this is my latest expiration time
        self._running_groups: Dict[GroupID, GroupAllReduce] = {}  # one or more assembled groups that run all-reduce

        if start:
            self.run_in_background(await_ready=True)

    @property
    def port(self) -> Optional[Port]:
        return self._port.value if self._port.value != 0 else None

    @property
    def endpoint(self) -> Endpoint:
        if not hasattr(self, '_averager_endpoint'):
            logger.info(f"Assuming averager endpoint to be {self._averager_endpoint}")
            self._averager_endpoint = f"{self.listen_on}:{self.port}"
        return self._averager_endpoint

    def _get_peer_stub(self, peer: Endpoint) -> averaging_pb2_grpc.DecentralizedAveragingStub:
        return ChannelCache.get_stub(peer, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)

    def __repr__(self):
        if len(self._running_groups) > 0:
            lfg_description = 'RUNNING_ALLREDUCE'
        elif not self._looking_for_group:
            lfg_description = 'NOT_LOOKING_FOR_GROUP'
        elif self._accepted_to_group_as_follower.is_set():
            lfg_description = 'WAITING_FOR_LEADER'
        else:
            lfg_description = f'LOOKING_FOR_GROUP; current followers = {len(self._current_followers)}'
        return f"{self.__class__.__name__}({self.endpoint}, {lfg_description})"

    def run(self):
        """ Serve DecentralizedAverager forever. This function will not return until the averager is shut down """
        if asyncio.get_event_loop().is_running():
            asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop

        uvloop.install()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # initialize asyncio synchronization primitives in this event loop
        self._lock_looking_for_group, self._lock_request_join_group = asyncio.Lock(), asyncio.Lock()
        self._accepted_to_group_as_follower, self._group_disbanded_cond = asyncio.Event(), asyncio.Condition()
        self._assembled_group = asyncio.Future()  # if _looking_for_group, this future will return that group or error
        self._received_latest_group_id = asyncio.Event()  # this event is set to False iff we requested someone to be
        self._received_latest_group_id.set()  # ... our leader but he has neither began allreduce nor rejected us YET.

        pipe_awaiter = ThreadPoolExecutor(self.receiver_threads)

        async def _run():
            grpc.aio.init_grpc_aio()
            server = grpc.aio.server(**self.kwargs)
            averaging_pb2_grpc.add_DecentralizedAveragingServicer_to_server(self, server)
            found_port = server.add_insecure_port(self.listen_on)
            assert found_port != 0, f"Failed to listen to {self.listen_on}"
            self._port.value = found_port
            await server.start()
            self.ready.set()

            while True:
                method, args, kwargs = await loop.run_in_executor(pipe_awaiter, self._pipe.recv)
                asyncio.create_task(getattr(self, method)(*args, **kwargs))

        loop.run_until_complete(_run())

    def run_in_background(self, await_ready=True, timeout=None):
        """
        Starts averager in a background process. if await_ready, this method will wait until background dht
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError(f"Server didn't notify .ready in {timeout} seconds")

    def shutdown(self) -> None:
        """ Shut down the averager process """
        # TODO notify peers before terminating
        if self.is_alive():
            self.terminate()
        else:
            logger.warning("DHT shutdown has no effect: the process is not alive")

    def group_allreduce(self, timeout: Optional[float] = None, return_future=False
                        ) -> Union[Sequence[torch.Tensor], Awaitable[Sequence[torch.Tensor]]]:
        """
        Set up the averager to look for a group and run all-reduce once, then return the averaged tensors

        :note: this function implemented for debugging and will be removed in future versions
        :param timeout: if averager was unable to *find* group in this many seconds, consider allreduce failed
        :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
        """
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_group_allreduce', [], dict(future=_future, timeout=timeout)))
        return future if return_future else future.result()

    async def _group_allreduce(self, *, future: MPFuture, timeout: Optional[float]):
        try:
            if self._lock_looking_for_group.locked():
                logger.debug("Another run_group_allreduce already in progress. The current run will be scheduled after"
                             " the existing group is assembled.")
            async with self._lock_looking_for_group:

                group_allreduce = await self._look_for_group(timeout=timeout)
            if group_allreduce is None:
                future.set_exception(AllreduceException(f"{self} - group_allreduce failed, unable to find group"))
            else:
                future.set_result(await self._run_allreduce(group_allreduce, timeout=self.allreduce_timeout))

        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            _ = self._running_groups.pop(group_allreduce.group_id, None)

    async def _look_for_group(self, *, timeout: Optional[float] = None) -> Optional[GroupAllReduce]:
        """
        :returns: an assembled group if successful, None if failed; does NOT perform the actual averaging
        Iterate over the averagers from a given group_identifier that have higher leadership priority than yourself.
        """
        assert self.pid == os.getpid(), f"this method can only be called from inside {self.__class__.__name__}"
        assert not self._looking_for_group, "already looking for group"
        assert not self._assembled_group.done(), f"already assembled group: {self._assembled_group.result()}"
        assert not self._accepted_to_group_as_follower.is_set(), f"already a accepted by some leader"
        assert len(self._current_followers) == 0, f"averager already has {len(self._current_followers)} followers"
        # ^-- TODO remove this check. This may happen normally under concurrency if we disbanded group but have not kicked out previous followers yet
        assert self._declared_expiration_time == -float('inf'), "should have cleared _declared_expiration_time"
        assert self._declared_allreduce_group is None, "should have cleared _declared_allreduce_group"

        allreduce_group = f"{self.prefix}.{self.group_bits}"  # TODO use interval?
        # TODO update group_bits on success! reduce number of bits on not enough peers.
        # TODO after allreduce finishes, we may need to ask leader to notify lower keys about this
        # (so as to fix possible network partitioning if some peers operate on a much smaller nbits)

        allowed_discrepancy = hivemind.utils.timed_storage.MAX_DHT_TIME_DISCREPANCY_SECONDS
        end_time = get_dht_time() + (timeout or float('inf'))

        known_averagers: Set[Tuple[Endpoint, DHTExpiration]] = set()
        heap: List[Tuple[DHTExpiration, Endpoint]] = []  # queue of all averagers, from earliest to latest expiration
        max_assured_time = -float('inf')
        # ^-- max_assured_time is such a time that all averagers below this expiration_time are in known_averagers

        async def _declare_averager():
            new_expiration_time = get_dht_time() + self.averaging_expiration
            stored_ok = await self.dht.declare_averager(self.endpoint, allreduce_group, new_expiration_time,
                                                        looking_for_group=True, return_future=True)
            if not stored_ok:
                logger.warning(f"failed to subscribe to group {allreduce_group} : store rejected by DHT peers")
            else:
                self._declared_expiration_time = new_expiration_time
                self._declared_allreduce_group = allreduce_group
            return new_expiration_time

        async def _fetch_more_peers():
            nonlocal max_assured_time, known_averagers, heap
            request_time = get_dht_time()
            new_peers = await self.dht.get_averagers(allreduce_group, only_active=True, return_future=True)
            max_assured_time = max(max_assured_time, request_time - allowed_discrepancy)

            for peer, peer_expiration_time in new_peers:
                if (peer, peer_expiration_time) not in known_averagers:
                    heapq.heappush(heap, (peer_expiration_time, peer))
                    known_averagers.add((peer, peer_expiration_time))
                    max_assured_time = max(max_assured_time, peer_expiration_time - allowed_discrepancy)

        async def _request_potential_leaders():
            """
            Iterate over active averagers before us and attempt to join their group until first success
            :returns: return GroupAllReduce if one of the averagers accepts and initiates allreduce, else return None
            """
            while True:
                next_best_expiration = heap[0][0] if heap else self._declared_expiration_time
                if next_best_expiration > max_assured_time:
                    # if there's a chance that DHT contains averagers newer than the earliest one from our local heap,
                    # ... then fetch more candidates from the DHT until we are confident we know the next-earliest peer
                    timeout = min(end_time, self._declared_expiration_time) - get_dht_time()
                    await asyncio.wait_for(_fetch_more_peers(), timeout=timeout if timeout != float('inf') else None)
                    continue

                if len(heap) == 0 or heap[0][0] >= self._declared_expiration_time:
                    break  # no extra averagers are available until our expiration AND we can't fetch more

                next_best_expiration, next_best_leader = heapq.heappop(heap)
                if next_best_expiration < get_dht_time():
                    continue  # this leader expired before we could request to join his group
                if next_best_leader == self.endpoint:
                    continue  # do not count myself as a potential leader for me (even at a later expiration time)

                maybe_group_allreduce = await self._request_join_group(next_best_leader, self._declared_expiration_time)
                if maybe_group_allreduce is not None:
                    self._assembled_group.set_result(maybe_group_allreduce)

        try:
            self._looking_for_group = True
            while True:
                # step 1: declare the averager to DHT, make it visible for other averagers
                await _declare_averager()

                # step 2: request potential leaders until first accept OR until we are chosen as a leader
                request_candidates_task = asyncio.create_task(_request_potential_leaders())

                try:  # wait until we are ready to run allreduce (as either follower or leader) or reach expiration
                    timeout = min(end_time, self._declared_expiration_time) - get_dht_time()
                    return await asyncio.wait_for(self._assembled_group, timeout if timeout != float('inf') else None)
                except asyncio.TimeoutError:
                    if len(self._current_followers) >= self.min_group_size:
                        # the time is up, we have a *good enough* group. run allreduce as is.
                        return await self._leader_assemble_group()
                    else:
                        await self._leader_disband_group()
                        # TODO maybe adjust grid size
                        continue  # re-declare averager with new expiration time
                finally:
                    request_candidates_task.cancel()
        finally:
            self._looking_for_group = False
            self._declared_expiration_time = float('-inf')
            self._declared_allreduce_group = None
            self._accepted_to_group_as_follower.clear()
            if self._assembled_group.done():
                self._assembled_group = asyncio.Future()
            asyncio.create_task(self.dht.declare_averager(
                self.endpoint, allreduce_group, expiration_time=self._declared_expiration_time,
                looking_for_group=False, return_future=True))

    async def _request_join_group(self, leader: Endpoint, expiration_time: DHTExpiration) -> Optional[GroupAllReduce]:
        """
        :param leader: request this peer to be your leader for allreduce
        :param expiration_time: inform leader that we intend to begin averaging before this expiration_time
        :returns: if leader leader accepted us and started AllReduce, return that AllReduce. Otherwise, return None
        """
        assert self.pid == os.getpid(), f"this method can only be called from inside {self.__class__.__name__}"
        assert not self._accepted_to_group_as_follower.is_set(), "already accepted to another group as a follower"
        call: Optional[grpc.aio.UnaryStreamCall[averaging_pb2.JoinRequest, averaging_pb2.MessageFromLeader]] = None
        try:
            async with self._lock_request_join_group:
                self._received_latest_group_id.clear()
                call = self._get_peer_stub(leader).rpc_join_group(averaging_pb2.JoinRequest(
                    endpoint=self.endpoint, schema_hash=self.schema_hash, expiration=expiration_time))

                message = await call.read()  # TODO use timeout?
                if message.code != averaging_pb2.ACCEPTED:
                    code = averaging_pb2.MessageCode.Name(message.code)
                    logger.debug(f"{self.endpoint} - requested {leader} to be my leader, but got rejected with {code}")
                    return None

                # else: we were accepted
                logger.debug(f"{self.endpoint} - joining the group of {leader}; waiting for peers")
                self._accepted_to_group_as_follower.set()
                self._current_leader = leader
                await self._leader_disband_group()

            message = await call.read()
            if message.code == averaging_pb2.BEGIN_ALLREDUCE:
                return await self._follower_assemble_group(leader, message.group_id, message.ordered_group_endpoints)
            elif message.code == averaging_pb2.GROUP_DISBANDED and bool(message.suggested_leader):
                logger.debug(f"{self} - leader disbanded group and redirected us to {message.suggested_leader}")
                return await self._request_join_group(message.suggested_leader, expiration_time)

            else:
                logger.debug(f"{self} - leader sent {averaging_pb2.MessageCode.Name(message.code)}, leaving group")
                return None

        finally:
            self._current_leader = None
            self._received_latest_group_id.set()
            if call is not None:
                call.cancel()

    async def rpc_join_group(self, request: averaging_pb2.JoinRequest, context: grpc.ServicerContext
                             ) -> AsyncIterator[averaging_pb2.MessageFromLeader]:
        """ accept or reject a join request from another averager; if accepted, run him through allreduce steps """
        try:
            # stage 1: check if there is a reason to reject a peer outright
            async with self._lock_request_join_group:
                if not self._looking_for_group:
                    yield averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_LOOKING_FOR_GROUP)
                    return
                if not is_valid_join_request(request):
                    yield averaging_pb2.MessageFromLeader(code=averaging_pb2.PROTOCOL_VIOLATION)
                    return
                if request.schema_hash != self.schema_hash:
                    yield averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_SCHEMA_HASH)
                    return
                if self._declared_expiration_time == float('-inf'):
                    yield averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_DECLARED)
                    return
                if self._declared_expiration_time > (request.expiration or float('inf')):
                    yield averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_EXPIRATION_TIME)
                    return
                if self._accepted_to_group_as_follower.is_set():
                    yield averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_A_LEADER,
                                                          suggested_leader=self._current_leader)
                    return
                if request.endpoint == self.endpoint or request.endpoint in self._current_followers:
                    yield averaging_pb2.MessageFromLeader(code=averaging_pb2.DUPLICATE_ENDPOINT)
                    return
                if len(self._current_followers) + 1 >= self.target_group_size:
                    yield averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_IS_FULL)
                    return

                # stage 2: if found no red flags, accept peer as your follower
                self._current_followers.add(request.endpoint)
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.ACCEPTED)

                if len(self._current_followers) + 1 >= self.target_group_size:
                    # outcome 1: if everything went well, we have assembled a full group and are ready for allreduce
                    await self._leader_assemble_group()  # note: this will trigger self._assembled_group

            # stage 3: wait for the group to be assembled (or for the follower to leave, whichever comes first)
            async with self._group_disbanded_cond:
                await asyncio.wait({self._assembled_group, self._accepted_to_group_as_follower.wait(),
                                    self._group_disbanded_cond.wait()},
                                   return_when=asyncio.FIRST_COMPLETED,
                                   timeout=max(0.0, self._declared_expiration_time - get_dht_time()))
                # TODO handle group disbanded without suggested leader - check if group_disbanded.wait is finished!

            if self._accepted_to_group_as_follower.is_set():
                # outcome 2: we were accepted to another averager's group => send all followers to our new leader
                assert self._current_leader is not None, "internal error: accepted to group but did not fill leader"
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_DISBANDED,
                                                      suggested_leader=self._current_leader)
                return

            if not self._assembled_group.done():
                if len(self._current_followers) + 1 < self.min_group_size \
                      or request.endpoint not in self._current_followers:
                    yield averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_DISBANDED)
                    return
                # outcome 3: the time is up, we have *some* followers => run allreduce with what we have
                await self._leader_assemble_group()

            group_allreduce = self._assembled_group.result()
            yield averaging_pb2.MessageFromLeader(
                code=averaging_pb2.BEGIN_ALLREDUCE, group_id=group_allreduce.group_id,
                ordered_group_endpoints=group_allreduce.ordered_group_endpoints)

        except Exception as e:
            logger.exception(e)
            yield averaging_pb2.MessageFromLeader(code=averaging_pb2.INTERNAL_ERROR)

        finally:  # note: this code is guaranteed to run even if the coroutine is destroyed prematurely
            self._current_followers.discard(request.endpoint)

    async def _leader_assemble_group(self) -> GroupAllReduce:
        """ Form up all current followers into a group and prepare to _run_allreduce  """
        group_id = DHTID.generate().to_bytes()
        assert group_id not in self._running_groups, "Randomly generated a group_id that already exists, " \
                                                     "this should normally happen once in >10^4 years."
        ordered_group_endpoints = list(self._current_followers)
        ordered_group_endpoints.append(self.endpoint)
        logger.debug(f"{self.endpoint} - leader started allreduce with {len(ordered_group_endpoints)} followers.")
        self._running_groups[group_id] = group_allreduce = GroupAllReduce(
            group_id=group_id, tensors=self.averaged_tensors, endpoint=self.endpoint,
            ordered_group_endpoints=ordered_group_endpoints)
        self._assembled_group.set_result(group_allreduce)
        self._looking_for_group = False
        return group_allreduce

    async def _follower_assemble_group(self, leader: Endpoint, group_id: GroupID,
                                       ordered_group_endpoints: Sequence[Endpoint]) -> GroupAllReduce:
        """ Prepare to run allreduce using a list of peers provided by our leader """
        logger.debug(f"{self.endpoint} - follower started allreduce after being prompted by leader {leader}.")
        assert self._accepted_to_group_as_follower.is_set(), "averager is not currently following any leader"
        assert self._current_leader == leader, f"averager does not follow this {leader} (real: {self._current_leader})"
        assert self.endpoint in ordered_group_endpoints, "Leader sent us group_endpoints that does not contain us!"
        assert group_id not in self._running_groups, "Duplicate group id, already running this _GroupAllReduce"
        self._running_groups[group_id] = group_allreduce = GroupAllReduce(
            group_id=group_id, tensors=self.averaged_tensors, endpoint=self.endpoint,
            ordered_group_endpoints=ordered_group_endpoints)
        self._assembled_group.set_result(group_allreduce)
        self._looking_for_group = False
        return group_allreduce

    async def _leader_disband_group(self):
        """ Cancel all followers """
        assert self._looking_for_group, "too late to disband: no longer looking for group"
        async with self._lock_request_join_group:
            for follower in list(self._current_followers):
                self._current_followers.discard(follower)  # this will cause rpc_foin_group to kick the follower out
            async with self._group_disbanded_cond:
                self._group_disbanded_cond.notify_all()

    async def _run_allreduce(self, group_id: GroupID, timeout: Optional[float] = None):
        """ send allreduce requests to all peers and collect results, return the averaged tensor """
        assert group_id in self._running_groups, f"unknown group id {group_id}, current groups: {self._running_groups}"
        group_allreduce = self._running_groups[group_id]

        async def _average_one_part(peer_endpoint: Endpoint, local_part: torch.Tensor):
            serialized_tensor_part = serialize_torch_tensor(local_part, self.compression_type, allow_inplace=False)
            response = await self._get_peer_stub(peer_endpoint).rpc_aggregate_part(
                averaging_pb2.AveragingData(code=averaging_pb2.PART_FOR_AVERAGING, group_id=group_allreduce.group_id,
                                            endpoint=group_allreduce.endpoint, tensor_part=serialized_tensor_part))
            if response.code == averaging_pb2.AVERAGED_PART:
                group_allreduce.register_averaged_part(peer_endpoint, deserialize_torch_tensor(response.tensor_part))
            else:
                message_code = averaging_pb2.MessageCode.Name(response.code)
                reference_code = averaging_pb2.MessageCode.Name(response.code)
                group_allreduce.set_exception(AllreduceException(f"peer {peer_endpoint} replied with {message_code}"
                                                                 f" instead of {reference_code}, allreduce failed"))

        try:
            for peer_endpoint, tensor_part in group_allreduce.local_tensor_parts.items():
                if peer_endpoint != self.endpoint:
                    asyncio.create_task(_average_one_part(peer_endpoint, tensor_part))
            return await asyncio.wait_for(group_allreduce.averaged_tensors, timeout=timeout)
        except Exception as e:
            code = averaging_pb2.CANCELLED if isinstance(e, asyncio.CancelledError) else averaging_pb2.INTERNAL_ERROR
            logger.debug(f"{self} - notifying peers about {averaging_pb2.MessageCode.Name(code)}")
            group_allreduce.set_exception(e)

            async def send_error_to_peer(peer_endpoint: Endpoint):
                await self._get_peer_stub(peer_endpoint).rpc_aggregate_part(averaging_pb2.AveragingData(
                    group_id=group_id, endpoint=group_allreduce.endpoint, code=code))

            for peer_endpoint in group_allreduce.ordered_group_endpoints:
                asyncio.create_task(send_error_to_peer(peer_endpoint))

    async def rpc_aggregate_part(self, request: averaging_pb2.AveragingData, context: grpc.ServicerContext):
        """ a groupmate sends us a part of his tensor; we should average it with other peers and return the result """
        if request.group_id not in self._running_groups and not self._received_latest_group_id.is_set():
            await self._received_latest_group_id.wait()  # this handles a special case when leader accepted us to group
            # AND began allreduce right away, but his response with group_id was delayed and other peers got to us first
        if request.group_id not in self._running_groups:
            return averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)
        group_allreduce = self._running_groups[request.group_id]

        if request.code == averaging_pb2.PART_FOR_AVERAGING:
            try:
                tensor_part = deserialize_torch_tensor(request.tensor_part)
                averaged_part = await group_allreduce.accumulate_part(request.endpoint, tensor_part)
                serialized = serialize_torch_tensor(averaged_part, request.tensor_part.compression, allow_inplace=False)
                return averaging_pb2.AveragingData(code=averaging_pb2.AVERAGED_PART, tensor_part=serialized)
            except Exception as e:
                group_allreduce.set_exception(e)
                logger.error(f"{self} - encountered {e} when aggregating part from {request.endpoint}")
                return averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
        else:
            error_code = averaging_pb2.MessageCode.Name(request.code)
            logger.debug(f"{self} - peer {request.endpoint} sent {error_code}, allreduce cannot continue")
            group_allreduce.set_exception(AllreduceException(f"peer {request.endpoint} sent {error_code}."))
            return averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)


def compute_schema_hash(tensors: Sequence[torch.Tensor]) -> bytes:
    """ A hash that describes follower's tensor shapes, dtypes, devices, but not the actual values """
    schema_dicts = [{field_name: str(field_value)
                    for field_name, field_value in asdict(TensorDescriptor.from_tensor(tensor)).items()}
                    for tensor in tensors]
    return DHTID.generate(source=MSGPackSerializer.dumps(schema_dicts)).to_bytes()


def is_valid_join_request(request: averaging_pb2.JoinRequest) -> bool:
    assert len(request.ListFields()) == 3, "this function assumes JoinRequest has three fields, it should be updated"
    return (isinstance(request.schema_hash, bytes) and len(request.schema_hash) > 0 and
            isinstance(request.expiration, DHTExpiration) and request.expiration != float('inf') and
            isinstance(request.endpoint, Endpoint) and len(request.endpoint) > 0)
