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
from hivemind.dht import DHTID, DHTExpiration, get_dht_time
from hivemind.utils import get_logger, Endpoint, Port, MPFuture, TensorDescriptor, MSGPackSerializer
from hivemind.utils.grpc import ChannelCache, serialize_torch_tensor, deserialize_torch_tensor
from hivemind.proto import averaging_pb2, averaging_pb2_grpc


# flavour types
GroupID = bytes
StreamCallToLeader = grpc.aio.UnaryStreamCall[averaging_pb2.PeerInfo, averaging_pb2.MessageFromLeader]


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
    :param initial_group_bits: TODO
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

    def __init__(self, averaged_tensors: Sequence[torch.Tensor], dht: hivemind.dht.DHT, *, start: bool,
                 group_prefix: str, target_group_size: int, initial_group_bits: Optional[str] = None,
                 averaging_expiration: float = 15, allreduce_timeout: float = float('inf'),
                 listen_on: Endpoint = '0.0.0.0:*', receiver_threads: int = 1,
                 channel_options: Optional[Sequence[Tuple[str, Any]]] = None, **kwargs):
        assert '.' not in group_prefix, "group prefix must be a string without ."
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

        self.group_prefix, self.group_bits, self.target_group_size = group_prefix, initial_group_bits, target_group_size
        self.averaging_expiration, self.allreduce_timeout = averaging_expiration, allreduce_timeout

        self.averaged_tensors = tuple(averaged_tensors)
        # TODO use mp.Lock to prevent someone from modifying tensors before we copy them! maybe.
        for tensor in self.averaged_tensors:
            assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
            tensor.share_memory_()

        self.schema_hash = compute_schema_hash(self.averaged_tensors)

        self._pipe, self.pipe = mp.Pipe(duplex=True)  # a control pipe used to communicate with a background process
        self._port = mp.Value(ctypes.c_uint32, 0)  # assigned when averager starts, accessible via self.port
        self._averager_endpoint: Optional[Endpoint] = None

        self.ready = mp.Event()  # whether the averager process has started (and ready for incoming requests)

        self._looking_for_group = False  # whether i am currently forming a group
        self._current_followers: Set[Endpoint] = set()  # if i am a leader, this contains my followers excluding myself
        self._running_groups: Dict[GroupID, GroupAllReduce] = {}  # one or more groups running all-reduce in background

        self._lock_looking_for_group = asyncio.Lock()
        self._lock_request_join_group = asyncio.Lock()
        self._accepted_to_group_as_follower = asyncio.Event()
        self._assembled_group: asyncio.Future[GroupAllReduce] = asyncio.Future()
        self._pending_leader_responded = asyncio.Event()  # this event is set to False iff we requested someone to be
        self._pending_leader_responded.set()  # ... our leader but he has neither began allreduce nor rejected us YET.
        # If leader accepted us, but his response (group_id) has been delayed by network, our groupmates may get to us
        # BEFORE we know they are our groupmates. We delay incoming rpc_average_part requests until leader responds.

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


    def run(self):
        """ Serve DecentralizedAverager forever. This function will not return until the averager is shut down """
        if asyncio.get_event_loop().is_running():
            asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop

        uvloop.install()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        listen_on, receiver_threads, server_kwargs = self.listen_on, self.receiver_threads, self.kwargs
        pipe_awaiter = ThreadPoolExecutor(receiver_threads)
        self._lock_looking_for_group = asyncio.Lock()

        async def _run():
            grpc.aio.init_grpc_aio()
            server = grpc.aio.server(**server_kwargs)
            averaging_pb2_grpc.add_DecentralizedAveragingServicer_to_server(self, server)
            found_port = server.add_insecure_port(listen_on)
            assert found_port != 0, f"Failed to listen to {listen_on}"
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

    def run_group_allreduce(self, return_future=False) -> Union[Sequence[torch.Tensor], Awaitable[Sequence[torch.Tensor]]]:
        """
        Set up the averager to look for a group and run all-reduce once, then return the averaged tensors

        :note: this function implemented for debugging and will be removed in future versions
        :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
        """
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_run_group_allreduce', [], dict(future=_future)))
        return future if return_future else future.result()

    # async def _group_allreduce(self, *, future: MPFuture):
    #     if self._forming_group is not None:
    #         logger.debug("Another run_group_allreduce already in progress. The current run will be scheduled after"
    #                      " the existing group is assembled.")
    #         raise NotImplementedError("Concurrent group allreduce is not implemented yet - use locks!")
    #
    #     try:
    #         self.set_flags_clear_events_clear_followers_if_any()
    #
    #         allowed_discrepancy = hivemind.utils.timed_storage.MAX_DHT_TIME_DISCREPANCY_SECONDS
    #         self._forming_group = group_allreduce = GroupAllReduce(self._averager_endpoint, expiration_time, self.averaged_tensors)
    #         # after this line, the averager will begin accepting incoming requests for that group
    #
    #         async for leader in self._iterate_potential_leaders(group_identifier, self._averager_endpoint, expiration_time):
    #             await self.interrupt_early_if_i_can_run_allreduce_as_a_leader()
    #             accepted = group_allreduce.request_join_group(leader)
    #             if accepted:
    #                 await self.send_my_followers_to(leader)
    #                 self._running_groups[group_allreduce.group_id] = group_allreduce
    #                 leader_started_allreduce = self.wait_for_leader_until(expiration_time + allowed_discrepancy + self.grace_period)
    #                 if leader_started_allreduce:
    #                     break
    #                 else:
    #                     continue # otherwise try next potential leader in line
    #
    #         self.maybe_i_can_assemble_my_group_until(expiration_time)
    #
    #
    #
    #
    #         #### PREVIOUS CODE BELOW
    #
    #
    #
    #         if leader_endpoint is None:
    #             async with self._lock_forming_a_group:
    #                 group_allreduce.start_new_group(max_size=self.target_group_size)
    #                 self._forming_group = self._pending_groups[group_allreduce.group_id] = group_allreduce
    #                 await asyncio.wait_for(group_allreduce.assembled_group, expiration - get_dht_time())
    #
    #             future.set_result(await group_allreduce.run_allreduce())
    #         else:
    #             async with self._lock_forming_a_group:
    #                 accepted = await group_allreduce.request_join_group(leader_endpoint)
    #                 if not accepted:
    #                     group_allreduce.set_exception(AllreduceException(f"Rejected by {leader_endpoint}"))
    #                     raise group_allreduce.exception()
    #
    #                 self._forming_group = self._pending_groups[group_allreduce.group_id] = group_allreduce
    #                 started_allreduce = await group_allreduce.wait_for_allreduce()
    #
    #                 if started_allreduce:
    #                     future.set_result(await group_allreduce.run_allreduce())
    #                 else:
    #                     future.set_exception(group_allreduce.exception())
    #
    #     except Exception as e:
    #         future.set_exception(e)
    #         raise
    #     finally:
    #         _ = self._pending_groups.pop(group_allreduce.group_id, None)
    #         if group_allreduce is self._forming_group:
    #             self._forming_group = None

    async def _look_for_group(self, timeout: Optional[float] = None) -> Optional[GroupAllReduce]:
        """
        :returns: an assembled group if successful, None if failed; does NOT perform the actual averaging
        Iterate over the averagers from a given group_identifier that have higher leadership priority than yourself.
        """
        assert self.pid == os.getpid(), f"this method can only be called from inside {self.__class__.__name__}"
        assert not self._looking_for_group, "already looking for group"
        assert not self._assembled_group.done(), f"already assembled group: {self._assembled_group.result()}"

        group_identifier = f"{self.group_prefix}.{self.group_bits}"  # TODO use interval?
        # TODO update group_bits on success! reduce number of bits on not enough peers.

        allowed_discrepancy = hivemind.utils.timed_storage.MAX_DHT_TIME_DISCREPANCY_SECONDS
        end_time = get_dht_time() + (timeout or float('inf'))
        declared_expiration_time = -float('inf')  # last _declared_ expiration time

        known_averagers: Set[Tuple[Endpoint, DHTExpiration]] = set()
        heap: List[Tuple[DHTExpiration, Endpoint]] = []  # queue of all averagers, from earliest to latest expiration
        max_assured_time = -float('inf')
        # ^-- max_assured_time is such a time that all averagers below this expiration_time are in known_averagers

        async def _fetch_more_peers():
            nonlocal max_assured_time, known_averagers, heap
            request_time = get_dht_time()
            new_peers = await self.dht.get_averagers(group_identifier, only_active=True, return_future=True)
            max_assured_time = max(max_assured_time, request_time - allowed_discrepancy)

            for peer, peer_expiration_time in new_peers:
                if (peer, peer_expiration_time) not in known_averagers:
                    heapq.heappush(heap, (peer_expiration_time, peer))
                    known_averagers.add((peer, peer_expiration_time))
                    max_assured_time = max(max_assured_time, peer_expiration_time - allowed_discrepancy)

        async def _try_join_group():
            """
            Iterate over active averagers before us and attempt to join their group until first success
            :returns: return GroupAllReduce if one of the averagers accepts and initiates allreduce, else return None
            """
            while True:
                next_best_expiration = heap[0][0] if heap else declared_expiration_time
                if next_best_expiration > max_assured_time:
                    # if there's a chance that DHT contains averagers newer than the earliest one from our local heap,
                    # ... then fetch more candidates from the DHT until we are confident we know the next-earliest peer
                    timeout = min(end_time, declared_expiration_time) - get_dht_time()
                    await asyncio.wait_for(_fetch_more_peers(), timeout=timeout if timeout != float('inf') else None)
                    continue

                if len(heap) == 0 or heap[0][0] >= declared_expiration_time:
                    break  # no extra averagers are available until our expiration AND we can't fetch more

                next_best_expiration, next_best_leader = heapq.heappop(heap)
                if next_best_expiration < get_dht_time():
                    continue  # this leader expired before we could request to join his group
                if next_best_leader == self.endpoint:
                    continue  # do not count myself as a potential leader for me (even at a later expiration time)

                maybe_group_allreduce = await self._request_join_group(next_best_leader, declared_expiration_time)
                if maybe_group_allreduce is not None:
                    self._assembled_group.set_result(maybe_group_allreduce)

        try:
            self._looking_for_group = True
            while True:
                # step 1: declare the averager to DHT, make it visible for other averagers
                new_expiration_time = get_dht_time() + self.averaging_expiration
                stored_ok = await self.dht.declare_averager(self.endpoint, group_identifier, new_expiration_time,
                                                            looking_for_group=True, return_future=True)
                assert stored_ok, f"failed to subscribe to group {group_identifier} : store rejected by DHT peers"
                declared_expiration_time = new_expiration_time

                # step 2: request potential leaders until first accept OR until we are chosen as a leader
                request_candidates_task = asyncio.create_task(_try_join_group())

                try:  # wait until we are ready to run allreduce (as either follower or leader) or reach expiration
                    timeout = min(end_time, declared_expiration_time) - get_dht_time()
                    return await asyncio.wait_for(self._assembled_group, timeout if timeout != float('inf') else None)
                except asyncio.TimeoutError:
                    #TODO maybe adjust grid size
                    if len(self._current_followers) > 0:  # TODO check against some min_followers, disband if too small?
                        # we have an incomplete group, but the time is already up. run allreduce as is.
                        return await self._leader_begin_allreduce()
                    else:
                        await self._leader_disband_group()
                        continue  # re-declare averager with new expiration time
                finally:
                    request_candidates_task.cancel()
        finally:
            self._looking_for_group = False
            self._assembled_group = asyncio.Future()
            asyncio.create_task(self.dht.declare_averager(self.endpoint, group_identifier, declared_expiration_time,
                                                          looking_for_group=False, return_future=True))

    def _get_peer_stub(self, peer: Endpoint) -> averaging_pb2_grpc.DecentralizedAveragingStub:
        return ChannelCache.get_stub(peer, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)

    async def _request_join_group(self, leader: Endpoint, expiration_time: DHTExpiration) -> Optional[GroupAllReduce]:
        """
        :param leader: request this peer to be your leader for allreduce
        :param expiration_time: inform leader that we intend to begin averaging before this expiration_time
        :returns: if leader leader accepted us and started AllReduce, return that AllReduce. Otherwise, return None
        """
        assert self.pid == os.getpid(), f"this method can only be called from inside {self.__class__.__name__}"
        assert not self._accepted_to_group_as_follower.is_set(), "already accepted to another group as a follower"
        stream_call: Optional[StreamCallToLeader] = None
        try:
            async with self._lock_request_join_group:
                self._pending_leader_responded.clear()
                stream_call = self._get_peer_stub(leader).rpc_join_group(averaging_pb2.JoinRequest(
                    endpoint=self.endpoint, schema_hash=self.schema_hash, expiration=expiration_time))

                message = await stream_call.read()
                if message.code != averaging_pb2.ACCEPTED:
                    code = averaging_pb2.MessageCode.Name(message.code)
                    logger.debug(f"{self.endpoint} - requested {leader} to be my leader, but got rejected with {code}")
                    return None

                # else: we were accepted

                logger.debug(f"{self.endpoint} - joining the group of {leader}; waiting for peers")
                self._accepted_to_group_as_follower.set()
                await self._leader_disband_group(suggested_leader=leader)

                message = await stream_call.read()
                if message.code == averaging_pb2.BEGIN_ALLREDUCE:
                    assert all(isinstance(p, Endpoint) for p in message.ordered_group_endpoints)
                    return GroupAllReduce(
                        ordered_group_endpoints=message.ordered_group_endpoints, target_group_size=self.target_group_size,
                        local_tensors=self.averaged_tensors,
                    )
                else:
                    self.if_leader_sent_us_another_better_leader_than_we_should_connect_to_him_right_now()
                    logger.debug(f"{self.endpoint} - leader sent {averaging_pb2.MessageCode.Name(message.code)}")
                    return None

        finally:
            self._pending_leader_responded.set()
            if stream_call is not None:
                stream_call.cancel()

    async def rpc_join_group(self, request: averaging_pb2.JoinRequest, context: grpc.ServicerContext):
        """ A peer wants me to be his leader. I will coordinate his actions with the rest of my group. Maybe. """
        raise NotImplementedError("copy stuff from allreduce.py")
        if not self._looking_for_group:
            yield averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_LOOKING_FOR_GROUP)
            return
        async for message in self._forming_group.handle_join_request(request):
            yield message

    async def _follower_begin_allreduce(self, group_id: GroupID, ordered_group_endpoints: Sequence[Endpoint]) -> GroupAllReduce:
        logger.debug(f"{self.endpoint} - follower started allreduce after being prompted by leader.")
        assert self.endpoint in ordered_group_endpoints, "Leader sent us group_endpoints that does not contain us!"
        self._running_groups[group_id] = group_allreduce = GroupAllReduce(
            group_id=group_id, ordered_group_endpoints=ordered_group_endpoints, local_tensors=self.averaged_tensors)
        self._assembled_group.set_result(group_allreduce)
        return group_allreduce

    async def _leader_begin_allreduce(self) -> GroupAllReduce:
        group_id = DHTID.generate().to_bytes()
        ordered_group_endpoints = list(self._current_followers)
        ordered_group_endpoints.append(self.endpoint)
        logger.debug(f"{self.endpoint} - leader started allreduce with {len(ordered_group_endpoints)} followers.")
        self._running_groups[group_id] = group_allreduce = GroupAllReduce(
            group_id=group_id, ordered_group_endpoints=ordered_group_endpoints, local_tensors=self.averaged_tensors)
        self._assembled_group.set_result(group_allreduce)
        return group_allreduce

    async def _leader_disband_group(self, suggested_leader: Optional[Endpoint] = None):
        raise NotImplementedError("Notify all current followers that the group is disbanded")

    async def rpc_aggregate_part(self, request: averaging_pb2.AveragingData, context: grpc.ServicerContext):
        if request.group_id not in self._pending_groups and not self._pending_leader_responded.set():
            await self._pending_leader_responded.wait()  # this handles a special case when leader accepted us to group ...
            # AND began allreduce right away, but his response with group_id was delayed and other peers got to us first

        if request.group_id not in self._pending_groups:
            return averaging_pb2.AveragingData(code=averaging_pb2.PROTOCOL_VIOLATION)
        return await self._pending_groups[request.group_id].handle_accumulate_request(request)

    def __repr__(self):
        if len(self._running_groups) > 0:
            lfg_description = 'RUNNING_ALLREDUCE'
        elif not self._looking_for_group:
            lfg_description = 'NOT_LOOKING_FOR_GROUP'
        elif self._accepted_to_group_as_follower.is_set():
            lfg_description = 'WAITING_FOR_LEADER'
        else:
            lfg_description = 'LOOKING_FOR_GROUP'
        return f"{self.__class__.__name__}({self.endpoint}, {lfg_description})"


class GroupAllReduce:
    def __init__(self, **kwargs): pass


def split_into_parts(tensors: Sequence[torch.Tensor], group_size: int) -> Tuple[torch.Tensor]:
    """ combines averaged_tensors into one tensor and splits them into equal chunks of size group_size """
    flat_tensor = torch.cat(tuple(map(torch.Tensor.flatten, tensors)))
    chunk_slices = torch.linspace(start=0, end=len(flat_tensor), steps=group_size + 1, dtype=torch.int64)
    chunk_slices[-1] = len(flat_tensor)
    return tuple(torch.as_tensor(flat_tensor[chunk_slices[i]: chunk_slices[i + 1]]) for i in range(group_size))


def restore_from_parts(chunks: Sequence[torch.Tensor], shapes: Sequence[torch.Size]) -> Tuple[torch.Tensor, ...]:
    """ restores the original tensor shapes from chunks obtained by split_into_chunks """
    flat_tensor = torch.cat(list(chunks))
    result_sizes = tuple(map(torch.Size.numel, shapes))
    flat_original_tensors = torch.split_with_sizes(flat_tensor, result_sizes)
    return tuple(map(torch.Tensor.reshape, flat_original_tensors, shapes))


def compute_schema_hash(tensors: Sequence[torch.Tensor]) -> bytes:
    """ A hash that describes follower's tensor shapes, dtypes, devices, but not the actual values """
    schema_dicts = [{field_name: str(field_value)
                    for field_name, field_value in asdict(TensorDescriptor.from_tensor(tensor)).items()}
                    for tensor in tensors]
    return DHTID.generate(source=MSGPackSerializer.dumps(schema_dicts)).to_bytes()


def is_valid_join_request(request: averaging_pb2.PeerInfo) -> bool:
    assert len(request.ListFields()) == 3, "this function assumes JoinRequest has three fields, it should be updated"
    return (isinstance(request.schema_hash, bytes) and
            isinstance(request.expiration, DHTExpiration) and
            isinstance(request.endpoint, Endpoint))
