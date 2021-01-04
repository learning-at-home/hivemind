""" A background process that averages your tensors with peers """

from __future__ import annotations

import contextlib
import random
from dataclasses import asdict
from math import isfinite
from typing import Sequence, Optional, AsyncIterator, Set, Tuple
import asyncio

import torch
import grpc

import hivemind
from hivemind.client.averaging.allreduce import AllReduceRunner, GroupID
from hivemind.dht import DHTID, DHTExpiration, get_dht_time, GroupKey
from hivemind.utils import get_logger, Endpoint, TensorDescriptor, MSGPackSerializer, TimedStorage
from hivemind.proto import averaging_pb2, averaging_pb2_grpc, runtime_pb2
from hivemind.utils.grpc import ChannelCache

logger = get_logger(__file__)


class Matchmaking(averaging_pb2_grpc.DecentralizedAveragingServicer):
    f"""
    An internal class that is used to form groups of averages for running allreduce
    See DecentralizedAverager docstring for the detailed description of all parameters
    
    :note: on implementation: the current matchmaker protocol can encounter one type of (temporary) deadlock;
      This deadlock occurs when averager A requests averager B at the same time as averager B requests averager A.
      In that case, neither averager can process the other one's request because it is awaiting lock_request_join_group.
      This deadlock only happens if averagers have outdated information on expirations (due to network delays). 
      While A->B->A deadlock is easy to fix, it gets much harder with more peers (e.g. A -> B -> C -> D -> A).
      Hence, instead of accounting for such deadlocks, we simply break them with request_timeout.
    
    """

    def __init__(self, endpoint: Endpoint, averaged_tensors: Sequence[torch.Tensor], dht: hivemind.dht.DHT, *,
                 prefix: str, target_group_size: int, min_group_size: int, initial_group_bits: Optional[str] = None,
                 averaging_expiration: float = 15, request_timeout: Optional[float] = 3,
                 compression_type: runtime_pb2.CompressionType = runtime_pb2.NONE):
        assert '.' not in prefix, "group prefix must be a string without ."
        if request_timeout is None or request_timeout >= averaging_expiration:
            logger.warning("It is recommended to use request_timeout smaller than averaging_expiration. Otherwise,"
                           "matchmaking can cause deadlocks in some rare cases. Please see Matchmaking docstring.")

        super().__init__()
        self.dht, self.endpoint, self.averaged_tensors = dht, endpoint, tuple(averaged_tensors)
        self.prefix, self.group_bits = prefix, initial_group_bits
        self.target_group_size, self.min_group_size = target_group_size, min_group_size
        self.averaging_expiration, self.request_timeout = averaging_expiration, request_timeout
        self.compression_type = compression_type

        self.schema_hash = compute_schema_hash(self.averaged_tensors)

        self.lock_looking_for_group = asyncio.Lock()
        self.lock_request_join_group = asyncio.Lock()
        self.cond_notify_followers = asyncio.Condition()
        self.cond_follower_discarded = asyncio.Condition()
        self.assembled_group = asyncio.Future()

        self.current_leader: Optional[Endpoint] = None  # iff i am a follower, this is a link to my current leader
        self.current_followers: Set[Endpoint] = set()  # iff i am a leader, this contains my followers excluding myself
        self.potential_leaders = PotentialLeaders(endpoint, dht, averaging_expiration, target_group_size)

    @property
    def is_looking_for_group(self):
        return self.lock_looking_for_group.locked()

    @property
    def current_group_key(self) -> GroupKey:
        return f"{self.prefix}.0b{self.group_bits}"

    def __repr__(self):
        lfg_status = "looking for group," if self.is_looking_for_group else "not looking for group,"
        if self.is_looking_for_group:
            if self.current_leader:
                lfg_status += f" following {self.current_leader},"
            if len(self.current_followers):
                lfg_status += f" leading {len(self.current_followers)} followers,"
        schema_hash_repr = f"{self.schema_hash[0]}...{self.schema_hash[-8:]}"
        return f"{self.__class__.__name__}(endpoint={self.endpoint}, schema={schema_hash_repr}, {lfg_status}" \
               f" current key = {self.current_group_key})"

    async def look_for_group(self, *, timeout: Optional[float] = None) -> Optional[AllReduceRunner]:
        """
        :returns: an assembled group if successful, None if failed; does NOT perform the actual averaging
        Iterate over the averagers from a given group_identifier that have higher leadership priority than yourself.
        """
        if self.is_looking_for_group:
            logger.info("Another look_for_group is already in progress. The current run will be scheduled after"
                        " the existing group is either assembled or disbanded.")
        async with self.lock_looking_for_group:
            request_leaders_task = asyncio.create_task(self._request_join_potential_leaders(timeout))
            try:
                return await asyncio.wait_for(self.assembled_group, timeout=timeout)
            except asyncio.TimeoutError:
                return None

            except BaseException as e:
                if len(self.current_followers) > 0:
                    async with self.lock_request_join_group:
                        await self.leader_disband_group()
                if not self.assembled_group.done():
                    self.assembled_group.set_exception(e)
                raise

            finally:
                if not request_leaders_task.done():
                    request_leaders_task.cancel()
                if not self.assembled_group.done():
                    self.assembled_group.cancel()
                while len(self.current_followers) > 0:
                    async with self.cond_follower_discarded:
                        await self.cond_follower_discarded.wait()
                # note: the code above ensures that we send all followers away before creating new future
                self.assembled_group = asyncio.Future()

    async def _request_join_potential_leaders(self, timeout: Optional[float]) -> AllReduceRunner:
        """ Request leaders from queue until we find the first runner. This coroutine is meant to run in background. """
        async with self.potential_leaders.begin_search(self.current_group_key, timeout):
            # TODO update group_bits on success! reduce number of bits on not enough peers.
            # TODO after allreduce finishes, we may need to ask leader to notify lower keys about this
            # (so as to fix possible network partitioning if some peers operate on a much smaller nbits)
            while True:
                try:
                    next_leader = await self.potential_leaders.pop_next_leader()  # throws TimeoutError on expiration

                    group = await self.request_join_group(next_leader, self.potential_leaders.request_expiration_time)
                    if group is not None:
                        return group

                except asyncio.TimeoutError:
                    async with self.lock_request_join_group:
                        if self.assembled_group.done():
                            return self.assembled_group.result()
                        elif len(self.current_followers) + 1 >= self.min_group_size:
                            # the time is up, we have a *good enough* group. run allreduce as is.
                            return await self.leader_assemble_group()
                        elif len(self.current_followers) > 0:
                            await self.leader_disband_group()
                            # TODO maybe adjust grid size
                        continue
                except Exception as e:
                    if not self.assembled_group.done():
                        self.assembled_group.set_exception(e)
                    raise e

    async def request_join_group(self, leader: Endpoint, expiration_time: DHTExpiration) -> Optional[AllReduceRunner]:
        """
        :param leader: request this peer to be your leader for allreduce
        :param expiration_time: inform leader that we intend to begin averaging before this expiration_time
        :returns: if leader leader accepted us and started AllReduce, return that AllReduce. Otherwise, return None
        :note: this function does not guarantee that your group leader is the same as :leader: parameter
          The originally specified leader can disband group and redirect us to a different leader
        """
        assert self.is_looking_for_group and self.current_leader is None
        try:
            async with self.lock_request_join_group:
                leader_stub = ChannelCache.get_stub(leader, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)
                call = leader_stub.rpc_join_group(averaging_pb2.JoinRequest(
                    endpoint=self.endpoint, schema_hash=self.schema_hash, expiration=expiration_time))

                message = await asyncio.wait_for(call.read(), timeout=self.request_timeout)

                if message.code == averaging_pb2.ACCEPTED:
                    logger.debug(f"{self.endpoint} - joining the group of {leader}; waiting for peers")
                    self.current_leader = leader
                    if len(self.current_followers) > 0:
                        await self.leader_disband_group()

            if message.code != averaging_pb2.ACCEPTED:
                code = averaging_pb2.MessageCode.Name(message.code)
                logger.debug(f"{self.endpoint} - requested {leader} to be my leader, but got rejected with {code}")
                return None

            async with self.potential_leaders.pause_search():
                time_to_expiration = max(expiration_time - get_dht_time(), 0.0)
                message = await asyncio.wait_for(call.read(), time_to_expiration + self.request_timeout)

                if message.code == averaging_pb2.BEGIN_ALLREDUCE:
                    async with self.lock_request_join_group:
                        return await self.follower_assemble_group(
                            leader, message.group_id, message.ordered_group_endpoints)

            if message.code in (averaging_pb2.GROUP_DISBANDED, averaging_pb2.CANCELLED):
                if message.suggested_leader and message.suggested_leader != self.endpoint:
                    logger.debug(f"{self} - leader disbanded group and redirected us to {message.suggested_leader}")
                    self.current_leader = None
                    return await self.request_join_group(message.suggested_leader, expiration_time)
                else:
                    logger.debug(f"{self} - leader disbanded group")
                    return None

            logger.debug(f"{self} - unexpected message from leader: {averaging_pb2.MessageCode.Name(message.code)}")
            return None
        except asyncio.TimeoutError:
            logger.debug(f"{self} - leader did not respond within {self.request_timeout}")
            return None
        finally:
            self.current_leader = None

    async def rpc_join_group(self, request: averaging_pb2.JoinRequest, context: grpc.ServicerContext
                             ) -> AsyncIterator[averaging_pb2.MessageFromLeader]:
        """ accept or reject a join request from another averager; if accepted, run him through allreduce steps """
        try:
            async with self.lock_request_join_group:
                reason_to_reject = self._check_reasons_to_reject(request)
                if reason_to_reject is not None:
                    yield reason_to_reject
                    return

                self.current_followers.add(request.endpoint)
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.ACCEPTED)

                if len(self.current_followers) + 1 >= self.target_group_size and not self.assembled_group.done():
                    # outcome 1: we have assembled a full group and are ready for allreduce
                    await self.leader_assemble_group()

            if not self.assembled_group.done():
                try:
                    async with self.cond_notify_followers:
                        # wait for the group to be assembled or disbanded
                        timeout = max(0.0, self.potential_leaders.declared_expiration_time - get_dht_time())
                        await asyncio.wait_for(self.cond_notify_followers.wait(), timeout=timeout)
                except (asyncio.TimeoutError, RuntimeError):
                    async with self.lock_request_join_group:
                        if self.assembled_group.done():
                            pass  # this covers a rare case when the group is assembled while the event loop was busy.
                        elif len(self.current_followers) + 1 >= self.min_group_size and self.is_looking_for_group:
                            # outcome 2: the time is up, run allreduce with what we have or disband
                            await self.leader_assemble_group()
                        else:
                            await self.leader_disband_group()

            if self.assembled_group.cancelled() or not self.assembled_group.done() or\
                    request.endpoint not in self.assembled_group.result():
                if self.current_leader is not None:
                    # outcome 3: found by a leader with higher priority, send our followers to him
                    yield averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_DISBANDED,
                                                          suggested_leader=self.current_leader)
                    return
                else:
                    yield averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_DISBANDED)
                    return

            allreduce_group = self.assembled_group.result()
            yield averaging_pb2.MessageFromLeader(
                code=averaging_pb2.BEGIN_ALLREDUCE, group_id=allreduce_group.group_id,
                ordered_group_endpoints=allreduce_group.ordered_group_endpoints)

        except Exception as e:
            logger.exception(e)
            yield averaging_pb2.MessageFromLeader(code=averaging_pb2.INTERNAL_ERROR)

        finally:  # note: this code is guaranteed to run even if the coroutine is destroyed prematurely
            self.current_followers.discard(request.endpoint)
            async with self.cond_follower_discarded:
                self.cond_follower_discarded.notify()

    def _check_reasons_to_reject(self, request: averaging_pb2.JoinRequest) -> Optional[averaging_pb2.MessageFromLeader]:
        """ :returns: if accepted, return None, otherwise return a reason for rejection """
        if not self.is_looking_for_group or self.assembled_group.done():
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_LOOKING_FOR_GROUP)

        if request.ListFields() == 3 and not isinstance(request.schema_hash, bytes) or len(request.schema_hash) == 0 \
                or not isinstance(request.expiration, DHTExpiration) or not isfinite(request.expiration) \
                or not isinstance(request.endpoint, Endpoint) or len(request.endpoint) == 0:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.PROTOCOL_VIOLATION)

        elif request.schema_hash != self.schema_hash:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_SCHEMA_HASH)
        elif self.potential_leaders.declared_group_key is None:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_DECLARED)
        elif self.potential_leaders.declared_expiration_time > (request.expiration or float('inf')):
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_EXPIRATION_TIME)
        elif self.current_leader is not None:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_A_LEADER, suggested_leader=self.current_leader
                                                   ) # note: this suggested leader is currently ignored
        elif request.endpoint == self.endpoint or request.endpoint in self.current_followers:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.DUPLICATE_ENDPOINT)
        elif len(self.current_followers) + 1 >= self.target_group_size:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_IS_FULL)
        else:
            return None

    async def leader_assemble_group(self) -> AllReduceRunner:
        """ Form up all current followers into a group and prepare to _run_allreduce """
        assert self.lock_looking_for_group.locked() and self.lock_request_join_group.locked()
        assert not self.assembled_group.done()
        group_id = DHTID.generate().to_bytes()
        ordered_group_endpoints = list(self.current_followers)
        ordered_group_endpoints.append(self.endpoint)
        random.shuffle(ordered_group_endpoints)
        logger.debug(f"{self.endpoint} - leader started allreduce for {len(ordered_group_endpoints)} peers.")
        allreduce_group = AllReduceRunner(
            group_id=group_id, tensors=self.averaged_tensors, endpoint=self.endpoint,
            ordered_group_endpoints=ordered_group_endpoints, compression_type=self.compression_type)
        self.assembled_group.set_result(allreduce_group)
        async with self.cond_notify_followers:
            self.cond_notify_followers.notify_all()
        return allreduce_group

    async def follower_assemble_group(self, leader: Endpoint, group_id: GroupID,
                                      ordered_group_endpoints: Sequence[Endpoint]) -> AllReduceRunner:
        """ Prepare to run allreduce using a list of peers provided by our leader """
        assert self.lock_looking_for_group.locked() and self.lock_request_join_group.locked()
        assert not self.assembled_group.done()
        logger.debug(f"{self.endpoint} - follower started allreduce after being prompted by leader {leader}.")
        assert self.current_leader == leader, f"averager does not follow {leader} (actual: {self.current_leader})"
        assert self.endpoint in ordered_group_endpoints, "Leader sent us group_endpoints that does not contain us!"
        allreduce_group = AllReduceRunner(
            group_id=group_id, tensors=self.averaged_tensors, endpoint=self.endpoint,
            ordered_group_endpoints=ordered_group_endpoints, compression_type=self.compression_type)
        self.assembled_group.set_result(allreduce_group)
        async with self.cond_notify_followers:
            self.cond_notify_followers.notify_all()
        return allreduce_group

    async def leader_disband_group(self):
        """ Kick out all followers immediately, optionally direct them to our new leader (if we found one) """
        assert self.lock_request_join_group.locked()
        self.current_followers.clear()  # this will cause rpc_join_group to kick all followers out
        async with self.cond_notify_followers:
            self.cond_notify_followers.notify_all()


class PotentialLeaders:
    """ An utility class that searches for averagers that could become our leaders """
    def __init__(self, endpoint: Endpoint, dht: hivemind.DHT, averaging_expiration: DHTExpiration,
                 target_group_size: Optional[int]):
        self.endpoint, self.dht, self.averaging_expiration = endpoint, dht, averaging_expiration
        self.target_group_size = target_group_size
        self.running, self.update_triggered, self.update_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.declared_new_expiration, self.lock_declare_expiration = asyncio.Event(), asyncio.Lock()
        self.leader_queue = TimedStorage[Endpoint, DHTExpiration]()
        self.past_attempts: Set[Tuple[Endpoint, DHTExpiration]] = set()
        self.declared_expiration_time = float('inf')
        self.declared_group_key: Optional[GroupKey] = None
        self.max_assured_time = float('-inf')
        self.search_end_time = float('inf')

    @contextlib.asynccontextmanager
    async def begin_search(self, group_key: GroupKey, timeout: Optional[float]):
        assert not self.running.is_set(), "already running"
        self.running.set()
        self.search_end_time = get_dht_time() + timeout if timeout is not None else float('inf')
        update_queue_task = asyncio.create_task(self._update_queue_periodically(group_key))
        declare_averager_task = asyncio.create_task(self._declare_averager_periodically(group_key))
        try:
            yield self
        finally:
            if not update_queue_task.done():
                update_queue_task.cancel()
            if not declare_averager_task.done():
                declare_averager_task.cancel()
            for field in (self.past_attempts, self.leader_queue, self.running,
                          self.update_finished, self.update_triggered, self.declared_new_expiration):
                field.clear()
            self.max_assured_time = float('-inf')
            self.search_end_time = float('inf')

    @contextlib.asynccontextmanager
    async def pause_search(self):
        was_running = self.running.is_set()
        try:
            self.running.clear()
            yield
        finally:
            if was_running:
                self.running.set()
            else:
                self.running.clear()

    async def pop_next_leader(self) -> Endpoint:
        """ Remove and return the next most suitable leader or throw an exception if reached timeout """
        assert self.running.is_set(), "Not running search at the moment"
        while True:
            maybe_next_leader, entry = self.leader_queue.top()

            if maybe_next_leader is None or self.max_assured_time <= entry.expiration_time <= self.search_end_time:
                self.update_triggered.set()

            if maybe_next_leader is None or entry.expiration_time >= self.declared_expiration_time:
                await asyncio.wait({self.update_finished.wait(), self.declared_new_expiration.wait()},
                                   return_when=asyncio.FIRST_COMPLETED)
                self.declared_new_expiration.clear()
                if self.update_finished.is_set():
                    self.update_finished.clear()
                    continue
                else:
                    raise asyncio.TimeoutError("pop_next_leader was invalidated: re-declared averager in background")

            del self.leader_queue[maybe_next_leader]
            self.past_attempts.add((maybe_next_leader, entry.expiration_time))
            return maybe_next_leader

    @property
    def request_expiration_time(self) -> float:
        """ this averager's current expiration time - used to send join requests to leaders """
        if isfinite(self.declared_expiration_time):
            return self.declared_expiration_time
        else:
            return min(get_dht_time() + self.averaging_expiration, self.search_end_time)

    async def _update_queue_periodically(self, group_key: GroupKey):
        DISCREPANCY = hivemind.utils.timed_storage.MAX_DHT_TIME_DISCREPANCY_SECONDS
        while get_dht_time() < self.search_end_time:
            new_peers = await self.dht.get_averagers(group_key, only_active=True, return_future=True)
            self.max_assured_time = max(self.max_assured_time, get_dht_time() + self.averaging_expiration - DISCREPANCY)

            self.leader_queue.clear()
            for peer, peer_expiration_time in new_peers:
                if peer == self.endpoint or (peer, peer_expiration_time) in self.past_attempts:
                    continue
                self.leader_queue.store(peer, peer_expiration_time, peer_expiration_time)
                self.max_assured_time = max(self.max_assured_time, peer_expiration_time - DISCREPANCY)

            self.update_finished.set()

            await asyncio.wait(
                {self.running.wait(), self.update_triggered.wait()}, return_when=asyncio.ALL_COMPLETED,
                timeout=self.search_end_time - get_dht_time() if isfinite(self.search_end_time) else None)
            self.update_triggered.clear()

    async def _declare_averager_periodically(self, group_key: GroupKey):
        async with self.lock_declare_expiration:
            try:
                while True:
                    await self.running.wait()

                    new_expiration_time = min(get_dht_time() + self.averaging_expiration, self.search_end_time)
                    self.declared_group_key, self.declared_expiration_time = group_key, new_expiration_time
                    self.declared_new_expiration.set()
                    await self.dht.declare_averager(group_key, self.endpoint, new_expiration_time,
                                                    looking_for_group=True, return_future=True)
                    await asyncio.sleep(self.declared_expiration_time - get_dht_time())
            except Exception as e:  # note: we catch exceptions here because otherwise they are never printed
                logger.error(f"{self.endpoint} - caught {type(e)}: {e}")
            finally:
                if self.declared_group_key is not None:
                    prev_declared_key, prev_expiration_time = self.declared_group_key, self.declared_expiration_time
                    self.declared_group_key, self.declared_expiration_time = None, float('inf')
                    self.leader_queue, self.max_assured_time = TimedStorage[Endpoint, DHTExpiration](), float('-inf')
                    await self.dht.declare_averager(prev_declared_key, self.endpoint, prev_expiration_time,
                                                    looking_for_group=False, return_future=True)


def compute_schema_hash(tensors: Sequence[torch.Tensor]) -> bytes:
    """ A hash that describes follower's tensor shapes, dtypes, devices, but not the actual values """
    schema_dicts = [{field_name: str(field_value)
                    for field_name, field_value in asdict(TensorDescriptor.from_tensor(tensor)).items()}
                    for tensor in tensors]
    return DHTID.generate(source=MSGPackSerializer.dumps(schema_dicts)).to_bytes()
