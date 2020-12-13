""" A background process that averages your tensors with peers """

from __future__ import annotations

from dataclasses import asdict
from math import isfinite
from typing import Sequence, Optional, AsyncIterator, Set
import asyncio

import torch
import grpc

import hivemind
from hivemind.client.averaging.allreduce import GroupAllReduce, GroupID
from hivemind.dht import DHTID, DHTExpiration, get_dht_time, GroupKey
from hivemind.utils import get_logger, Endpoint, TensorDescriptor, MSGPackSerializer, TimedStorage
from hivemind.proto import averaging_pb2, averaging_pb2_grpc, runtime_pb2
from hivemind.utils.grpc import ChannelCache


GROUP_NBITS_INTERVAL = 3
logger = get_logger(__file__)


class Matchmaking(averaging_pb2_grpc.DecentralizedAveragingServicer):
    """
    An internal class that is used to form groups of averages for running allreduce
    TODO docstring
    """
    def __init__(self, averager_endpoint: Endpoint, averaged_tensors: Sequence[torch.Tensor], dht: hivemind.dht.DHT, *,
                 prefix: str, target_group_size: int, min_group_size: int = 1, initial_group_bits: Optional[str] = None,
                 averaging_expiration: float = 15):
        assert '.' not in prefix, "group prefix must be a string without ."

        super().__init__()
        self.dht, self.endpoint, self.averaged_tensors = dht, averager_endpoint, tuple(averaged_tensors)
        self.prefix, self.group_bits = prefix, initial_group_bits
        self.target_group_size, self.min_group_size = target_group_size, min_group_size
        self.averaging_expiration = averaging_expiration

        self.schema_hash = compute_schema_hash(self.averaged_tensors)

        self.lock_looking_for_group = asyncio.Lock()
        self.lock_request_join_group = asyncio.Lock()
        self.assembled_group = asyncio.Future()
        self.cond_notify_followers = asyncio.Condition()

        self.current_leader: Optional[Endpoint] = None  # iff i am a follower, this is a link to my current leader
        self.current_followers: Set[Endpoint] = set()  # iff i am a leader, this contains my followers excluding myself

        self.declared_expiration_time = -float('inf')  # iff i am looking for group, this is my latest expiration time
        self.declared_group_key: Optional[GroupKey] = None

        self.leader_queue = LeaderQueue()
        self.max_assured_time = float('-inf')  # all averagers below this expiration_time are in leader_queue

    def __repr__(self):
        lfg_status = "looking for group," if self.lock_looking_for_group.locked() else "not looking for group,"
        if self.lock_looking_for_group.locked():
            if self.current_leader:
                lfg_status += f" following {self.current_leader},"
            if len(self.current_followers):
                lfg_status += f" leading {len(self.current_followers)} followers,"
        declared_status = f"declared to DHT: {self.declared_expiration_time >= get_dht_time()}"
        schema_hash_repr = f"{self.schema_hash[0]}...{self.schema_hash[-8:]}"
        return f"{self.__class__.__name__}(endpoint={self.endpoint}, schema={schema_hash_repr}, {lfg_status}" \
               f" current key = {self.prefix}.{self.group_bits}, {declared_status})"

    async def look_for_group(self, *, timeout: Optional[float] = None) -> GroupAllReduce:
        """
        :returns: an assembled group if successful, None if failed; does NOT perform the actual averaging
        Iterate over the averagers from a given group_identifier that have higher leadership priority than yourself.
        """
        if self.lock_looking_for_group.locked():
            logger.debug("Another look_for_group is already in progress. The current run will be scheduled after"
                         " the existing group is either assembled or botched.")
        async with self.lock_looking_for_group:
            end_time = get_dht_time() + (timeout or float('inf'))

            group_key = f"{self.prefix}.{self.group_bits}"
            # TODO update group_bits on success! reduce number of bits on not enough peers.
            # TODO after allreduce finishes, we may need to ask leader to notify lower keys about this
            # (so as to fix possible network partitioning if some peers operate on a much smaller nbits)

            try:
                while True:
                    # step 1: declare the averager to DHT, make it visible for other averagers
                    await self.publish_averager(group_key)

                    # step 2: request potential leaders until first accept OR until we are chosen as a leader
                    request_candidates_task = asyncio.create_task(self.request_join_potential_leaders(group_key, ))

                    try:  # wait until we are ready to run allreduce (as either follower or leader) or reach expiration
                        timeout = min(end_time, self.declared_expiration_time) - get_dht_time()
                        return await asyncio.wait_for(self.assembled_group, timeout if isfinite(timeout) else None)

                    except asyncio.TimeoutError:
                        if len(self.current_followers) >= self.min_group_size:
                            # the time is up, we have a *good enough* group. run allreduce as is.
                            return await self.leader_assemble_group()
                        else:
                            await self.leader_disband_group()
                            # TODO maybe adjust grid size
                            continue  # re-declare averager with new expiration time
                    finally:
                        request_candidates_task.cancel()

            except Exception as e:
                if len(self.current_followers) > 0:
                    await self.leader_disband_group()
                self.assembled_group.set_exception(e)
            finally:
                asyncio.create_task(self.unpublish_averager())
                if self.assembled_group.done():
                    self.assembled_group = asyncio.Future()

    async def publish_averager(self, group_key: GroupKey) -> Optional[DHTExpiration]:
        """ Subscribe thyself to a given group key and become visible to other averagers """
        if group_key != self.declared_group_key:
            await self.unpublish_averager()

        new_expiration_time = get_dht_time() + self.averaging_expiration
        stored_ok = await self.dht.declare_averager(self.endpoint, group_key, new_expiration_time,
                                                    looking_for_group=True, return_future=True)
        if stored_ok:
            self.declared_expiration_time, self.declared_group_key = new_expiration_time, group_key
            return new_expiration_time
        else:
            logger.warning(f"failed to subscribe to group {group_key} : store rejected by DHT peers")
            return None

    async def unpublish_averager(self):
        """ Remove the previously published entries from the DHT """
        declared_expiration_time, declared_group_key = self.declared_expiration_time, self.declared_group_key
        if isinstance(declared_group_key, GroupKey):
            self.declared_expiration_time, self.declared_group_key = float('-inf'), None
            self.leader_queue, self.max_assured_time = LeaderQueue(), float('-inf')
            await self.dht.declare_averager(declared_group_key, self.endpoint,
                                            expiration_time=declared_expiration_time,
                                            looking_for_group=False, return_future=True)
        else:
            logger.debug("unpublish_averager has no effect: not published.")

    async def request_join_potential_leaders(self, group_key: GroupKey, timeout: Optional[float] = None
                                             ) -> Optional[GroupAllReduce]:
        """ Find peers in a given DHT key that might accept us as a follower """
        end_time = self.declared_expiration_time
        allowed_discrepancy = hivemind.utils.timed_storage.MAX_DHT_TIME_DISCREPANCY_SECONDS
        if timeout is not None:
            end_time = min(self.declared_expiration_time, get_dht_time() + timeout)
        assert isfinite(end_time)

        while True:
            maybe_next_leader, maybe_next_expiration = self.leader_queue.top()[1] or (None, None)
            if (maybe_next_expiration or get_dht_time()) > self.max_assured_time:
                # if there's a chance that DHT contains averagers newer than the earliest one from our local heap,
                # ... then fetch more candidates from the DHT until we are confident we know the next-earliest peer
                new_peers = await self.dht.get_averagers(group_key, only_active=True, return_future=True)
                self.max_assured_time = max(self.max_assured_time, get_dht_time() - allowed_discrepancy)
                for peer, peer_expiration_time in new_peers:
                    self.leader_queue.store(peer, peer_expiration_time, peer_expiration_time)
                    self.max_assured_time = max(self.max_assured_time, peer_expiration_time - allowed_discrepancy)
                continue

            if maybe_next_expiration is None or maybe_next_expiration >= self.declared_expiration_time:
                break  # no potential leaders are available until our expiration AND we can't fetch more

            if maybe_next_expiration < get_dht_time():
                continue  # this leader expired before we could request to join his group
            if maybe_next_leader is None or maybe_next_leader == self.endpoint:
                continue  # do not count myself as a potential leader for me (even at a later expiration time)

            maybe_group_allreduce = await self.request_join_group(maybe_next_leader, self.declared_expiration_time)
            if maybe_group_allreduce is not None:
                return maybe_group_allreduce

    async def request_join_group(self, leader: Endpoint, expiration_time: DHTExpiration) -> Optional[GroupAllReduce]:
        """
        :param leader: request this peer to be your leader for allreduce
        :param expiration_time: inform leader that we intend to begin averaging before this expiration_time
        :returns: if leader leader accepted us and started AllReduce, return that AllReduce. Otherwise, return None
        """
        assert self.lock_looking_for_group.locked() and self.current_leader is None
        call: Optional[grpc.aio.UnaryStreamCall[averaging_pb2.JoinRequest, averaging_pb2.MessageFromLeader]] = None
        try:
            async with self.lock_request_join_group:
                leader_stub = ChannelCache.get_stub(leader, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)
                call = leader_stub.rpc_join_group(averaging_pb2.JoinRequest(
                    endpoint=self.endpoint, schema_hash=self.schema_hash, expiration=expiration_time))

                message = await call.read()  # TODO use timeout?
                if message.code != averaging_pb2.ACCEPTED:
                    code = averaging_pb2.MessageCode.Name(message.code)
                    logger.debug(f"{self.endpoint} - requested {leader} to be my leader, but got rejected with {code}")
                    return None

                # else: we were accepted
                logger.debug(f"{self.endpoint} - joining the group of {leader}; waiting for peers")
                self.current_leader = leader
                await self.leader_disband_group()

            message = await call.read()
            if message.code == averaging_pb2.BEGIN_ALLREDUCE:
                return await self.follower_assemble_group(leader, message.group_id, message.ordered_group_endpoints)
            elif message.code == averaging_pb2.GROUP_DISBANDED and bool(message.suggested_leader):
                logger.debug(f"{self} - leader disbanded group and redirected us to {message.suggested_leader}")
                return await self.request_join_group(message.suggested_leader, expiration_time)

            else:
                logger.debug(f"{self} - leader sent {averaging_pb2.MessageCode.Name(message.code)}, leaving group")
                return None

        finally:
            self.current_leader = None
            if call is not None:
                call.cancel()

    async def rpc_join_group(self, request: averaging_pb2.JoinRequest, context: grpc.ServicerContext
                             ) -> AsyncIterator[averaging_pb2.MessageFromLeader]:
        """ accept or reject a join request from another averager; if accepted, run him through allreduce steps """
        try:
            # stage 1: check if there is a reason to reject a peer outright
            if not self.lock_looking_for_group.locked():
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_LOOKING_FOR_GROUP)
                return
            if not is_valid_join_request(request):
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.PROTOCOL_VIOLATION)
                return
            if request.schema_hash != self.schema_hash:
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_SCHEMA_HASH)
                return
            if self.declared_expiration_time == float('-inf'):
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_DECLARED)
                return
            if self.declared_expiration_time > (request.expiration or float('inf')):
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_EXPIRATION_TIME)
                return
            if self.current_leader is not None:
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_A_LEADER,
                                                      suggested_leader=self.current_leader)
                return
            if request.endpoint == self.endpoint or request.endpoint in self.current_followers:
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.DUPLICATE_ENDPOINT)
                return
            if len(self.current_followers) + 1 >= self.target_group_size:
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_IS_FULL)
                return

            async with self.lock_request_join_group:
                # stage 2: if there are no red flags, accept peer as your follower
                self.current_followers.add(request.endpoint)
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.ACCEPTED)

                if len(self.current_followers) + 1 >= self.target_group_size:
                    # outcome 1: if everything went well, we have assembled a full group and are ready for allreduce
                    await self.leader_assemble_group()  # note: this will trigger self._assembled_group

            # stage 3: wait for the group to be assembled (or for the follower to leave, whichever comes first)
            current_group = self.assembled_group  # copy current assembled_group to avoid overwriting
            async with self.cond_notify_followers:
                await asyncio.wait_for(self.cond_notify_followers.wait(),
                                       timeout=max(0.0, self.declared_expiration_time - get_dht_time()))

            if self.current_leader is not None:
                # outcome 2: we were accepted to another averager's group => send all followers to our new leader
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_DISBANDED,
                                                      suggested_leader=self.current_leader)
                return

            if not current_group.done():
                # outcome 3: the time is up, we have *enough* followers => run allreduce
                if len(self.current_followers) + 1 >= self.min_group_size:
                    await self.leader_assemble_group()
                else:  # ... or disband if not enough followers
                    await self.leader_disband_group()

            if request.endpoint not in self.current_followers or self.assembled_group is not current_group:
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_DISBANDED)
                return

            # finally, run allreduce
            group_allreduce = current_group.result()
            yield averaging_pb2.MessageFromLeader(
                code=averaging_pb2.BEGIN_ALLREDUCE, group_id=group_allreduce.group_id,
                ordered_group_endpoints=group_allreduce.ordered_group_endpoints)

        except Exception as e:
            logger.exception(e)
            yield averaging_pb2.MessageFromLeader(code=averaging_pb2.INTERNAL_ERROR)

        finally:  # note: this code is guaranteed to run even if the coroutine is destroyed prematurely
            self.current_followers.discard(request.endpoint)

    async def leader_assemble_group(self) -> GroupAllReduce:
        """ Form up all current followers into a group and prepare to _run_allreduce  """
        group_id = DHTID.generate().to_bytes()
        ordered_group_endpoints = list(self.current_followers)
        ordered_group_endpoints.append(self.endpoint)
        logger.debug(f"{self.endpoint} - leader started allreduce with {len(ordered_group_endpoints)} followers.")
        async with self.lock_request_join_group:
            group_allreduce = GroupAllReduce(group_id=group_id, tensors=self.averaged_tensors, endpoint=self.endpoint,
                                             ordered_group_endpoints=ordered_group_endpoints)
            self.assembled_group.set_result(group_allreduce)
            async with self.cond_notify_followers:
                self.cond_notify_followers.notify_all()
            return group_allreduce

    async def follower_assemble_group(self, leader: Endpoint, group_id: GroupID,
                                      ordered_group_endpoints: Sequence[Endpoint]) -> GroupAllReduce:
        """ Prepare to run allreduce using a list of peers provided by our leader """
        logger.debug(f"{self.endpoint} - follower started allreduce after being prompted by leader {leader}.")
        assert self.current_leader == leader, f"averager does not follow {leader} (actual: {self.current_leader})"
        assert self.endpoint in ordered_group_endpoints, "Leader sent us group_endpoints that does not contain us!"
        async with self.lock_request_join_group:
            group_allreduce = GroupAllReduce(group_id=group_id, tensors=self.averaged_tensors, endpoint=self.endpoint,
                                             ordered_group_endpoints=ordered_group_endpoints)
            self.assembled_group.set_result(group_allreduce)
            async with self.cond_notify_followers:
                self.cond_notify_followers.notify_all()
            return group_allreduce

    async def leader_disband_group(self):
        """ Cancel all followers """
        assert self.lock_looking_for_group.locked(), "can not disband: no longer looking for group"
        async with self.lock_request_join_group:
            for follower in list(self.current_followers):
                self.current_followers.discard(follower)  # this will cause rpc_foin_group to kick the follower out
            async with self.cond_notify_followers:
                self.cond_notify_followers.notify_all()


class LeaderQueue(TimedStorage[Endpoint, DHTExpiration]):
    """ A queue of averagers that were found in currently declared group key """
    frozen = True


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
