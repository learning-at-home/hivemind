""" A background process that averages your tensors with peers """

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import random
from math import isfinite
from typing import AsyncIterator, Dict, Optional, Set, Tuple, Type

import numpy as np

from hivemind.averaging.control import StepControl
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.key_manager import GroupKey, GroupKeyManager
from hivemind.dht import DHT, DHTID, DHTExpiration
from hivemind.p2p import P2P, P2PContext, P2PHandlerError, PeerID, ServicerBase
from hivemind.p2p.p2p_daemon_bindings.utils import ControlFailure, DispatchFailure
from hivemind.proto import averaging_pb2
from hivemind.utils import TimedStorage, get_dht_time, get_logger, timed_storage
from hivemind.utils.asyncio import anext, cancel_and_wait

logger = get_logger(__name__)


class Matchmaking:
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

    def __init__(
        self,
        p2p: P2P,
        schema_hash: bytes,
        dht: DHT,
        *,
        servicer_type: Type[ServicerBase],
        prefix: str,
        target_group_size: Optional[int],
        min_group_size: int,
        min_matchmaking_time: float,
        request_timeout: float,
        client_mode: bool,
        initial_group_bits: str = "",
    ):
        assert "." not in prefix, "group prefix must be a string without ."
        if request_timeout is None or request_timeout >= min_matchmaking_time:
            logger.warning(
                "It is recommended to use request_timeout smaller than min_matchmaking_time. Otherwise,"
                " matchmaking can cause deadlocks in some rare cases. Please see Matchmaking docstring."
            )

        super().__init__()
        self._p2p = p2p

        if not issubclass(servicer_type, ServicerBase):
            raise TypeError("`servicer_type` is expected to be a ServicerBase subclass")
        self._servicer_type = servicer_type
        self._prefix = prefix

        self.peer_id = p2p.peer_id
        self.schema_hash = schema_hash
        self.group_key_manager = GroupKeyManager(dht, prefix, initial_group_bits, target_group_size)
        self.target_group_size, self.min_group_size = target_group_size, min_group_size
        self.min_matchmaking_time, self.request_timeout = min_matchmaking_time, request_timeout
        self.client_mode = client_mode

        self.lock_looking_for_group = asyncio.Lock()
        self.lock_request_join_group = asyncio.Lock()
        self.follower_was_discarded = asyncio.Event()
        self.was_accepted_to_group = asyncio.Event()
        self.assembled_group = asyncio.Future()

        self.current_leader: Optional[PeerID] = None  # iff i am a follower, this is a link to my current leader
        self.current_followers: Dict[PeerID, averaging_pb2.JoinRequest] = {}  # my current followers excluding myself
        self.potential_leaders = PotentialLeaders(self.peer_id, min_matchmaking_time, target_group_size)
        self.step_control: Optional[StepControl] = None

    @contextlib.asynccontextmanager
    async def looking_for_group(self, step_control: StepControl):
        async with self.lock_looking_for_group:
            assert self.step_control is None
            try:
                self.step_control = step_control
                yield
            finally:
                self.step_control = None

    @property
    def is_looking_for_group(self):
        return self.lock_looking_for_group.locked()

    def __repr__(self):
        lfg_status = "looking for group," if self.is_looking_for_group else "not looking for group,"
        if self.is_looking_for_group:
            if self.current_leader:
                lfg_status += f" following {self.current_leader},"
            if len(self.current_followers):
                lfg_status += f" leading {len(self.current_followers)} followers,"
        schema_hash_repr = f"{self.schema_hash[0]}...{self.schema_hash[-8:]}"
        return (
            f"{self.__class__.__name__}(peer_id={self.peer_id}, schema={schema_hash_repr}, {lfg_status}"
            f" current key = {self.group_key_manager.current_key}, client_mode={self.client_mode})"
        )

    async def look_for_group(self, step: StepControl) -> Optional[GroupInfo]:
        """
        :param step: step parameters and user control structure for the current step
        :returns: an assembled group if successful, None if failed; does NOT perform the actual averaging
        Iterate over the averagers from a given group_identifier that have higher leadership priority than yourself.
        """
        if self.is_looking_for_group:
            logger.info(
                "Another look_for_group is already in progress. The current run will be scheduled after"
                " the existing group is either assembled or disbanded."
            )
        async with self.looking_for_group(step):
            request_leaders_task = asyncio.create_task(self._request_join_potential_leaders(step))
            try:
                return await asyncio.wait_for(self.assembled_group, timeout=step.get_timeout())
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
                await cancel_and_wait(request_leaders_task)
                self.assembled_group.cancel()

                while len(self.current_followers) > 0:
                    await self.follower_was_discarded.wait()
                    self.follower_was_discarded.clear()
                # note: the code above ensures that we send all followers away before creating new future
                self.assembled_group = asyncio.Future()
                self.was_accepted_to_group.clear()

    async def _request_join_potential_leaders(self, step: StepControl) -> GroupInfo:
        """Request leaders from queue until we find the first runner. This coroutine is meant to run in background."""
        assert self.is_looking_for_group
        async with self.potential_leaders.begin_search(step, self.group_key_manager, declare=not self.client_mode):
            while True:
                try:
                    next_leader = await self.potential_leaders.pop_next_leader()  # throws TimeoutError on expiration

                    group = await self._request_join_group(next_leader)
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
                        continue
                except (concurrent.futures.CancelledError, asyncio.CancelledError):
                    break  # note: this is a compatibility layer for python3.7
                except Exception as e:
                    if not self.assembled_group.done():
                        self.assembled_group.set_exception(e)
                    raise e

    async def _request_join_group(self, leader: PeerID) -> Optional[GroupInfo]:
        """
        :param leader: request this peer to be your leader for allreduce
        :returns: if leader leader accepted us and started AllReduce, return that AllReduce. Otherwise, return None
        :note: this function does not guarantee that your group leader is the same as :leader: parameter
          The originally specified leader can disband group and redirect us to a different leader
        """
        assert self.is_looking_for_group and self.current_leader is None
        stream: Optional[AsyncIterator[averaging_pb2.MessageFromLeader]] = None
        try:
            async with self.lock_request_join_group:
                leader_stub = self._servicer_type.get_stub(self._p2p, leader, namespace=self._prefix)
                request_expiration_time = self.get_request_expiration_time()
                stream = await leader_stub.rpc_join_group(
                    averaging_pb2.JoinRequest(
                        schema_hash=self.schema_hash,
                        expiration=request_expiration_time,
                        client_mode=self.client_mode,
                        gather=self.step_control.data_for_gather,
                        group_key=self.group_key_manager.current_key,
                    )
                )
                message = await asyncio.wait_for(anext(stream), timeout=self.request_timeout)

                if message.code == averaging_pb2.ACCEPTED:
                    logger.debug(f"{self.peer_id} - joining the group of {leader}; waiting for peers")
                    self.current_leader = leader
                    self.was_accepted_to_group.set()
                    if len(self.current_followers) > 0:
                        await self.leader_disband_group()

            if message.code != averaging_pb2.ACCEPTED:
                code = averaging_pb2.MessageCode.Name(message.code)
                logger.debug(f"{self.peer_id} - requested {leader} to be my leader, but got rejected with {code}")
                return None

            async with self.potential_leaders.pause_search():
                time_to_expiration = max(0.0, request_expiration_time - get_dht_time())
                message = await asyncio.wait_for(anext(stream), time_to_expiration + self.request_timeout)

                if message.code == averaging_pb2.BEGIN_ALLREDUCE:
                    async with self.lock_request_join_group:
                        return await self.follower_assemble_group(leader, message)

            if message.code in (averaging_pb2.GROUP_DISBANDED, averaging_pb2.CANCELLED):
                if message.suggested_leader:
                    suggested_leader = PeerID(message.suggested_leader)
                    if suggested_leader != self.peer_id:
                        logger.debug(f"{self} - leader disbanded group and redirected us to {suggested_leader}")
                        self.current_leader = None
                        try:
                            await stream.aclose()
                        except RuntimeError as e:
                            logger.debug(e, exc_info=True)
                        return await self._request_join_group(suggested_leader)
                logger.debug(f"{self} - leader disbanded group")
                return None

            logger.debug(f"{self} - unexpected message from leader: {averaging_pb2.MessageCode.Name(message.code)}")
            return None
        except asyncio.TimeoutError:
            logger.debug(f"{self} - potential leader {leader} did not respond within {self.request_timeout}")
            return None
        except (P2PHandlerError, ControlFailure, DispatchFailure, StopAsyncIteration) as e:
            logger.debug(f"{self} - failed to request potential leader {leader}:")
            return None

        finally:
            self.was_accepted_to_group.clear()
            self.current_leader = None
            if stream is not None:
                try:
                    await stream.aclose()
                except RuntimeError as e:
                    logger.debug(e, exc_info=True)

    def get_request_expiration_time(self) -> float:
        """Returns the averager's current expiration time, which is used to send join requests to leaders"""
        if isfinite(self.potential_leaders.declared_expiration_time):
            return self.potential_leaders.declared_expiration_time
        else:
            scheduled_time = max(self.step_control.scheduled_time, get_dht_time() + self.min_matchmaking_time)
            return min(scheduled_time, self.potential_leaders.search_end_time)

    async def rpc_join_group(
        self, request: averaging_pb2.JoinRequest, context: P2PContext
    ) -> AsyncIterator[averaging_pb2.MessageFromLeader]:
        """accept or reject a join request from another averager; if accepted, run him through allreduce steps"""
        try:
            async with self.lock_request_join_group:
                reason_to_reject = self._check_reasons_to_reject(request, context)
                if reason_to_reject is not None:
                    yield reason_to_reject
                    return

                self.current_followers[context.remote_id] = request
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.ACCEPTED)

                if (
                    self.target_group_size is not None
                    and len(self.current_followers) + 1 >= self.target_group_size
                    and not self.assembled_group.done()
                ):
                    # outcome 1: we have assembled a full group and are ready for allreduce
                    await self.leader_assemble_group()

            # wait for the group to be assembled or disbanded
            timeout = max(0.0, self.potential_leaders.declared_expiration_time - get_dht_time())
            await asyncio.wait(
                {self.assembled_group, self.was_accepted_to_group.wait()},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout,
            )
            if not self.assembled_group.done() and not self.was_accepted_to_group.is_set():
                async with self.lock_request_join_group:
                    if self.assembled_group.done():
                        pass  # this covers a rare case when the group is assembled while the event loop was busy.
                    elif len(self.current_followers) + 1 >= self.min_group_size and self.is_looking_for_group:
                        # outcome 2: the time is up, run allreduce with what we have or disband
                        await self.leader_assemble_group()
                    else:
                        await self.leader_disband_group()

            if (
                self.was_accepted_to_group.is_set()
                or not self.assembled_group.done()
                or self.assembled_group.cancelled()
                or context.remote_id not in self.assembled_group.result()
            ):
                if self.current_leader is not None:
                    # outcome 3: found by a leader with higher priority, send our followers to him
                    yield averaging_pb2.MessageFromLeader(
                        code=averaging_pb2.GROUP_DISBANDED, suggested_leader=self.current_leader.to_bytes()
                    )
                    return
                else:
                    yield averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_DISBANDED)
                    return

            group_info = self.assembled_group.result()
            yield averaging_pb2.MessageFromLeader(
                code=averaging_pb2.BEGIN_ALLREDUCE,
                group_id=group_info.group_id,
                ordered_peer_ids=[item.to_bytes() for item in group_info.peer_ids],
                gathered=group_info.gathered,
            )
        except (concurrent.futures.CancelledError, asyncio.CancelledError):
            return  # note: this is a compatibility layer for python3.7
        except Exception as e:
            logger.exception(e)
            yield averaging_pb2.MessageFromLeader(code=averaging_pb2.INTERNAL_ERROR)

        finally:  # note: this code is guaranteed to run even if the coroutine is destroyed prematurely
            self.current_followers.pop(context.remote_id, None)
            self.follower_was_discarded.set()

    def _check_reasons_to_reject(
        self, request: averaging_pb2.JoinRequest, context: P2PContext
    ) -> Optional[averaging_pb2.MessageFromLeader]:
        """:returns: if accepted, return None, otherwise return a reason for rejection"""
        if not self.is_looking_for_group or self.assembled_group.done():
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_LOOKING_FOR_GROUP)

        if (
            request.ListFields() == 3
            and not isinstance(request.schema_hash, bytes)
            or len(request.schema_hash) == 0
            or not isinstance(request.expiration, DHTExpiration)
            or not isfinite(request.expiration)
            or self.client_mode
            or not isinstance(request.group_key, GroupKey)
        ):
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.PROTOCOL_VIOLATION)

        elif request.schema_hash != self.schema_hash:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_SCHEMA_HASH)
        elif request.group_key != self.group_key_manager.current_key:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_GROUP_KEY)
        elif self.potential_leaders.declared_group_key is None:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_DECLARED)
        elif self.potential_leaders.declared_expiration_time > (request.expiration or float("inf")):
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_EXPIRATION_TIME)
        elif self.current_leader is not None:
            return averaging_pb2.MessageFromLeader(
                code=averaging_pb2.NOT_A_LEADER, suggested_leader=self.current_leader.to_bytes()
            )
        elif context.remote_id == self.peer_id or context.remote_id in self.current_followers:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.DUPLICATE_PEER_ID)
        elif self.target_group_size is not None and len(self.current_followers) + 1 >= self.target_group_size:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_IS_FULL)
        else:
            return None

    async def leader_assemble_group(self) -> GroupInfo:
        """Form up all current followers into a group and gather metadata"""
        assert self.lock_looking_for_group.locked() and self.lock_request_join_group.locked() and not self.client_mode
        assert not self.assembled_group.done()
        group_id = DHTID.generate().to_bytes()  # note: both groupd_id and the order of peer_ids must be random
        ordered_peer_ids = list(self.current_followers)
        ordered_peer_ids.append(self.peer_id)
        random.shuffle(ordered_peer_ids)

        gathered = tuple(
            self.step_control.data_for_gather if peer_id == self.peer_id else self.current_followers[peer_id].gather
            for peer_id in ordered_peer_ids
        )

        logger.debug(f"{self.peer_id} - assembled group of {len(ordered_peer_ids)} peers")
        group_info = GroupInfo(group_id, tuple(ordered_peer_ids), gathered)
        await self.group_key_manager.update_key_on_group_assembled(group_info, is_leader=True)
        self.assembled_group.set_result(group_info)
        return group_info

    async def follower_assemble_group(self, leader: PeerID, msg: averaging_pb2.MessageFromLeader) -> GroupInfo:
        """Form a group from using peers and metadata provided by our leader"""
        assert self.lock_looking_for_group.locked() and self.lock_request_join_group.locked()
        assert not self.assembled_group.done()
        assert self.current_leader == leader, f"averager does not follow {leader} (actual: {self.current_leader})"

        group_id = msg.group_id
        ordered_peer_ids = [PeerID(item) for item in msg.ordered_peer_ids]
        assert self.peer_id in ordered_peer_ids, "Leader sent us group_peer_ids that does not contain us!"
        assert len(ordered_peer_ids) == len(msg.gathered)

        logger.debug(f"{self.peer_id} - follower assembled group with leader {leader}")
        group_info = GroupInfo(group_id, tuple(ordered_peer_ids), tuple(msg.gathered))
        await self.group_key_manager.update_key_on_group_assembled(group_info)
        self.assembled_group.set_result(group_info)
        return group_info

    async def leader_disband_group(self):
        """Kick out all followers immediately, optionally direct them to our new leader (if we found one)"""
        assert self.lock_request_join_group.locked() and not self.client_mode
        self.current_followers.clear()  # this will cause rpc_join_group to kick all followers out


class PotentialLeaders:
    """An utility class that searches for averagers that could become our leaders"""

    def __init__(self, peer_id: PeerID, min_matchmaking_time: DHTExpiration, target_group_size: Optional[int]):
        self.peer_id, self.min_matchmaking_time = peer_id, min_matchmaking_time
        self.target_group_size = target_group_size
        self.running, self.update_triggered, self.update_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.declared_expiration, self.lock_search, self.lock_declare = asyncio.Event(), asyncio.Lock(), asyncio.Lock()
        self.leader_queue = TimedStorage[PeerID, DHTExpiration]()
        self.past_attempts: Set[Tuple[PeerID, DHTExpiration]] = set()
        self.declared_expiration_time = float("inf")
        self.declared_group_key: Optional[GroupKey] = None
        self.max_assured_time = float("-inf")
        self.search_end_time = float("inf")

    @contextlib.asynccontextmanager
    async def begin_search(self, step: StepControl, key_manager: GroupKeyManager, declare: bool = True):
        async with self.lock_search:
            self.running.set()
            self.search_end_time = step.deadline if step.deadline is not None else float("inf")
            update_queue_task = asyncio.create_task(self._update_queue_periodically(key_manager))
            if declare:
                declare_averager_task = asyncio.create_task(self._declare_averager_periodically(step, key_manager))

            try:
                yield self
            finally:
                await cancel_and_wait(update_queue_task)
                if declare:
                    await cancel_and_wait(declare_averager_task)

                for field in (
                    self.past_attempts,
                    self.leader_queue,
                    self.running,
                    self.update_finished,
                    self.update_triggered,
                    self.declared_expiration,
                ):
                    field.clear()
                self.max_assured_time = float("-inf")
                self.search_end_time = float("inf")

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

    async def pop_next_leader(self) -> PeerID:
        """Remove and return the next most suitable leader or throw an exception if reached timeout"""
        assert self.running.is_set(), "Not running search at the moment"
        while True:
            maybe_next_leader, entry = self.leader_queue.top()

            if maybe_next_leader is None or self.max_assured_time <= entry.expiration_time <= self.search_end_time:
                self.update_triggered.set()

            if maybe_next_leader is None or (entry.expiration_time, maybe_next_leader.to_bytes()) > (
                self.declared_expiration_time,
                self.peer_id.to_bytes(),
            ):
                await asyncio.wait(
                    {self.update_finished.wait(), self.declared_expiration.wait()}, return_when=asyncio.FIRST_COMPLETED
                )
                self.declared_expiration.clear()
                if self.update_finished.is_set():
                    self.update_finished.clear()
                    continue
                else:
                    raise asyncio.TimeoutError("pop_next_leader was invalidated: re-declared averager in background")

            del self.leader_queue[maybe_next_leader]
            self.past_attempts.add((maybe_next_leader, entry.expiration_time))
            return maybe_next_leader

    async def _update_queue_periodically(self, key_manager: GroupKeyManager) -> None:
        DISCREPANCY = timed_storage.MAX_DHT_TIME_DISCREPANCY_SECONDS
        while get_dht_time() < self.search_end_time:
            new_peers = await key_manager.get_averagers(key_manager.current_key, only_active=True)
            self.max_assured_time = max(
                self.max_assured_time, get_dht_time() + self.min_matchmaking_time - DISCREPANCY
            )

            self.leader_queue.clear()
            for peer, peer_expiration_time in new_peers:
                if peer == self.peer_id or (peer, peer_expiration_time) in self.past_attempts:
                    continue
                self.leader_queue.store(peer, peer_expiration_time, peer_expiration_time)
                self.max_assured_time = max(self.max_assured_time, peer_expiration_time - DISCREPANCY)

            self.update_finished.set()

            await asyncio.wait(
                {self.running.wait(), self.update_triggered.wait()},
                return_when=asyncio.ALL_COMPLETED,
                timeout=self.search_end_time - get_dht_time() if isfinite(self.search_end_time) else None,
            )
            self.update_triggered.clear()

    async def _declare_averager_periodically(self, step: StepControl, key_manager: GroupKeyManager) -> None:
        async with self.lock_declare:
            try:
                while True:
                    await self.running.wait()
                    new_expiration_time = float(
                        min(max(step.scheduled_time, get_dht_time() + self.min_matchmaking_time), self.search_end_time)
                    )
                    self.declared_group_key = group_key = key_manager.current_key
                    self.declared_expiration_time = new_expiration_time
                    self.declared_expiration.set()
                    await key_manager.declare_averager(group_key, self.peer_id, expiration_time=new_expiration_time)
                    await asyncio.sleep(self.declared_expiration_time - get_dht_time())
                    if self.running.is_set() and len(self.leader_queue) == 0:
                        await key_manager.update_key_on_not_enough_peers()
            finally:
                if self.declared_group_key is not None:
                    prev_declared_key, prev_expiration_time = self.declared_group_key, self.declared_expiration_time
                    self.declared_group_key, self.declared_expiration_time = None, float("inf")
                    self.leader_queue, self.max_assured_time = TimedStorage[PeerID, DHTExpiration](), float("-inf")
                    await key_manager.declare_averager(
                        prev_declared_key, self.peer_id, prev_expiration_time, looking_for_group=False
                    )


class MatchmakingException(Exception):
    """An internal exception that marks undesired edge cases during averaging"""
