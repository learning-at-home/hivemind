""" This file contains a state machine that defines DecentralizedAverager protocol """
from __future__ import annotations
import asyncio
import random
from typing import List, Set, Optional, Sequence, Tuple, Any
from dataclasses import dataclass, field

import torch

from hivemind.utils import Endpoint
from hivemind.dht import DHTID, DHTExpiration


@dataclass
class AveragingOutcome:
    """ A data structure that encodes the outcome of a single attempt at group all-reduce """
    message: Any = None
    was_accepted_to_group: bool = False
    was_leader: bool = False
    suggested_leader: Optional[Endpoint] = None
    canceled_looking_for_group: bool = False
    group_dismissed_by_leader: bool = False
    started_allreduce: bool = False
    succeeded_allreduce: bool = False
    finished: asyncio.Event = field(default_factory=asyncio.Event, init=False)

    async def wait(self):
        return await self.finished.wait()


@dataclass(init=False)
class ProtocolState:
    """ The current state of the averaging protocol with auxiliary variables (base class for all states) """
    def __init__(self):  # note: we could use ABC, but it would be too verbose
        raise ValueError("Please use one of the subclasses")

    # mandatory fields for all states
    _is_active_state: bool = field(default=False, init=False)
    outcome: AveragingOutcome = field(init=False)

    @staticmethod
    def transition(method):
        """
        Decorator: when a transition is called, its output state is marked as active and current state becomes inactive.
        A single state can only perform one transition before it becomes inactive.
        """
        def transition_method(state: ProtocolState, *args, **kwargs):
            assert state._is_active_state, f"State {state} is not the active state, no longer able to transition"
            new_state: ProtocolState = method(state, *args, **kwargs)
            state._is_active_state, new_state._is_active_state = False, True
            return new_state
        return transition_method


@dataclass
class Idle(ProtocolState):
    """ the averager is running, but not interested in averaging at this time """
    _is_active_state: bool = field(default=True, init=False)  # this is the entry state, it becomes active by default
    outcome: AveragingOutcome = field(default_factory=AveragingOutcome, init=False)
    # the outcome field is created here and transferred to all subsequent states until finished or failed

    @ProtocolState.transition
    def look_for_group(self, my_endpoint: Endpoint, my_expiration: DHTExpiration) -> LookingForGroup:
        return LookingForGroup(my_endpoint, my_expiration, outcome=self.outcome)


@dataclass
class LookingForGroup(ProtocolState):
    """ i am currently looking for a group in DHT """
    my_endpoint: Endpoint
    my_expiration: DHTExpiration
    outcome: AveragingOutcome
    one_request_at_a_time: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @ProtocolState.transition
    def become_follower(self, leader_endpoint: Endpoint, group_id: bytes) -> FollowerWaitingForLeader:
        self.outcome.was_accepted_to_group = True
        return FollowerWaitingForLeader(my_endpoint=self.my_endpoint, my_expiration=self.my_expiration,
                                        leader_endpoint=leader_endpoint, group_id=group_id, outcome=self.outcome)

    @ProtocolState.transition
    def become_leader(self, group_id: Optional[bytes] = None, max_size: Optional[int] = None) -> LeaderWaitingForFollowers:
        if group_id is None:
            # note: we generate group_id as DHTID for convenience. Do not assume that it is a DHTID-like sequence
            group_id = DHTID.generate().to_bytes()
        self.outcome.was_leader = True
        return LeaderWaitingForFollowers(
            leader_endpoint=self.my_endpoint, leader_expiration=self.my_expiration, group_endpoints={self.my_endpoint},
            group_id=group_id, outcome=self.outcome, max_size=max_size)

    @ProtocolState.transition
    def cancel_looking_for_group(self, message: Any) -> Idle:
        self.outcome.canceled_looking_for_group = True
        self.outcome.message = message
        self.outcome.finished.set()
        return Idle()


@dataclass
class FollowerWaitingForLeader(ProtocolState):
    """ i am in a group, but not a leader; waiting for leader to begin all-reduce """
    my_endpoint: Endpoint
    my_expiration: DHTExpiration
    leader_endpoint: Endpoint
    group_id: bytes
    outcome: AveragingOutcome

    @ProtocolState.transition
    def begin_allreduce(self, group_endpoints: Tuple[Endpoint, ...]) -> RunningAllReduce:
        assert isinstance(group_endpoints, tuple) and self.my_endpoint in group_endpoints and len(group_endpoints) > 1
        self.outcome.started_allreduce = True
        return RunningAllReduce(my_endpoint=self.my_endpoint, group_endpoints=group_endpoints, outcome=self.outcome)

    @ProtocolState.transition
    def cancel_waiting_for_group(self, message: Any) -> Idle:
        self.outcome.left_group = True
        self.outcome.message = message
        self.outcome.finished.set()
        return Idle()


@dataclass
class LeaderWaitingForFollowers(ProtocolState):
    """ i am a leader of a group, waiting for more peers (or timeout) """
    leader_endpoint: Endpoint
    leader_expiration: DHTExpiration
    group_endpoints: Set[Endpoint]
    group_id: bytes
    outcome: AveragingOutcome
    max_size: Optional[int]
    group_assembled: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    ordered_group_endpoints: Optional[Tuple[Endpoint, ...]] = field(default=None, init=False)

    def add_follower(self, follower: Endpoint):
        assert follower not in self.group_endpoints
        assert self.group_size < (self.max_size or float('inf'))
        self.group_endpoints.add(follower)
        if len(self.group_endpoints) >= self.max_size:
            self.group_assembled.set()

    @property
    def group_size(self) -> int:
        return len(self.group_endpoints)

    @ProtocolState.transition
    def begin_allreduce(self) -> RunningAllReduce:
        assert len(self.group_endpoints) > 1 and self.leader_expiration in self.group_endpoints, self.group_endpoints
        self.group_assembled.set()
        self.outcome.started_allreduce = True
        if self.ordered_group_endpoints is None:
            ordered_group_endpoints = list(self.group_endpoints)
            random.shuffle(ordered_group_endpoints)
            self.ordered_group_endpoints = tuple(ordered_group_endpoints)
        return RunningAllReduce(self.leader_endpoint, tuple(self.ordered_group_endpoints), outcome=self.outcome)


@dataclass
class RunningAllReduce(ProtocolState):
    my_endpoint: Endpoint
    group_endpoints: Tuple[Endpoint, ...]
    outcome: AveragingOutcome

    accumulator: Optional[torch.Tensor] = field(default=None, init=False)  # the sum of incoming vector parts
    average_tensor: Optional[torch.Tensor] = field(default=None, init=False)  # accumulator / group size
    received_from: Set[Endpoint] = field(default_factory=set, init=False)  # peers that have sent me their chunk
    finished_accumulating: asyncio.Event = field(default_factory=asyncio.Event, init=False)

    async def accumulate(self, source: Endpoint, part: torch.Tensor) -> torch.Tensor:
        """ Add your vector to accumulator, wait for all other vectors to be added, return the average """
        assert not self.finished_accumulating.is_set(), "averaging is already finished"
        assert source in self.group_endpoints and source not in self.received_from, "unexpected source endpoint"
        if self.accumulator is None:
            self.accumulator = part.clone()
        else:
            assert part.shape == self.accumulator.shape
            self.accumulator.add_(part)

        self.received_from.add(source)
        if len(self.received_from) == len(self.group_endpoints):
            self.average_tensor = self.accumulator.div_(len(self.received_from))
            self.finished_accumulating.set()
        else:
            await self.finished_accumulating.wait()  # wait for other peers to send their part

        assert self.average_tensor is not None
        return self.average_tensor


def split_into_chunks(tensors: Sequence[torch.Tensor], group_size: int) -> Tuple[torch.Tensor]:
    """ combines averaged_tensors into one tensor and splits them into equal chunks of size group_size """
    flat_tensor = torch.cat(tuple(map(torch.Tensor.flatten, tensors)))
    chunk_slices = torch.linspace(start=0, end=len(flat_tensor), steps=group_size + 1, dtype=torch.int64).numpy()
    chunk_slices[-1] = len(flat_tensor)
    return tuple(torch.as_tensor(flat_tensor[chunk_slices[i]: chunk_slices[i + 1]]) for i in range(group_size))


def restore_from_chunks(chunks: Sequence[torch.Tensor], shapes: Sequence[torch.Size]) -> Tuple[torch.Tensor, ...]:
    """ restores the original tensor shapes from chunks obtained by split_into_chunks """
    flat_tensor = torch.cat(list(chunks))
    result_sizes = tuple(map(torch.Size.numel, shapes))
    flat_original_tensors = torch.split_with_sizes(flat_tensor, result_sizes)
    return tuple(map(torch.Tensor.reshape, flat_original_tensors, shapes))
