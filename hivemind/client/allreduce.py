""" This file contains a state machine that defines allreduce protocol used in DecentralizedAverager """
from __future__ import annotations
import asyncio
import random
from typing import Set, Optional, Sequence, Tuple, Dict
from enum import Enum, auto

import torch

from hivemind.utils import Endpoint, get_logger
from hivemind.dht import DHTID, DHTExpiration
from hivemind.proto import averaging_pb2

logger = get_logger(__name__)

# flavour types
GroupID = bytes


class ProtocolState(Enum):
    LOOKING_FOR_GROUP = auto()   # i want to run averaging, but haven't found any peers yet
    LEADER_WAITING_FOR_PEERS = auto()     # i am a leader, waiting for more peers to join
    FOLLOWER_WAITING_FOR_LEADER = auto()  # i am a follower, my leader is assembling the group
    RUNNING_ALLREDUCE = auto()   # we are currently exchanging tensors in a group
    FINISHED_NORMALLY = auto()   # we ran allreduce and finished without errors
    GROUP_DISBANDED = auto()     # leader disbanded the group before we began allreduce
    PROTOCOL_VIOLATION = auto()  # someone else messed up and we can't recover
    INTERNAL_ERROR = auto()      # i messed up and can't recover


class GroupAllReduce:
    """
    An internal class that keeps track of one group allreduce for DecentralizedAverager.
    GroupAllReduce is meant to be modified with methods, no direct variable assignments is allowed outside of debugging.

    :param my_endpoint: my endpoint, as seen by the group leader
    :param expiration: the time after which the group should begin allreduce or be disbanded
    :param my_tensors: a sequence of torch tensors that i intend to average with peers
    """

    def __init__(self, *, my_endpoint: Endpoint, expiration: DHTExpiration, my_tensors: Sequence[torch.Tensor]):
        logger.debug(f"{my_endpoint} - looking for group")
        self.my_endpoint, self.expiration, self.my_tensors = my_endpoint, expiration, my_tensors
        self.state = ProtocolState.LOOKING_FOR_GROUP
        self.error_message: Optional[str] = None
        self.finished = asyncio.Event()  # this event is set if either we finished, failed or disbanded the group

        self.leader_endpoint: Optional[Endpoint] = None
        self.group_id: Optional[GroupID] = None  # a unique identifier of this one group all-reduce

        # populated when assembling a group
        self.pending_group_endpoints: Optional[Set[Endpoint]] = None  # used only if you are a group leader
        self.ordered_group_endpoints: Optional[Sequence[Endpoint]] = None  # provided by leader when it starts allreduce
        self.group_assembled = asyncio.Event()  # set when the leader is done waiting for peers and starts allreduce

        # populated when running allreduce
        self.accumulator: Optional[torch.Tensor] = None   # the sum of averaged tensors so far, init with zeros
        self.accumulated_from: Set[Endpoint] = set()      # peers that we have accumulated our part from
        self.average_tensor_part: Optional[torch.Tensor] = None   # accumulator / num_parts, after done_accumulating
        self.done_accumulating = asyncio.Event()  # set if we averaged our chunk of data and can send it to peers

    @property
    def success(self):
        return self.finished.is_set() and self.error_message is None

    def start_new_group(self) -> GroupAllReduce:
        """ Create new group with a random id, become its leader and the only participant """
        assert self.state == ProtocolState.LOOKING_FOR_GROUP
        self.group_id = DHTID.generate().to_bytes()
        # note: we generate group_id as DHTID for convenience. Do not assume that it has DHTID-like properties
        logger.debug(f"{self.my_endpoint} - starting a new group as a leader. Group id: {self.group_id}")
        self.state = ProtocolState.LEADER_WAITING_FOR_PEERS
        self.leader_endpoint = self.my_endpoint
        self.pending_group_endpoints = set()
        return self

    @property
    def pending_group_size(self):
        assert self.state == ProtocolState.LEADER_WAITING_FOR_PEERS
        return len(self.pending_group_endpoints)

    def join_group(self, group_id: GroupID, leader_endpoint: Endpoint, my_visible_endpoint: Endpoint) -> GroupAllReduce:
        """ After you were accepted by a leader, create your local instance using the metadata he sent """
        logger.debug(f"{self.my_endpoint} - joining the group of {leader_endpoint}. Group id: {self.group_id}")
        self.my_endpoint, self.group_id, self.leader_endpoint = my_visible_endpoint, group_id, leader_endpoint
        return self

    def add_peer_to_group(self, follower: Endpoint):
        assert self.state == ProtocolState.LEADER_WAITING_FOR_PEERS
        assert follower not in self.pending_group_endpoints
        logger.debug(f"{self.my_endpoint} - adding {follower} to my group. New size = {self.pending_group_size}")
        self.pending_group_endpoints.add(follower)

    def remove_peer_from_group(self, follower: Endpoint):
        assert self.state == ProtocolState.LEADER_WAITING_FOR_PEERS
        assert follower != self.leader_endpoint
        logger.info(f"{self.my_endpoint} - removed {follower} from the group. New size = f{self.pending_group_size}")
        self.pending_group_endpoints.remove(follower)

    def check_reasons_to_reject(self, request: averaging_pb2.MessageToLeader
                                ) -> Optional[averaging_pb2.MessageFromLeader]:
        """ :return: None if peer can be added to the group, otherwise return the error message """
        if self.state == ProtocolState.LOOKING_FOR_GROUP:
            if self.expiration > (request.expiration or float('inf')):
                return averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_EXPIRATION_TIME)
            elif request.my_endpoint == self.my_endpoint or not isinstance(request.my_endpoint, Endpoint):
                return averaging_pb2.MessageFromLeader(code=averaging_pb2.DUPLICATE_ENDPOINT)
            else:
                return None  # we should create a new group and accept the peer

        elif self.state == ProtocolState.LEADER_WAITING_FOR_PEERS:
            if self.expiration > (request.expiration or float('inf')):
                return averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_EXPIRATION_TIME)
            elif request.my_endpoint in self.pending_group_endpoints or not isinstance(request.my_endpoint, Endpoint):
                return averaging_pb2.MessageFromLeader(code=averaging_pb2.DUPLICATE_ENDPOINT)
            elif self.group_assembled.is_set():
                return averaging_pb2.MessageFromLeader(code=averaging_pb2.ALREADY_RUNNING)
            else:
                return None  # we can accept peer in our current group

        elif self.state == ProtocolState.FOLLOWER_WAITING_FOR_LEADER:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_A_LEADER,
                                                   suggested_leader=self.leader_endpoint)
        elif self.state == ProtocolState.RUNNING_ALLREDUCE:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.ALREADY_RUNNING)
        else:
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_LOOKING_FOR_GROUP)

    def leader_begin_allreduce(self) -> GroupAllReduce:
        assert self.state == ProtocolState.LEADER_WAITING_FOR_PEERS and self.pending_group_size > 1
        logger.debug(f"{self.my_endpoint} - initiating allreduce for {self.pending_group_size} peers.")
        ordered_group_endpoints = list(self.pending_group_endpoints)
        random.shuffle(ordered_group_endpoints)
        self.ordered_group_endpoints = tuple(ordered_group_endpoints)
        self.state = ProtocolState.RUNNING_ALLREDUCE
        self.group_assembled.set()
        return self

    def follower_begin_allreduce(self, ordered_group_endpoints: Sequence[Endpoint]) -> GroupAllReduce:
        assert self.state == ProtocolState.FOLLOWER_WAITING_FOR_LEADER and self.my_endpoint in ordered_group_endpoints
        logger.debug(f"{self.my_endpoint} - received peer order from the leader, beginning allreduce.")
        self.ordered_group_endpoints = ordered_group_endpoints
        self.state = ProtocolState.RUNNING_ALLREDUCE
        self.group_assembled.set()
        #TODO IF we already accumulated some chunks, check that they came from valid peers

        return self

    def dispatch_chunks_to_peers(self) -> Dict[Endpoint, averaging_pb2.AveragingData]:
        """ For each peer in you group, generate averaging chunks """
        assert self.state == ProtocolState.RUNNING_ALLREDUCE
        # TODO accumulate your OWN chunk to the accumulator right now
        raise NotImplementedError()

    async def accumulate(self, source: Endpoint, part: torch.Tensor) -> torch.Tensor:
        """ Add your vector to accumulator, wait for all other vectors to be added, return the average """
        assert not self.done_accumulating.is_set(), "averaging is already finished"
        assert source not in self.accumulated_from, "duplicate source endpoint, already received that part"
        if self.accumulator is None:
            self.accumulator = part.clone()
        else:
            assert part.shape == self.accumulator.shape
            self.accumulator.add_(part)

        self.accumulated_from.add(source)

        if len(self.accumulated_from) == len(self.ordered_group_endpoints):
            self.average_tensor = self.accumulator.div_(len(self.accumulated_from))
            self.done_accumulating.set()
        else:
            await self.done_accumulating.wait()  # wait for other peers to send their part

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
