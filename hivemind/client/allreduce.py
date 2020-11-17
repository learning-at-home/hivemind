""" This file contains a state machine that defines allreduce protocol used in DecentralizedAverager """
from __future__ import annotations
import asyncio
import random
from dataclasses import asdict
from typing import Set, Optional, Sequence, Tuple, Dict, AsyncIterator
from enum import Enum, auto

import grpc
import torch

from hivemind.dht import DHTID, DHTExpiration
from hivemind.utils import Endpoint, get_logger, MSGPackSerializer
from hivemind.utils import TensorDescriptor, deserialize_torch_tensor, serialize_torch_tensor
from hivemind.proto import averaging_pb2, averaging_pb2_grpc, runtime_pb2

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
    ERROR = auto()               # someone (maybe i) messed up and we can't recover
    CANCELLED = auto()           # i have unilaterally cancelled GroupAllreduce


class GroupAllReduce:
    """
    An internal class that keeps track of one group allreduce for DecentralizedAverager.
    GroupAllReduce is meant to be modified with methods, no direct variable assignments is allowed outside of debugging.

    :param endpoint: my endpoint, as seen by the group leader
    :param expiration: the time after which the group should begin allreduce or be disbanded
    :param tensors: a sequence of torch tensors that i intend to average with peers
    """
    compression_type = runtime_pb2.NONE

    def __init__(self, endpoint: Endpoint, expiration: DHTExpiration, tensors: Sequence[torch.Tensor]):
        assert all(tensor.dtype == torch.float32 and tensor.device == torch.device('cpu') for tensor in tensors)
        self.local_tensors = tensors
        self.state = ProtocolState.LOOKING_FOR_GROUP
        self.info = averaging_pb2.PeerInfo(endpoint=endpoint, expiration=expiration,
                                           schema_hash=compute_schema_hash(tensors))

        self.leader_endpoint: Optional[Endpoint] = None
        self.group_id: Optional[GroupID] = None  # a unique identifier of this one group all-reduce
        self.max_size = float('inf')  # maximum group size, only enforced for group leader

        # populated when assembling a group
        self.group_endpoints_set: Optional[Set[Endpoint]] = None
        self.assembled_group: asyncio.Future[Sequence[Endpoint]] = asyncio.Future()  # final ordered endpoints
        self.concurrent_requests_lock = asyncio.Lock()  # lock inbound/outbound requests to join group

        # populated when running allreduce
        self.accumulator: Optional[torch.Tensor] = None   # the sum of averaged tensors so far, init with zeros
        self.accumulated_from: Set[Endpoint] = set()      # peers that we have accumulated our part from
        self.averaged_part: asyncio.Future[torch.Tensor] = asyncio.Future()

        self.average_tensor_parts: Dict[Endpoint, torch.Tensor] = {}  # averaged chunks from all peers
        self.averaged_tensors: asyncio.Future[Sequence[torch.Tensor]] = asyncio.Future()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.info.endpoint}, {self.state})"

    def __await__(self):
        return self.averaged_tensors.__await__()

    def start_new_group(self, max_size: Optional[int] = None):
        """ Create new group with a random id, become its leader and the only participant """
        assert self.state == ProtocolState.LOOKING_FOR_GROUP
        self.group_id = DHTID.generate().to_bytes()
        # note: we generate group_id as DHTID for convenience. Do not assume that it has DHTID-like properties
        logger.debug(f"{self} - starting a new group as a leader. Group id: {self.group_id}")
        self.state = ProtocolState.LEADER_WAITING_FOR_PEERS
        self.leader_endpoint = self.info.endpoint
        self.group_endpoints_set = {self.info.endpoint}
        if max_size is not None:
            self.max_size = max_size

    @property
    def group_size(self):
        assert self.state in (ProtocolState.LEADER_WAITING_FOR_PEERS, ProtocolState.RUNNING_ALLREDUCE)
        return len(self.group_endpoints_set)

    def join_group(self, leader_endpoint: Endpoint, group_id: GroupID):
        """ After you were accepted by a leader, create your local instance using the metadata he sent """
        self.group_id, self.leader_endpoint = group_id, leader_endpoint
        logger.debug(f"{self} - joining the group of {leader_endpoint}. Group id: {self.group_id}")
        self.state = ProtocolState.FOLLOWER_WAITING_FOR_LEADER

    def add_peer_to_group(self, follower: Endpoint):
        """ Add peer to a group, assuming that he can be added (self.get_reasons_to_reject(peer) is None) """
        assert self.state == ProtocolState.LEADER_WAITING_FOR_PEERS
        assert follower not in self.group_endpoints_set
        self.group_endpoints_set.add(follower)
        logger.debug(f"{self} - adding {follower} to my group. New size = {self.group_size}")
        if self.group_size > self.max_size:
            logger.warning(f"{self} - group size ({self.group_size}) exceeded max size ({self.max_size})")

    def remove_peer_from_group(self, follower: Endpoint):
        """ Remove a disconnected peer from current group """
        assert self.state == ProtocolState.LEADER_WAITING_FOR_PEERS
        assert follower in self.group_endpoints_set and follower != self.leader_endpoint
        self.group_endpoints_set.remove(follower)
        logger.info(f"{self} - removed {follower} from the group. New size = {self.group_size}")

    def disband_group(self):
        assert self.state == ProtocolState.LEADER_WAITING_FOR_PEERS and self.group_size == 1
        logger.info(f"{self} - disbanded group (reason = empty)")
        self.state = ProtocolState.LOOKING_FOR_GROUP

    def leader_begin_allreduce(self) -> averaging_pb2.MessageFromLeader:
        """ As a leader, distribute allreduce metadata to peers and start allreduce """
        assert self.state == ProtocolState.LEADER_WAITING_FOR_PEERS and self.group_size > 1
        logger.debug(f"{self} - initiating allreduce for {self.group_endpoints_set} peers.")
        ordered_group_endpoints = list(self.group_endpoints_set)
        random.shuffle(ordered_group_endpoints)
        self.assembled_group.set_result(ordered_group_endpoints)
        self.state = ProtocolState.RUNNING_ALLREDUCE

    def follower_begin_allreduce(self, ordered_group_endpoints: Sequence[Endpoint]):
        """ As a follower, receive the final list of peers from the leader and begin sending data around """
        assert self.state == ProtocolState.FOLLOWER_WAITING_FOR_LEADER and self.info.endpoint in ordered_group_endpoints
        logger.debug(f"{self} - received peer order from the leader, beginning allreduce.")
        self.group_endpoints_set = set(ordered_group_endpoints)
        self.assembled_group.set_result(ordered_group_endpoints)
        self.state = ProtocolState.RUNNING_ALLREDUCE

    async def accumulate(self, source: Endpoint, part: torch.Tensor) -> torch.Tensor:
        """ Add vector part to accumulator, wait for all other vectors to be added, return the average """
        assert source not in self.accumulated_from, "duplicate endpoint, already received that part"
        assert self.accumulator is None or self.accumulator.shape == part.shape
        logger.debug(f"{self} - accumulated part from {source}")

        self.accumulator = part if self.accumulator is None else self.accumulator.add_(part)
        self.accumulated_from.add(source)

        ordered_group_endpoints = await self.assembled_group
        assert len(self.accumulated_from) <= len(ordered_group_endpoints)
        if len(self.accumulated_from) == len(ordered_group_endpoints):
            self.averaged_part.set_result(self.accumulator.div_(len(self.accumulated_from)))

        return await self.averaged_part

    def _get(self, peer: Endpoint) -> averaging_pb2_grpc.DecentralizedAveragingStub:
        """ TODO this function is deprecated and will be replaced by a shared channel cache """
        channel = grpc.aio.insecure_channel(peer)
        return averaging_pb2_grpc.DecentralizedAveragingStub(channel)

    async def handle_join_request(self, request: averaging_pb2.PeerInfo
                                  ) -> AsyncIterator[averaging_pb2.MessageFromLeader]:
        """ accept or reject a join request; if accepted, run him through allreduce steps """
        should_remove_peer = False
        try:
            # stage 1: check if there is a reason to reject a peer outright
            if not is_valid_join_request(request):
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.PROTOCOL_VIOLATION)
                return
            if self.info.expiration > (request.expiration or float('inf')):
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_EXPIRATION_TIME)
            elif request.schema_hash != self.info.schema_hash:
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_SCHEMA_HASH)
                return
            elif request.endpoint == self.info.endpoint or request.endpoint in (self.group_endpoints_set or ()):
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.DUPLICATE_ENDPOINT)
                return
            elif self.state == ProtocolState.FOLLOWER_WAITING_FOR_LEADER:
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_A_LEADER,
                                                      suggested_leader=self.leader_endpoint)
                return
            elif self.state == ProtocolState.RUNNING_ALLREDUCE or len(self.accumulated_from) > 0:
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.ALREADY_RUNNING)
                return
            if self.state == ProtocolState.LEADER_WAITING_FOR_PEERS and self.group_size >= self.max_size:
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_IS_FULL)
                return

            # stage 2: add peer to group, optionally start a new one
            async with self.concurrent_requests_lock:
                if self.state == ProtocolState.LOOKING_FOR_GROUP:
                    self.start_new_group()

                assert self.state == ProtocolState.LEADER_WAITING_FOR_PEERS

                self.add_peer_to_group(request.endpoint)
                should_remove_peer = True
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.ACCEPTED, group_id=self.group_id)

            if self.group_size >= self.max_size:
                self.leader_begin_allreduce()

            # stage 3: wait for the group to be assembled and return
            ordered_group_endpoints = await self.assembled_group
            if ordered_group_endpoints is not None:
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.BEGIN_ALLREDUCE,
                                                      ordered_group_endpoints=ordered_group_endpoints)
            else:
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_DISBANDED)

        except Exception as e:
            logger.exception(e)
            yield averaging_pb2.MessageFromLeader(code=averaging_pb2.INTERNAL_ERROR)

        finally:  # this code is guaranteed to run if the iterator is destroyed prematurely
            if should_remove_peer:
                self.remove_peer_from_group(request.endpoint)
                if self.group_size <= 1:
                    self.set_exception(ValueError("All peers have left"))

    async def request_join_group(self, leader: Endpoint
                                 ) -> Optional[grpc.aio.UnaryStreamCall[averaging_pb2.MessageFromLeader]]:
        """ request a given peer to be your leader for allreduce. if accepted, return a grpc stream """
        assert self.state == ProtocolState.LOOKING_FOR_GROUP
        try:
            async with self.concurrent_requests_lock:
                stream = self._get(leader).rpc_group_allreduce(self.info)
                message = await stream.read()
                logger.debug(f"{self} - requested {leader} to be my leader, received "
                             f"{averaging_pb2.MessageCode.Name(message.code)}")
                if message.code == averaging_pb2.ACCEPTED:
                    self.join_group(leader, message.group_id)
                    return stream

        except Exception as e:
            self.set_exception(e)

    async def wait_for_allreduce(self, stream: grpc.aio.UnaryStreamCall[averaging_pb2.MessageFromLeader]) -> bool:
        """ the second part of request_join_group, return True if started allreduce, False if failed or disbanded """
        try:
            message = await stream.read()
            if message.code == averaging_pb2.BEGIN_ALLREDUCE:
                logger.debug(f"{self} - leader triggered allreduce")
                assert all(isinstance(p, Endpoint) for p in message.ordered_group_endpoints)
                self.follower_begin_allreduce(message.ordered_group_endpoints)
                return True
            else:
                logger.debug(f"{self} - leader sent {averaging_pb2.MessageCode.Name(message.code)}, leaving group")
                self.state = ProtocolState.GROUP_DISBANDED
                return False
        except Exception as e:
            self.set_exception(e)
            return False

    async def run_allreduce(self) -> Sequence[torch.Tensor]:
        """ send allreduce requests to all peers and collect results, return the averaged tensor """
        assert self.state == ProtocolState.RUNNING_ALLREDUCE
        ordered_group_endpoints = await self.assembled_group
        ordered_local_parts = split_into_parts(self.local_tensors, group_size=self.group_size)

        async def send_part(peer_endpoint: Endpoint, local_part: torch.Tensor):
            if peer_endpoint == self.info.endpoint:
                self.average_tensor_parts[peer_endpoint] = await self.accumulate(peer_endpoint, local_part)
            else:
                serialized_tensor_part = serialize_torch_tensor(local_part, self.compression_type, allow_inplace=False)
                response = await self._get(peer_endpoint).rpc_aggregate_part(averaging_pb2.AveragingData(
                    group_id=self.group_id, endpoint=self.info.endpoint, tensor_part=serialized_tensor_part))

                if response.code == averaging_pb2.ACCEPTED:
                    self.average_tensor_parts[peer_endpoint] = deserialize_torch_tensor(response.tensor_part)
                else:
                    raise ValueError(f"peer {peer_endpoint} replied {averaging_pb2.MessageCode.Name(response.code)}")

            if len(self.average_tensor_parts) >= len(self.group_endpoints_set):
                ordered_parts = [self.average_tensor_parts[peer] for peer in ordered_group_endpoints]
                tensor_shapes = [tensor.shape for tensor in self.local_tensors]
                self.averaged_tensors.set_result(restore_from_parts(ordered_parts, tensor_shapes))

        try:
            await asyncio.gather(*map(send_part, ordered_group_endpoints, ordered_local_parts))
            return await self.averaged_tensors
        except Exception as e:
            code = averaging_pb2.CANCELLED if isinstance(e, asyncio.CancelledError) else averaging_pb2.INTERNAL_ERROR

            async def send_error_to_peer(peer_endpoint):
                await self._get(peer_endpoint).rpc_aggregate_part(averaging_pb2.AveragingData(
                    group_id=self.group_id, endpoint=self.info.endpoint, code=code))
            for peer_endpoint in ordered_group_endpoints:
                asyncio.create_task(send_error_to_peer(peer_endpoint))
            if code == averaging_pb2.CANCELLED:
                self.cancel()
            else:
                self.set_exception(e)
            raise

    async def handle_accumulate_request(self, request: averaging_pb2.AveragingData) -> averaging_pb2.AveragingData:
        """ respond to an incoming rpc_accumulate_part """
        if self.state not in (ProtocolState.RUNNING_ALLREDUCE, ProtocolState.FOLLOWER_WAITING_FOR_LEADER):
            return averaging_pb2.AveragingData(code=averaging_pb2.PROTOCOL_VIOLATION)
        elif request.group_id != self.group_id:
            return averaging_pb2.AveragingData(code=averaging_pb2.PROTOCOL_VIOLATION)
        elif request.endpoint in self.accumulated_from:
            return averaging_pb2.AveragingData(code=averaging_pb2.DUPLICATE_ENDPOINT)

        if request.code in (averaging_pb2.INTERNAL_ERROR, averaging_pb2.CANCELLED):
            self.set_exception(ValueError(f"{request.endpoint} sent {averaging_pb2.MessageCode.Name(request.code)}"))
            return averaging_pb2.AveragingData(code=averaging_pb2.PROTOCOL_VIOLATION)

        try:
            received_part = deserialize_torch_tensor(request.tensor_part)
            averaged_part = await self.accumulate(request.endpoint, received_part)
            serialized = serialize_torch_tensor(averaged_part, request.tensor_part.compression, allow_inplace=False)
            return averaging_pb2.AveragingData(code=averaging_pb2.ACCEPTED, tensor_part=serialized)
        except asyncio.CancelledError:
            self.cancel()
            return averaging_pb2.AveragingData(code=averaging_pb2.CANCELLED)
        except Exception as e:
            self.set_exception(e)
            return averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)

    def cancel(self):
        logger.debug(f"{self} - cancelled")
        self.state = ProtocolState.CANCELLED
        for future in self.assembled_group, self.averaged_part, self.averaged_tensors:
            future.cancel()

    def set_exception(self, exception: Exception):
        logger.debug(f"{self} - {exception}")
        self.state = ProtocolState.ERROR
        for future in self.assembled_group, self.averaged_part, self.averaged_tensors:
            future.set_exception(exception)


def split_into_parts(tensors: Sequence[torch.Tensor], group_size: int) -> Tuple[torch.Tensor]:
    """ combines averaged_tensors into one tensor and splits them into equal chunks of size group_size """
    flat_tensor = torch.cat(tuple(map(torch.Tensor.flatten, tensors)))
    chunk_slices = torch.linspace(start=0, end=len(flat_tensor), steps=group_size + 1, dtype=torch.int64).numpy()
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
