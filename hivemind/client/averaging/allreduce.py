import asyncio
from typing import Sequence, Set, Dict, Tuple

import grpc
import torch

from hivemind.utils import Endpoint, get_logger, serialize_torch_tensor, deserialize_torch_tensor, ChannelCache
from hivemind.proto import averaging_pb2_grpc, runtime_pb2, averaging_pb2

# flavour types
GroupID = bytes
logger = get_logger(__name__)


class AllReduceProtocol:
    """
    An internal class that runs butterfly AllReduce in a predefined group of averagers

    :param tensors: local tensors that should be averaged with groupmates
    :param endpoint: your endpoint, must be included in ordered_group_endpoints
    :param ordered_group_endpoints: group endpoints ordered s.t. i-th endpoint is responsible for averaging i-th part
    """
    def __init__(self, *, group_id: GroupID, tensors: Sequence[torch.Tensor], endpoint: Endpoint,
                 ordered_group_endpoints: Sequence[Endpoint]):
        assert endpoint in ordered_group_endpoints, "endpoint is not a part of the group"
        self.group_id, self.endpoint, self.ordered_group_endpoints = group_id, endpoint, ordered_group_endpoints
        self.local_tensor_parts = dict(zip(ordered_group_endpoints, split_into_parts(tensors, self.group_size)))
        self.tensor_shapes = tuple(tensor.shape for tensor in tensors)

        self.accumulator = self.local_tensor_parts[self.endpoint].clone()  # sum inputs from peers to this tensor
        self.accumulated_from: Set[Endpoint] = {self.endpoint}  # peers that we have accumulated our part from
        self.averaged_part: asyncio.Future[torch.Tensor] = asyncio.Future()  # will be set to [accumulator / group size]
        self.averaged_tensor_parts: Dict[Endpoint, torch.Tensor] = {}  # averaged chunks from all peers will be put here
        self.averaged_tensors: asyncio.Future[Sequence[torch.Tensor]] = asyncio.Future()  # final result or exception

    def __repr__(self):
        return f"{self.__class__.__name__}({self.endpoint}, group_size={self.group_size})"

    def __await__(self):
        return self.averaged_tensors.__await__()

    @property
    def group_size(self):
        return len(self.ordered_group_endpoints)

    async def accumulate_part(self, source: Endpoint, remote_part: torch.Tensor) -> torch.Tensor:
        """ Add vector part to accumulator, wait for all other vectors to be added, then return the average part """
        assert not self.averaged_part.done(), f"already finished averaging part: {self.averaged_part}"
        assert not self.averaged_tensors.done(), f"already finished allreduce: {self.averaged_tensors}"
        assert source in self.local_tensor_parts, "unexpected source, not a part of current group"
        assert source not in self.accumulated_from, "duplicate source, already received that part"
        logger.debug(f"{self} - accumulating tensor part from {source}")

        self.accumulator.add_(remote_part)
        self.accumulated_from.add(source)

        assert len(self.accumulated_from) <= self.group_size
        if len(self.accumulated_from) == len(self.local_tensor_parts):
            average_result = self.accumulator.div_(len(self.accumulated_from))
            self.register_averaged_part(self.endpoint, average_result)
            self.averaged_part.set_result(average_result)

        return await self.averaged_part

    def register_averaged_part(self, source: Endpoint, averaged_part: torch.Tensor):
        assert not self.averaged_tensors.done(), f"already finished allreduce: {self.averaged_tensors}"
        assert source in self.local_tensor_parts, "the provider of averaged part is not from my group"
        assert source not in self.averaged_tensor_parts, "already registered the average from this peer"
        assert averaged_part.shape == self.local_tensor_parts[source].shape, "averaged part shape mismatch"
        assert averaged_part.dtype == self.local_tensor_parts[source].dtype, "averaged part dtype mismatch"
        logger.debug(f"{self} - receiving averaged tensor part from {source}")
        self.averaged_tensor_parts[source] = averaged_part
        if len(self.averaged_tensor_parts) == len(self.local_tensor_parts):
            ordered_averaged_parts = [self.averaged_tensor_parts[endpoint] for endpoint in self.ordered_group_endpoints]
            self.averaged_tensors.set_result(restore_from_parts(ordered_averaged_parts, self.tensor_shapes))

    def cancel(self) -> bool:
        if not self.averaged_tensors.done():
            logger.debug(f"{self} - cancelled")
            self.averaged_tensors.cancel()
            if not self.averaged_part.done():
                self.averaged_part.cancel()
            return True
        else:
            logger.debug(f"{self} - failed to cancel, allreduce is already finished: {self.averaged_tensors}")
            return False

    def set_exception(self, exception: Exception) -> bool:
        if not self.averaged_tensors.done():
            logger.debug(f"{self} - {exception}")
            self.averaged_tensors.set_exception(exception)
            if not self.averaged_part.done():
                self.averaged_part.cancel()
            return True
        else:
            logger.debug(f"{self} - failed to set {exception}, allreduce already finished: {self.averaged_tensors}")
            return False


class AllReduceRunner(AllReduceProtocol, averaging_pb2_grpc.DecentralizedAveragingServicer):
    """
    A class that implements ButterflyAllReduceProtocol on top of a gRPC servicer
    """
    def __init__(self, *, group_id: GroupID, tensors: Sequence[torch.Tensor], endpoint: Endpoint,
                 ordered_group_endpoints: Sequence[Endpoint], compression_type: runtime_pb2.CompressionType):
        super().__init__(group_id=group_id, tensors=tensors, endpoint=endpoint,
                         ordered_group_endpoints=ordered_group_endpoints)
        self.compression_type = compression_type

    def _get_peer_stub(self, peer: Endpoint) -> averaging_pb2_grpc.DecentralizedAveragingStub:
        return ChannelCache.get_stub(peer, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)

    async def _average_one_part(self, peer_endpoint: Endpoint, local_part: torch.Tensor) -> torch.Tensor:
        """ Send one part of local tensors to one groupmate and collect the average for this part """
        serialized_tensor_part = serialize_torch_tensor(local_part, self.compression_type, allow_inplace=False)
        response = await self._get_peer_stub(peer_endpoint).rpc_aggregate_part(
            averaging_pb2.AveragingData(code=averaging_pb2.PART_FOR_AVERAGING, group_id=self.group_id,
                                        endpoint=self.endpoint, tensor_part=serialized_tensor_part))
        if response.code == averaging_pb2.AVERAGED_PART:
            averaged_part = deserialize_torch_tensor(response.tensor_part)
            self.register_averaged_part(peer_endpoint, averaged_part)
            return averaged_part
        else:
            raise AllreduceException(f"peer {peer_endpoint} returned {averaging_pb2.MessageCode.Name(response.code)}"
                                     f" instead of {averaging_pb2.MessageCode.Name(averaging_pb2.AVERAGED_PART)},"
                                     f" allreduce failed")

    async def _send_error_to_peer(self, peer_endpoint: Endpoint, code: averaging_pb2.MessageCode):
        await self._get_peer_stub(peer_endpoint).rpc_aggregate_part(averaging_pb2.AveragingData(
            group_id=self.group_id, endpoint=self.endpoint, code=code))

    async def run(self) -> Sequence[torch.Tensor]:
        """ send allreduce requests to all peers and collect results, return the averaged tensor """
        try:
            await asyncio.gather(self, *(self._average_one_part(peer, part)
                                         for peer, part in self.local_tensor_parts.items() if peer != self.endpoint))
            return await self
        except Exception as e:
            code = averaging_pb2.CANCELLED if isinstance(e, asyncio.CancelledError) else averaging_pb2.INTERNAL_ERROR
            logger.debug(f"{self} - notifying peers about {averaging_pb2.MessageCode.Name(code)}")
            self.set_exception(e)
            for peer_endpoint in self.ordered_group_endpoints:
                asyncio.create_task(self._send_error_to_peer(peer_endpoint, code))
            raise

    async def rpc_aggregate_part(self, request: averaging_pb2.AveragingData, context: grpc.ServicerContext):
        """ a groupmate sends us a part of his tensor; we should average it with other peers and return the result """
        if request.group_id != self.group_id:
            return averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)

        if request.code == averaging_pb2.PART_FOR_AVERAGING:
            try:
                tensor_part = deserialize_torch_tensor(request.tensor_part)
                averaged_part = await self.accumulate_part(request.endpoint, tensor_part)
                serialized = serialize_torch_tensor(averaged_part, request.tensor_part.compression, allow_inplace=False)
                return averaging_pb2.AveragingData(code=averaging_pb2.AVERAGED_PART, tensor_part=serialized)
            except Exception as e:
                self.set_exception(e)
                return averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
        else:
            error_code = averaging_pb2.MessageCode.Name(request.code)
            logger.debug(f"{self} - peer {request.endpoint} sent {error_code}, allreduce cannot continue")
            self.set_exception(AllreduceException(f"peer {request.endpoint} sent {error_code}."))
            return averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)


def split_into_parts(tensors: Sequence[torch.Tensor], group_size: int) -> Tuple[torch.Tensor, ...]:
    """ combines averaged_tensors into one tensor and splits them into equal chunks of size group_size """
    flat_tensor = torch.cat(tuple(map(torch.Tensor.flatten, tensors)))
    chunk_slices = torch.linspace(start=0, end=len(flat_tensor), steps=group_size + 1, dtype=torch.int64)
    chunk_slices[-1] = len(flat_tensor)
    return tuple(flat_tensor[chunk_slices[i]: chunk_slices[i + 1]] for i in range(group_size))


def restore_from_parts(chunks: Sequence[torch.Tensor], shapes: Sequence[torch.Size]) -> Tuple[torch.Tensor, ...]:
    """ restores the original tensor shapes from chunks obtained by split_into_chunks """
    flat_tensor = torch.cat(tuple(chunks))
    result_sizes = tuple(map(torch.Size.numel, shapes))
    flat_original_tensors = torch.split_with_sizes(flat_tensor, result_sizes)
    return tuple(map(torch.Tensor.reshape, flat_original_tensors, shapes))


class AllreduceException(Exception):
    """ A special exception that is raised when allreduce can't continue normally (e.g. disbanded/bad request/etc) """
