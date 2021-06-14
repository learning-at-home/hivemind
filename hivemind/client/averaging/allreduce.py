import asyncio
from typing import Sequence, Set, Dict, Tuple, Iterable, AsyncIterator, Any, Optional
from enum import Enum

import grpc
import torch

from hivemind.utils import Endpoint, get_logger, ChannelCache, anext
from hivemind.utils import split_for_streaming, combine_from_streaming
from hivemind.utils.compression import serialize_torch_tensor, deserialize_torch_tensor
from hivemind.proto import averaging_pb2_grpc, runtime_pb2, averaging_pb2

# flavour types
GroupID = bytes
logger = get_logger(__name__)


class AveragingMode(Enum):
    NODE = 0
    CLIENT = 1
    AUX = 2


class AllReduceProtocol:
    """
    An internal class that runs butterfly AllReduce in a predefined group of averagers

    :param tensors: local tensors that should be averaged with groupmates
    :param endpoint: your endpoint, must be included in ordered_group_endpoints
    :param ordered_group_endpoints: group endpoints ordered s.t. i-th endpoint is responsible for averaging i-th part
    :param part_sizes: for each peer, a number of vector elements that this peer is responsible for averaging
    :param return_deltas: if True, returns the element-wise differences (averaged_tensors - original_tensors)
           default (False) - return averaged_tensors by themselves
    """

    def __init__(self, *, group_id: GroupID, tensors: Sequence[torch.Tensor], endpoint: Endpoint,
                 ordered_group_endpoints: Sequence[Endpoint], part_sizes: Tuple[int, ...], return_deltas: bool = False,
                 modes: Optional[Sequence[AveragingMode]] = None):
        assert endpoint in ordered_group_endpoints, "endpoint is not a part of the group"
        self.group_id, self.endpoint = group_id, endpoint
        self.ordered_group_endpoints, self.part_sizes = ordered_group_endpoints, part_sizes
        if modes is None:
            modes = [AveragingMode.CLIENT if part_size == 0 else AveragingMode.NODE for part_size in part_sizes]
        assert any(mode != AveragingMode.CLIENT for mode in modes), "Cannot run allreduce without reducers."
        self.peer_modes = dict(zip(ordered_group_endpoints, modes))

        self.local_tensor_parts = dict(zip(ordered_group_endpoints, split_into_parts(tensors, part_sizes)))
        self.tensor_shapes = tuple(tensor.shape for tensor in tensors)
        self.return_deltas = return_deltas

        self.accumulator = torch.zeros_like(self.local_tensor_parts[self.endpoint])
        self.denominator = 0.0  # number of peers added to accumulator or sum of their weights
        self.accumulated_from: Set[Endpoint] = set()  # peers that we have accumulated our part from
        self.averaged_part: asyncio.Future[torch.Tensor] = asyncio.Future()  # will be set to [accumulator / group size]
        self.averaged_tensor_parts: Dict[Endpoint, torch.Tensor] = {}  # averaged chunks from all peers will be put here
        self.future: asyncio.Future[Sequence[torch.Tensor]] = asyncio.Future()  # final result or exception

        self.num_senders = len([mode for mode in modes if mode != AveragingMode.AUX])

        if self.num_senders == 0:
            self.future.set_result(None)
        for endpoint, mode in self.peer_modes.items():
            if mode == AveragingMode.CLIENT:
                self.averaged_tensor_parts[endpoint] = torch.tensor([])

    def __repr__(self):
        return f"{self.__class__.__name__}({self.endpoint}, group_size={self.group_size})"

    def __await__(self):
        return self.future.__await__()

    def __contains__(self, endpoint: Endpoint):
        return endpoint in self.local_tensor_parts

    @property
    def group_size(self):
        return len(self.ordered_group_endpoints)

    async def accumulate_part(self, source: Endpoint, remote_part: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        """ Add vector part to accumulator, wait for all other vectors to be added, then return the average part """
        assert not self.averaged_part.done(), f"already finished averaging part: {self.averaged_part}"
        assert not self.future.done(), f"already finished allreduce: {self.future}"
        assert source in self.local_tensor_parts, "unexpected source, not a part of current group"
        assert source not in self.accumulated_from, "duplicate source, already received that part"
        assert self.peer_modes[self.endpoint] != AveragingMode.CLIENT, f"{self.endpoint} is in AveragingMode.client mode"
        assert isinstance(weight, (int, float)) and weight > 0, "averaging weights must be a non-negative int/float"

        logger.debug(f"{self} - accumulating tensor part from {source}")
        self.accumulator.add_(remote_part, alpha=weight)
        self.denominator += weight
        self.accumulated_from.add(source)

        assert len(self.accumulated_from) <= self.num_senders
        if len(self.accumulated_from) == self.num_senders:
            average_result = self.accumulator.div_(self.denominator)
            self.averaged_part.set_result(average_result)

            if self.peer_modes[self.endpoint] == AveragingMode.AUX:
                self.future.set_result(None)  # auxiliary mode has finished averaging
            else:
                self.register_averaged_part(self.endpoint, average_result)

        return await self.averaged_part

    def register_averaged_part(self, source: Endpoint, averaged_part: torch.Tensor):
        assert not self.future.done(), f"already finished allreduce: {self.future}"
        assert source in self.local_tensor_parts, "the provider of averaged part is not from my group"
        assert source not in self.averaged_tensor_parts, "already registered the average from this peer"
        assert averaged_part.shape == self.local_tensor_parts[source].shape, "averaged part shape mismatch"
        assert averaged_part.dtype == self.local_tensor_parts[source].dtype, "averaged part dtype mismatch"
        assert self.peer_modes[self.endpoint] != AveragingMode.AUX, "Auxiliary peers do not have local tensors for sending"
        logger.debug(f"{self} - receiving averaged tensor part from {source}")
        self.averaged_tensor_parts[source] = averaged_part
        if len(self.averaged_tensor_parts) == len(self.local_tensor_parts):
            ordered_averaged_parts = [self.averaged_tensor_parts[endpoint] for endpoint in self.ordered_group_endpoints]
            outputs = restore_from_parts(ordered_averaged_parts, self.tensor_shapes)

            if self.return_deltas:
                local_parts = [self.local_tensor_parts[peer] for peer in self.ordered_group_endpoints]
                with torch.no_grad():
                    original_tensors = restore_from_parts(local_parts, self.tensor_shapes)
                    for averaged_tensor, original_tensor in zip(outputs, original_tensors):
                        averaged_tensor -= original_tensor

            self.future.set_result(outputs)

    def cancel(self) -> bool:
        if not self.future.done():
            logger.debug(f"{self} - cancelled")
            self.future.cancel()
            if not self.averaged_part.done():
                self.averaged_part.cancel()
            return True
        else:
            logger.debug(f"{self} - failed to cancel, allreduce is already finished: {self.future}")
            return False

    def set_exception(self, exception: Exception) -> bool:
        if not self.future.done():
            logger.debug(f"{self} - {exception}")
            self.future.set_exception(exception)
            if not self.averaged_part.done():
                self.averaged_part.cancel()
            return True
        else:
            logger.debug(f"{self} - failed to set {exception}, allreduce already finished: {self.future}")
            return False


class AllReduceRunner(AllReduceProtocol, averaging_pb2_grpc.DecentralizedAveragingServicer):
    """
    A class that implements ButterflyAllReduceProtocol on top of a gRPC servicer
    """

    def __init__(self, *, group_id: GroupID, tensors: Sequence[torch.Tensor], endpoint: Endpoint,
                 ordered_group_endpoints: Sequence[Endpoint], compression_type: runtime_pb2.CompressionType,
                 chunk_size_bytes: int, part_sizes: Tuple[int, ...], weights: Tuple[float, ...],
                 gathered: Dict[Endpoint, Any], return_deltas: bool = False, **kwargs):
        super().__init__(group_id=group_id, tensors=tensors, endpoint=endpoint, part_sizes=part_sizes,
                         ordered_group_endpoints=ordered_group_endpoints, return_deltas=return_deltas, **kwargs)
        self.compression_type, self.chunk_size_bytes, self.gathered = compression_type, chunk_size_bytes, gathered
        self.peer_weights = dict(zip(self.ordered_group_endpoints, weights))

    def _get_peer_stub(self, peer: Endpoint) -> averaging_pb2_grpc.DecentralizedAveragingStub:
        return ChannelCache.get_stub(peer, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)

    async def _communicate_with_peer(self, peer_endpoint: Endpoint, local_part: torch.Tensor) -> torch.Tensor:
        """ Send a part of local tensors and metadata to a single peer, receive the average for that part of tensors """
        assert self.peer_modes[self.endpoint] != AveragingMode.AUX, "Auxiliary peers are disallowed from sending tensors"
        if peer_endpoint == self.endpoint:
            return await self.accumulate_part(self.endpoint, local_part, weight=self.peer_weights[self.endpoint])
        serialized_tensor_part = serialize_torch_tensor(local_part, self.compression_type, allow_inplace=False)
        chunks = split_for_streaming(serialized_tensor_part, self.chunk_size_bytes)

        stream = self._get_peer_stub(peer_endpoint).rpc_aggregate_part()
        await stream.write(averaging_pb2.AveragingData(code=averaging_pb2.PART_FOR_AVERAGING, group_id=self.group_id,
                                                       endpoint=self.endpoint, tensor_part=next(chunks)))
        for chunk in chunks:
            await stream.write(averaging_pb2.AveragingData(tensor_part=chunk))
        await stream.done_writing()

        outputs: Sequence[averaging_pb2.AveragingData] = [message async for message in stream]
        code = outputs[0].code if outputs else averaging_pb2.INTERNAL_ERROR
        if code != averaging_pb2.AVERAGED_PART:
            raise AllreduceException(f"peer {peer_endpoint} returned {averaging_pb2.MessageCode.Name(code)}"
                                     f" instead of {averaging_pb2.MessageCode.Name(averaging_pb2.AVERAGED_PART)},"
                                     f" allreduce failed")

        try:
            averaged_part = local_part + deserialize_torch_tensor(combine_from_streaming(
                [message.tensor_part for message in outputs]))
        except RuntimeError as e:
            raise AllreduceException(f"Could not deserialize averaged part from {peer_endpoint}: {e}")

        self.register_averaged_part(peer_endpoint, averaged_part)
        return averaged_part

    async def _send_error_to_peer(self, peer_endpoint: Endpoint, code: averaging_pb2.MessageCode):
        stream = self._get_peer_stub(peer_endpoint).rpc_aggregate_part()
        await stream.write(averaging_pb2.AveragingData(group_id=self.group_id, endpoint=self.endpoint, code=code))
        await stream.done_writing()

    async def run(self) -> Sequence[torch.Tensor]:
        """
        send allreduce requests to all peers and collect results, return the averaged tensor (or deltas)
        """
        try:
            if self.peer_modes[self.endpoint] != AveragingMode.AUX:
                await asyncio.gather(self, *(self._communicate_with_peer(peer, self.local_tensor_parts[peer])
                                            for i, peer in enumerate(self.ordered_group_endpoints)
                                            if self.peer_modes[peer] != AveragingMode.CLIENT))
            return await self
        except BaseException as e:
            code = averaging_pb2.CANCELLED if isinstance(e, asyncio.CancelledError) else averaging_pb2.INTERNAL_ERROR
            logger.debug(f"{self} - notifying peers about {averaging_pb2.MessageCode.Name(code)}")
            self.set_exception(e)
            for peer_endpoint, part_size in zip(self.ordered_group_endpoints, self.part_sizes):
                if peer_endpoint != self.endpoint and part_size > 0:
                    asyncio.create_task(self._send_error_to_peer(peer_endpoint, code))
            raise

    async def accumulate_part_streaming(self, source: Endpoint, stream_messages: Iterable[runtime_pb2.Tensor]
                                        ) -> Iterable[runtime_pb2.Tensor]:
        """ accumulate_part using streams of serialized tensors. Used to prevent duplicate work in serialization """
        try:
            tensor_part = deserialize_torch_tensor(combine_from_streaming(stream_messages))
        except RuntimeError as e:
            raise AllreduceException(f"Could not deserialize tensor part from {source} for streaming {e}")

        averaged_part = await self.accumulate_part(source, tensor_part, weight=self.peer_weights[source])
        serialized_tensor = serialize_torch_tensor(averaged_part - tensor_part, self.compression_type, allow_inplace=False)
        stream_chunks = tuple(split_for_streaming(serialized_tensor, self.chunk_size_bytes))
        return stream_chunks

    async def rpc_aggregate_part(self, stream: AsyncIterator[averaging_pb2.AveragingData], context: grpc.ServicerContext
                                 ) -> AsyncIterator[averaging_pb2.AveragingData]:
        """ a groupmate sends us a part of his tensor; we should average it with other peers and return the delta"""
        request: averaging_pb2.AveragingData = await anext(stream)

        if request.group_id != self.group_id:
            yield averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)

        elif request.code == averaging_pb2.PART_FOR_AVERAGING:
            try:
                tensor_chunks = (request.tensor_part, *[msg.tensor_part async for msg in stream])
                averaged_chunks = iter(await self.accumulate_part_streaming(request.endpoint, tensor_chunks))
                yield averaging_pb2.AveragingData(code=averaging_pb2.AVERAGED_PART, tensor_part=next(averaged_chunks))
                for averaged_chunk in averaged_chunks:
                    yield averaging_pb2.AveragingData(tensor_part=averaged_chunk)

            except Exception as e:
                self.set_exception(e)
                yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
        else:
            error_code = averaging_pb2.MessageCode.Name(request.code)
            logger.debug(f"{self} - peer {request.endpoint} sent {error_code}, allreduce cannot continue")
            self.set_exception(AllreduceException(f"peer {request.endpoint} sent {error_code}."))
            yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)


def split_into_parts(tensors: Sequence[torch.Tensor], part_sizes: Tuple[int, ...]) -> Tuple[torch.Tensor, ...]:
    """ combines averaged_tensors into one tensor and splits them into equal chunks of size group_size """
    flat_tensor = torch.cat(tuple(map(torch.Tensor.flatten, tensors)))
    return torch.split_with_sizes(flat_tensor, part_sizes, dim=0)


def restore_from_parts(chunks: Sequence[torch.Tensor], shapes: Sequence[torch.Size]) -> Tuple[torch.Tensor, ...]:
    """ restores the original tensor shapes from chunks obtained by split_into_chunks """
    flat_tensor = torch.cat(tuple(chunks))
    result_sizes = tuple(map(torch.Size.numel, shapes))
    flat_original_tensors = torch.split_with_sizes(flat_tensor, result_sizes)
    return tuple(map(torch.Tensor.reshape, flat_original_tensors, shapes))


class AllreduceException(Exception):
    """ A special exception that is raised when allreduce can't continue normally (e.g. disbanded/bad request/etc) """
