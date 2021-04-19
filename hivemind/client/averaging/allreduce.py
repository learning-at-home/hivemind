import asyncio
from typing import Sequence, Set, Dict, Tuple, List, Iterable, AsyncIterator, Any

import grpc
import torch

from hivemind.utils import Endpoint, get_logger, ChannelCache, anext
from hivemind.utils import split_for_streaming, combine_from_streaming
from hivemind.utils.compression import serialize_torch_tensor, deserialize_torch_tensor
from hivemind.proto import averaging_pb2_grpc, runtime_pb2, averaging_pb2

# flavour types
GroupID = bytes
TensorID = int
logger = get_logger(__name__)

class TensorPartContainer:
    def __init__(self, tensors: Sequence[torch.Tensor], part_sizes: Tuple[int, ...], endpoints: Sequence[Endpoint]):
        assert len(endpoints) == len(part_sizes), "length of part_sizes mismatch with endpoints"
        # Sort tensors descending by size
        tensor_ids, tensors = zip(*sorted(enumerate(tensors), key=lambda idx_tensor: -idx_tensor[-1].numel()))
        # Sort peers ascending by part_size
        endpoints, part_sizes = zip(*sorted(zip(endpoints, part_sizes), key=lambda peer_size: peer_size[-1]))

        chunks, chunk_ids = self._split_into_parts(tensors, tensor_ids=tensor_ids, part_sizes=part_sizes)
        self._orig_shapes = {idx: tensor.shape for idx, tensor in zip(tensor_ids, tensors)}

        self._make_views(chunks, chunk_ids, endpoints)
        self.endpoints = tuple(endpoints)

    def _make_views(self, chunks, chunk_ids, endpoints):
        self._peer_tensor_id_view: Dict[Endpoint, Tuple[TensorID, ...]] = dict()
        self._chunk_view: Dict[Tuple[Endpoint, TensorID], torch.Tensor] = dict()

        tensor_parts = []
        for part_chunks, part_ids, endpoint in zip(chunks, chunk_ids, endpoints):
            self._peer_tensor_id_view[endpoint] = tuple(part_ids)

            part_keys = zip([endpoint] * len(part_ids), part_ids)
            tensor_parts.extend(zip(part_keys, part_chunks))

        self._chunk_view = dict(tensor_parts)

    def get_chunk(self, peer: Endpoint, tensor_id: int) -> torch.Tensor:
        return self._chunk_view[(peer, tensor_id)]

    def get_part(self, peer: Endpoint) -> List[torch.Tensor]:
        """Return peer part of tensor chunks. Chunks ordered by ids"""
        return [self.get_chunk(peer, idx) for idx in self._peer_tensor_id_view[peer]]

    def get_part_with_ids(self, peer: Endpoint) -> List[Tuple[TensorID, torch.Tensor]]:
        """Return peer part of tensor chunks. Chunks ordered by ids"""
        return list(zip(self._peer_tensor_id_view[peer], self.get_part(peer)))

    def set_chunk(self, chunk: torch.Tensor, peer: Endpoint, tensor_id: int):
        assert self._chunk_view[(peer, tensor_id)].shape == chunk.shape, "chunk shape mismatch"
        assert self._chunk_view[(peer, tensor_id)].dtype == chunk.dtype, "chunk dtype mismatch"
        self._chunk_view[(peer, tensor_id)] = chunk

    def set_part(self, peer_part: List[torch.Tensor], peer: Endpoint, tensor_ids: List[TensorID] = None):
        """Chunks must be ordered by tensor id"""
        if tensor_ids is None:
            tensor_ids = self._peer_tensor_id_view[peer]
        for tensor_id, tensor in zip(tensor_ids, peer_part):
            self.set_chunk(tensor, peer, tensor_id)

    def assert_part(self, peer_part: List[torch.Tensor], peer: Endpoint, tensor_ids: List[TensorID] = None, msg_template='{err_type}'):
        if tensor_ids is None:
        tensor_ids = self._peer_tensor_id_view[peer]
        for tensor_id, peer_chunk in zip(tensor_ids, peer_part):
            chunk = self._chunk_view[(peer, tensor_id)]
            assert peer_chunk.shape == chunk.shape, msg_template.format("chunk shape mismatch")
            assert peer_chunk.dtype == chunk.dtype, msg_template.format("chunk dtype mismatch")

    @property
    def tensors(self) -> Tuple[torch.Tensor, ...]:
        part_keys, chunks = self._chunk_view.items()
        _, chunk_ids = zip(*part_keys)
        return self._restore_from_parts(chunk_ids, chunks, self._orig_shapes)

    @staticmethod
    def _split_into_parts(tensors: Sequence[torch.Tensor],
                          tensor_ids: Sequence[torch.Tensor],
                          part_sizes: Tuple[int]) -> Tuple[Sequence[List[torch.Tensor]], Sequence[List[int]]]:
        """ combines averaged_tensors into one tensor and splits them into equal chunks of size group_size """
        tensors = tuple(map(torch.Tensor.flatten, tensors))
        enumerated_tensors = zip(tensor_ids, tensors)
        chunks, chunk_ids = [], []

        for part_size in part_sizes:
            part, part_ids = [], []
            accum_size = 0

            for tensor_id, tensor in enumerated_tensors:
                tensor_size = tensor.numel()

                if accum_size + tensor_size <= part_size:
                    part.append(tensor)
                    part_ids.append(tensor_id)
                    accum_size += tensor_size

                    if accum_size == part_size:
                        break
                else:
                    residue = part_size - accum_size
                    print(accum_size, tensor_size, part_size, residue)
                    shards = tensor[:residue], tensor[residue:]

                    part.append(shards[0])
                    part_ids.append(tensor_id)
                    enumerated_tensors = iter([(tensor_id, shards[1])] + list(enumerated_tensors))
                    break

            chunks.append(part)
            chunk_ids.append(part_ids)
        return chunks, chunk_ids

    @staticmethod
    def _restore_from_parts(chunk_ids: Sequence[TensorID],
                            chunks: Sequence[torch.Tensor],
                            tensor_shapes: Dict[TensorID, torch.Size]) -> Tuple[torch.Tensor, ...]:
        """ restores the original tensor shapes from chunks obtained by split_into_chunks """
        restored, restored_ids, shapes = [], [], []
        for tensor_id in chunk_ids:
            if tensor_id not in restored_ids:
                tensor = torch.cat([tensor for tid, tensor in zip(chunk_ids, chunks) if tid == tensor_id])

                restored_ids.append(tensor_id)
                restored.append(tensor)
                shapes.append(tensor_shapes[tensor_id])

        return tuple(map(torch.Tensor.reshape, restored, shapes))



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
                 ordered_group_endpoints: Sequence[Endpoint], part_sizes: Tuple[int, ...], return_deltas: bool = False):
        assert endpoint in ordered_group_endpoints, "endpoint is not a part of the group"
        self.group_id, self.endpoint = group_id, endpoint
        self.ordered_group_endpoints, self.part_sizes = ordered_group_endpoints, part_sizes
        self.client_mode_endpoints = {endpoint for endpoint, part_size in zip(self.ordered_group_endpoints, part_sizes)
                                      if part_size == 0}
        self.local_tensor_parts = dict(zip(ordered_group_endpoints, split_into_parts(tensors, part_sizes)))

        self.return_deltas = return_deltas

        self.accumulator = torch.zeros_like(self.local_tensor_parts[self.endpoint])
        self.denominator = 0.0  # number of peers added to accumulator or sum of their weights
        self.accumulated_from: Set[Endpoint] = set()  # peers that we have accumulated our part from
        self.averaged_part: asyncio.Future[torch.Tensor] = asyncio.Future()  # will be set to [accumulator / group size]
        self.averaged_tensor_parts: Dict[Endpoint, torch.Tensor] = {}  # averaged chunks from all peers will be put here
        self.future: asyncio.Future[Sequence[torch.Tensor]] = asyncio.Future()  # final result or exception
        for endpoint in self.client_mode_endpoints:
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
        assert not self.endpoint in self.client_mode_endpoints, f"{self.endpoint} is in client mode"
        assert isinstance(weight, (int, float)) and weight > 0, "averaging weights must be a non-negative int/float"
        logger.debug(f"{self} - accumulating tensor part from {source}")

        self.accumulator.add_(remote_part, alpha=weight)
        self.denominator += weight
        self.accumulated_from.add(source)

        assert len(self.accumulated_from) <= self.group_size
        if len(self.accumulated_from) == len(self.local_tensor_parts):
            average_result = self.accumulator.div_(self.denominator)
            self.register_averaged_part(self.endpoint, average_result)
            self.averaged_part.set_result(average_result)

        return await self.averaged_part

    def register_averaged_part(self, source: Endpoint, averaged_part: torch.Tensor):
        assert not self.future.done(), f"already finished allreduce: {self.future}"
        assert source in self.local_tensor_parts, "the provider of averaged part is not from my group"
        assert source not in self.averaged_tensor_parts, "already registered the average from this peer"
        assert averaged_part.shape == self.local_tensor_parts[source].shape, "averaged part shape mismatch"
        assert averaged_part.dtype == self.local_tensor_parts[source].dtype, "averaged part dtype mismatch"
        logger.debug(f"{self} - receiving averaged tensor part from {source}")
        self.averaged_tensor_parts[source] = averaged_part
        if len(self.averaged_tensor_parts) == len(self.local_tensor_parts):
            ordered_averaged_parts = [self.averaged_tensor_parts[endpoint] for endpoint in self.ordered_group_endpoints]
            outputs = restore_from_parts(ordered_averaged_parts,)

            if self.return_deltas:
                local_parts = [self.local_tensor_parts[peer] for peer in self.ordered_group_endpoints]
                with torch.no_grad():
                    original_tensors = restore_from_parts(local_parts,)
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
                 gathered: Dict[Endpoint, Any], return_deltas: bool = False):
        super().__init__(group_id=group_id, tensors=tensors, endpoint=endpoint, part_sizes=part_sizes,
                         ordered_group_endpoints=ordered_group_endpoints, return_deltas=return_deltas)
        self.compression_type, self.chunk_size_bytes, self.gathered = compression_type, chunk_size_bytes, gathered
        self.peer_weights = dict(zip(self.ordered_group_endpoints, weights))
        self.averaged_part_stream: asyncio.Future[Tuple[runtime_pb2.Tensor, ...]] = asyncio.Future()

    def _get_peer_stub(self, peer: Endpoint) -> averaging_pb2_grpc.DecentralizedAveragingStub:
        return ChannelCache.get_stub(peer, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)

    async def _communicate_with_peer(self, peer_endpoint: Endpoint, local_part: torch.Tensor) -> torch.Tensor:
        """ Send a part of local tensors and metadata to a single peer, receive the average for that part of tensors """
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
            averaged_part = deserialize_torch_tensor(combine_from_streaming(
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
            await asyncio.gather(self, *(self._communicate_with_peer(peer, self.local_tensor_parts[peer])
                                         for i, peer in enumerate(self.ordered_group_endpoints)
                                         if peer not in self.client_mode_endpoints))
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
        if not self.averaged_part_stream.done():
            serialized_tensor = serialize_torch_tensor(averaged_part, self.compression_type, allow_inplace=False)
            stream_chunks = tuple(split_for_streaming(serialized_tensor, self.chunk_size_bytes))
            self.averaged_part_stream.set_result(stream_chunks)
            return stream_chunks
        else:
            return self.averaged_part_stream.result()

    async def rpc_aggregate_part(self, stream: AsyncIterator[averaging_pb2.AveragingData], context: grpc.ServicerContext
                                 ) -> AsyncIterator[averaging_pb2.AveragingData]:
        """ a groupmate sends us a part of his tensor; we should average it with other peers and return the result """
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

class AllreduceException(Exception):
    """ A special exception that is raised when allreduce can't continue normally (e.g. disbanded/bad request/etc) """
