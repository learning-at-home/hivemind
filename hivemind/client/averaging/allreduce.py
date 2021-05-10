import asyncio
from typing import Sequence, Set, Dict, Tuple, List, Iterable, AsyncIterator, Any, Union

import grpc
import torch

from hivemind.utils import Endpoint, get_logger, ChannelCache, anext
from hivemind.utils import split_for_streaming, combine_from_streaming
from hivemind.utils.compression import serialize_torch_tensor, deserialize_torch_tensor
from hivemind.proto import averaging_pb2_grpc, runtime_pb2, averaging_pb2
from collections import OrderedDict

# flavour types
GroupID = bytes
TensorID = int
CompressionType = Any
Part = Tuple[torch.Tensor, ...]
logger = get_logger(__name__)


class TensorPartContainer:
    def build_from_tensors(self,
                           tensors: Sequence[torch.Tensor],
                           part_sizes: Tuple[int, ...],
                           endpoints: Sequence[Endpoint],
                           compression_type: Sequence[CompressionType] = None):
        assert len(endpoints) == len(part_sizes), "length of part_sizes mismatch with endpoints"

        # Sort tensors descending by size
        tensor_ids, tensors = zip(*sorted(enumerate(tensors), key=lambda idx_tensor: -idx_tensor[-1].numel()))
        # Sort peers ascending by part_size
        endpoints, part_sizes = zip(*sorted(zip(endpoints, part_sizes), key=lambda peer_size: peer_size[-1]))

        parts, part_ids = self._split_into_parts(tensors, tensor_ids=tensor_ids, part_sizes=part_sizes)
        tensor_shapes = {idx: tensor.shape for idx, tensor in zip(tensor_ids, tensors)}

        self.__init__(parts, part_ids, tensor_shapes, endpoints, compression_type)

    def __init__(self,
                 parts: Sequence[Part],
                 part_ids: Sequence[List[TensorID]],
                 tensor_shapes: Sequence[torch.Size],
                 endpoints: Sequence[Endpoint],
                 compression_type: Sequence[CompressionType] = None):
        assert len(parts) == len(endpoints)
        assert len(parts) == len(part_ids)

        if compression_type is not None:
            assert len(compression_type) == len(tensor_shapes), \
                "length of compression type mismatch with number of tensors"
            self._compression_type = compression_type
        else:
            self._compression_type = None

        self._orig_shapes = tensor_shapes
        self.endpoints = tuple(endpoints)
        self._make_views(parts, part_ids, endpoints)

    def _make_views(self, pieces, piece_ids, endpoints):
        self._peer_tensor_id_view: Dict[Endpoint, Tuple[TensorID, ...]] = dict()
        self._piece_view: Dict[Tuple[Endpoint, TensorID], torch.Tensor] = dict()

        tensor_parts = []
        for part_pieces, part_ids, endpoint in zip(pieces, piece_ids, endpoints):
            self._peer_tensor_id_view[endpoint] = tuple(part_ids)

            part_keys = zip([endpoint] * len(part_ids), part_ids)
            tensor_parts.extend(zip(part_keys, part_pieces))

        self._piece_view = dict(tensor_parts)

    def get_piece(self, peer: Endpoint, tensor_id: int) -> torch.Tensor:
        return self._piece_view[(peer, tensor_id)]

    def get_part(self, peer: Endpoint) -> Part:
        """Return peer part of tensor pieces. pieces ordered by ids"""
        return tuple(self.get_piece(peer, idx) for idx in self._peer_tensor_id_view[peer])

    def get_part_with_ids(self, peer: Endpoint) -> Tuple[Tuple[TensorID, ...], Part]:
        """Return peer part of tensor pieces. pieces ordered by ids"""
        return self._peer_tensor_id_view[peer], self.get_part(peer)

    def set_piece(self, piece: torch.Tensor, peer: Endpoint, tensor_id: int):
        assert self._piece_view[(peer, tensor_id)].shape == piece.shape, "piece shape mismatch"
        assert self._piece_view[(peer, tensor_id)].dtype == piece.dtype, "piece dtype mismatch"
        self._piece_view[(peer, tensor_id)] = piece

    def set_part(self, peer_part: Part, peer: Endpoint, tensor_ids: List[TensorID] = None):
        """pieces must be ordered by tensor id"""
        if tensor_ids is None:
            tensor_ids = self._peer_tensor_id_view[peer]
        for tensor_id, tensor in zip(tensor_ids, peer_part):
            self.set_piece(tensor, peer, tensor_id)

    def get_shapes(self, peer: Endpoint) -> Tuple[torch.Size, ...]:
        return tuple(piece.shape for piece in self.get_part(peer))

    def get_dtypes(self, peer: Endpoint) -> Tuple[torch.dtype, ...]:
        return tuple(piece.dtype for piece in self.get_part(peer))

    def get_part_compression_type(self, peer: Endpoint) -> Tuple[CompressionType, ...]:
        if self._compression_type is None:
            return None
        return tuple(self._compression_type[tensor_id] for tensor_id in self._peer_tensor_id_view[peer])

    @property
    def tensors(self) -> Sequence[torch.Tensor]:
        part_keys, pieces = zip(*self._piece_view.items())
        _, piece_ids = zip(*part_keys)

        tensor_ids = sorted(self._orig_shapes.keys())
        restored = self._restore_from_parts(piece_ids, pieces)
        restored = [restored.get(idx, torch.tensor([])) for idx in tensor_ids]
        shapes = [self._orig_shapes[idx] for idx in tensor_ids]

        return list(map(torch.Tensor.reshape, restored, shapes))

    @staticmethod
    def _split_into_parts(tensors: Sequence[torch.Tensor],
                          tensor_ids: Sequence[torch.Tensor],
                          part_sizes: Tuple[int]) -> Tuple[Sequence[Part], Sequence[List[int]]]:
        """ combines averaged_tensors into one tensor and splits them into equal pieces of size group_size """
        tensors = tuple(map(torch.Tensor.flatten, tensors))
        enumerated_tensors = zip(tensor_ids, tensors)
        peer_parts, peer_part_ids = [], []

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
                    shards = tensor[:residue], tensor[residue:]

                    part.append(shards[0])
                    part_ids.append(tensor_id)
                    enumerated_tensors = iter([(tensor_id, shards[1])] + list(enumerated_tensors))
                    break

            part = tuple(part)
            peer_parts.append(part)
            peer_part_ids.append(part_ids)
        return peer_parts, peer_part_ids

    @staticmethod
    def _restore_from_parts(piece_ids: Sequence[TensorID],
                            pieces: Sequence[torch.Tensor]) -> Dict[TensorID, torch.Tensor]:
        """ restores the original tensor shapes from pieces obtained by split_into_pieces """
        restored, shapes = dict(), []
        for tensor_id in piece_ids:
            if tensor_id not in restored:
                tensor = torch.cat([tensor for tid, tensor in zip(piece_ids, pieces) if tid == tensor_id])
                restored[tensor_id] = tensor

        return restored


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

    def __init__(self,
                 *,
                 group_id: GroupID,
                 tensors: Sequence[torch.Tensor],
                 endpoint: Endpoint,
                 ordered_group_endpoints: Sequence[Endpoint],
                 part_sizes: Tuple[int, ...],
                 return_deltas: bool = False,
                 compression_type: Sequence[CompressionType] = None):
        assert endpoint in ordered_group_endpoints, "endpoint is not a part of the group"
        self.group_id, self.endpoint = group_id, endpoint
        self.ordered_group_endpoints, self.part_sizes = ordered_group_endpoints, part_sizes
        self.client_mode_endpoints = {endpoint for endpoint, part_size in zip(self.ordered_group_endpoints, part_sizes)
                                      if part_size == 0}
        self.compression_type = compression_type
        self.tensor_shapes = tuple(tensor.shape for tensor in tensors)

        parts = split_into_parts(tensors, part_sizes)
        self.local_tensor_parts = TensorPartContainer(
            parts=[(t,) for t in parts],
            part_ids=[[i] for i in range(len(parts))],
            tensor_shapes={i: part.shape for i, part in enumerate(parts)},
            endpoints=ordered_group_endpoints,
            # todo
            compression_type=[self.compression_type[0]]*len(parts)
        )

        self.averaged_tensor_parts = TensorPartContainer(
            parts=[(torch.zeros_like(t),) for t in parts],
            part_ids=[[i] for i in range(len(parts))],
            tensor_shapes={i: part.shape for i, part in enumerate(parts)},
            endpoints=ordered_group_endpoints
        )

        self.accumulators = OrderedDict((idx, torch.zeros_like(piece))
                                        for idx, piece in zip(*self.local_tensor_parts.get_part_with_ids(endpoint)))
        self.denominator = 0.0  # number of peers added to accumulator or sum of their weights
        self.accumulated_from: Set[Endpoint] = set()  # peers that we have accumulated our part from
        self.registered_from: Set[Endpoint] = set()  # peers that we have accumulated our part from
        self.averaged_part: asyncio.Future[Part] = asyncio.Future()  # will be set to [accumulator / group size]

        self.future: asyncio.Future[Sequence[torch.Tensor]] = asyncio.Future()  # final result or exception
        self.return_deltas = return_deltas
        # for endpoint in self.client_mode_endpoints:
        #     self.averaged_tensor_parts[endpoint] = torch.tensor([])
        self.registered_from.update(self.client_mode_endpoints)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.endpoint}, group_size={self.group_size})"

    def __await__(self):
        return self.future.__await__()

    def __contains__(self, endpoint: Endpoint):
        return endpoint in self.ordered_group_endpoints

    @property
    def group_size(self):
        return len(self.ordered_group_endpoints)

    async def accumulate_part(self, source: Endpoint, remote_part: Part, weight: float = 1.0) \
            -> Part:
        """ Add vector part to accumulator, wait for all other vectors to be added, then return the average part """
        assert not self.averaged_part.done(), f"already finished averaging part: {self.averaged_part}"
        assert not self.future.done(), f"already finished allreduce: {self.future}"
        assert source in self.ordered_group_endpoints, "unexpected source, not a part of current group"
        assert source not in self.accumulated_from, "duplicate source, already received that part"
        assert not self.endpoint in self.client_mode_endpoints, f"{self.endpoint} is in client mode"
        assert isinstance(weight, (int, float)) and weight > 0, "averaging weights must be a non-negative int/float"
        logger.debug(f"{self} - accumulating tensor part from {source}")

        tensor_ids, _ = self.local_tensor_parts.get_part_with_ids(self.endpoint)
        assert len(remote_part) == len(tensor_ids)

        self.denominator += weight
        for tensor_id, piece in zip(tensor_ids, remote_part):
            self.accumulators[tensor_id].add_(piece, alpha=weight)

        self.accumulated_from.add(source)
        assert len(self.accumulated_from) <= self.group_size
        if len(self.accumulated_from) == len(self.ordered_group_endpoints):
            for tensor_id, piece in zip(tensor_ids, remote_part):
                self.accumulators[tensor_id].div_(self.denominator)
            average_result = tuple(self.accumulators.values())
            self.register_averaged_part(self.endpoint, average_result)
            self.averaged_part.set_result(average_result)

        return await self.averaged_part

    def register_averaged_part(self, source: Endpoint, averaged_part: Part):
        assert not self.future.done(), f"already finished allreduce: {self.future}"
        assert source in self.ordered_group_endpoints, "the provider of averaged part is not from my group"
        assert source not in self.registered_from, "already registered the average from this peer"
        averaged_part_shapes = tuple(t.shape for t in averaged_part)
        averaged_part_dtypes = tuple(t.dtype for t in averaged_part)
        assert averaged_part_shapes == self.local_tensor_parts.get_shapes(source), \
            f"averaged part shape mismatch {averaged_part_shapes} : {self.local_tensor_parts.get_shapes(source)}"
        assert averaged_part_dtypes == self.local_tensor_parts.get_dtypes(source), "averaged part dtype mismatch"
        logger.debug(f"{self} - receiving averaged tensor part from {source}")

        self.averaged_tensor_parts.set_part(averaged_part, source)
        self.registered_from.add(source)
        if len(self.registered_from) == len(self.ordered_group_endpoints):
            outputs = self.averaged_tensor_parts.tensors
            #todo
            outputs = restore_from_parts(outputs, self.tensor_shapes)

            if self.return_deltas:
                #todo
                local_tensors = self.local_tensor_parts.tensors
                local_tensors = restore_from_parts(local_tensors, self.tensor_shapes)
                with torch.no_grad():
                    for averaged_tensor, original_tensor in zip(outputs, local_tensors):
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

    def __init__(self,
                 *,
                 group_id: GroupID,
                 tensors: Sequence[torch.Tensor],
                 endpoint: Endpoint,
                 ordered_group_endpoints: Sequence[Endpoint],
                 compression_type: Union[CompressionType, List[CompressionType]],
                 chunk_size_bytes: int,
                 part_sizes: Tuple[int, ...],
                 weights: Tuple[float, ...],
                 gathered: Dict[Endpoint, Any],
                 return_deltas: bool = False):
        if not isinstance(compression_type, (list, tuple)):
            compression_type = [compression_type, ] * len(tensors)

        super().__init__(group_id=group_id, tensors=tensors, endpoint=endpoint, part_sizes=part_sizes,
                         ordered_group_endpoints=ordered_group_endpoints, return_deltas=return_deltas,
                         compression_type=compression_type)
        self.chunk_size_bytes, self.gathered = chunk_size_bytes, gathered
        self.peer_weights = dict(zip(self.ordered_group_endpoints, weights))

    def _get_peer_stub(self, peer: Endpoint) -> averaging_pb2_grpc.DecentralizedAveragingStub:
        return ChannelCache.get_stub(peer, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)

    def _serialize_to_chunks(self, local_part: Part, compression_type: Sequence[CompressionType],
                             chunk_size_bytes: int, header=False) -> Iterable[averaging_pb2.AveragingData]:
        assert len(local_part) == len(compression_type)

        for i, (tensor, compression) in enumerate(zip(local_part, compression_type)):
            serialized_tensor = serialize_torch_tensor(tensor, compression, allow_inplace=False)
            chunks = split_for_streaming(serialized_tensor, chunk_size_bytes)

            code = averaging_pb2.PART_FOR_AVERAGING if i == 0 else averaging_pb2.TENSOR_SEP
            if header:
                yield averaging_pb2.AveragingData(code=code, group_id=self.group_id,
                                                  endpoint=self.endpoint, tensor_part=next(chunks))
            for chunk in chunks:
                yield averaging_pb2.AveragingData(tensor_part=chunk)

    def _deserialize_from_chunks(self, chunks: Iterable[averaging_pb2.AveragingData]) -> Part:
        piece, averaged_pieces = [], []
        for chunk in chunks:
            if chunk.code == averaging_pb2.TENSOR_SEP:
                averaged_pieces.append(piece)
                piece = []
            piece.append(chunk.tensor_part)
        averaged_pieces.append(piece)

        averaged_part = []
        for piece in averaged_pieces:
            averaged_part.append(deserialize_torch_tensor(combine_from_streaming(piece)))
        return tuple(averaged_part)

    async def _communicate_with_peer(self, peer_endpoint: Endpoint, local_part: Part) -> Part:
        """ Send a part of local tensors and metadata to a single peer, receive the average for that part of tensors """
        if peer_endpoint == self.endpoint:
            return await self.accumulate_part(self.endpoint, local_part, weight=self.peer_weights[self.endpoint])
        stream = self._get_peer_stub(peer_endpoint).rpc_aggregate_part()

        compression_type = self.local_tensor_parts.get_part_compression_type(peer_endpoint)
        for chunk in self._serialize_to_chunks(local_part, compression_type, self.chunk_size_bytes, header=True):
            await stream.write(chunk)
        await stream.done_writing()

        outputs: Sequence[averaging_pb2.AveragingData] = [message async for message in stream]
        code = outputs[0].code if outputs else averaging_pb2.INTERNAL_ERROR
        if code != averaging_pb2.AVERAGED_PART:
            raise AllreduceException(f"peer {peer_endpoint} returned {averaging_pb2.MessageCode.Name(code)}"
                                     f" instead of {averaging_pb2.MessageCode.Name(averaging_pb2.AVERAGED_PART)},"
                                     f" allreduce failed")

        try:
            delta_part = self._deserialize_from_chunks(outputs)
            averaged_part = tuple(
                local + delta for local, delta in zip(local_part, delta_part))
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
            await asyncio.gather(self, *(self._communicate_with_peer(peer, self.local_tensor_parts.get_part(peer))
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

    async def accumulate_part_streaming(self, source: Endpoint, stream_messages: Iterable[averaging_pb2.AveragingData]
                                        ) -> Iterable[runtime_pb2.Tensor]:
        """ accumulate_part using streams of serialized tensors. Used to prevent duplicate work in serialization """
        try:
            tensor_part = self._deserialize_from_chunks(stream_messages)
        except RuntimeError as e:
            raise AllreduceException(f"Could not deserialize tensor part from {source} for streaming {e}")
        averaged_part = await self.accumulate_part(source, tensor_part, weight=self.peer_weights[source])

        delta_part = tuple(
            averaged - tensor for averaged, tensor in zip(averaged_part, tensor_part))

        compression_type = self.local_tensor_parts.get_part_compression_type(self.endpoint)
        stream_chunks = self._serialize_to_chunks(delta_part, compression_type, self.chunk_size_bytes)

        stream_chunks = [chunk.tensor_part for chunk in stream_chunks]
        return stream_chunks

    async def rpc_aggregate_part(self, stream: AsyncIterator[averaging_pb2.AveragingData], context: grpc.ServicerContext
                                 ) -> AsyncIterator[averaging_pb2.AveragingData]:
        """ a groupmate sends us a part of his tensor; we should average it with other peers and return the delta"""
        request: averaging_pb2.AveragingData = await anext(stream)

        if request.group_id != self.group_id:
            yield averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)

        elif request.code == averaging_pb2.PART_FOR_AVERAGING:
            try:
                tensor_chunks = (request, *[msg async for msg in stream])
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


def split_into_parts(tensors: Sequence[torch.Tensor], part_sizes: Tuple[int]) -> Tuple[torch.Tensor, ...]:
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
