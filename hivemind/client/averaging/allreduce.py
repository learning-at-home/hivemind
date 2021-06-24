import asyncio
from typing import Sequence, Set, Dict, Tuple, Iterable, AsyncIterator, Any, Optional
from enum import Enum

import grpc
import torch

from hivemind.client.averaging.partition import TensorPartContainer, TensorPartReducer, AllreduceException
from hivemind.utils import Endpoint, get_logger, ChannelCache
from hivemind.utils.asyncio import anext, achain, aiter, aenumerate
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


class AllReduceProtocol(averaging_pb2_grpc.DecentralizedAveragingServicer):
    """
    An internal class that runs butterfly AllReduce in a predefined group of averagers

    :param tensors: local tensors that should be averaged with groupmates
    :param endpoint: your endpoint, must be included in ordered_group_endpoints
    :param ordered_group_endpoints: group endpoints ordered s.t. i-th endpoint is responsible for averaging i-th part
    :param peer_fractions: for each peer, a target fraction of vector elements that this peer should average
      (the actual number of values by peer will be nearly proportional, but there are no exact guarantees)
    :param return_deltas: if True, returns the element-wise differences (averaged_tensors - original_tensors)
           default (False) - return averaged_tensors by themselves
    """

    def __init__(self, *, group_id: GroupID, tensors: Sequence[torch.Tensor], endpoint: Endpoint,
                 ordered_group_endpoints: Sequence[Endpoint], peer_fractions: Tuple[float, ...],
                 weights: Sequence[float], modes: Optional[Sequence[AveragingMode]], return_deltas: bool = False,
                 compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
                 part_size_bytes: int = 2 ** 20, gathered: Dict[Endpoint, Any]):
        assert endpoint in ordered_group_endpoints, "endpoint is not a part of the group"
        if modes is None:
            modes = [AveragingMode.CLIENT if frac == 0 else AveragingMode.NODE for frac in peer_fractions]
        assert any(mode != AveragingMode.CLIENT for mode in modes), "Cannot run allreduce without reducers."
        assert all(mode != AveragingMode.CLIENT or (frac == 0 and weight == 0) for mode, frac, weight
                   in zip(modes, peer_fractions, weights)), "client peers should have zero fraction and zero weight"

        self.group_id, self.endpoint, self.ordered_group_endpoints = group_id, endpoint, ordered_group_endpoints
        self.compression_type, self.part_size_bytes, self.gathered = compression_type, part_size_bytes, gathered
        self.endpoint_index = self.ordered_group_endpoints.index(self.endpoint)
        self.modes, self.peer_fractions = modes, peer_fractions
        self.return_deltas = return_deltas
        self._future = asyncio.Future()

        self.sender_endpoints, self.sender_weights = zip(*(
            (endpoint, weight) for endpoint, weight, mode in zip(self.ordered_group_endpoints, weights, modes)
            if mode != AveragingMode.AUX))
        self.sender_to_index = {endpoint: i for i, endpoint in enumerate(self.sender_endpoints)}

        self.tensor_part_container = TensorPartContainer(tensors, peer_fractions, compression_type, part_size_bytes)
        self.parts_for_local_averaging = self.tensor_part_container.get_raw_input_parts()
        self.tensor_part_reducer = TensorPartReducer(tuple(part.shape for part in self.parts_for_local_averaging),
                                                     len(self.sender_endpoints), self.sender_weights)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.endpoint}, group_size={self.group_size})"

    def __await__(self):
        return self._future.__await__()

    def __contains__(self, endpoint: Endpoint):
        return endpoint in self.ordered_group_endpoints

    @property
    def group_size(self):
        return len(self.ordered_group_endpoints)

    def _get_peer_stub(self, peer: Endpoint) -> averaging_pb2_grpc.DecentralizedAveragingStub:
        return ChannelCache.get_stub(peer, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)

    async def run(self) -> None:
        """
        send allreduce requests to all peers and collect results, return the averaged tensor (or deltas)
        """
        try:
            if len(self.sender_endpoints) == 0:
                logger.debug(f"{self} - finished all-reduce early: all peers are auxiliaries ({self.modes})")
                self.terminate()
                return

            elif self.endpoint in self.sender_endpoints:
                await asyncio.gather(self, *map(self._communicate_with_peer, self.sender_endpoints))

            return await self
        except BaseException as e:
            code = averaging_pb2.CANCELLED if isinstance(e, asyncio.CancelledError) else averaging_pb2.INTERNAL_ERROR
            logger.debug(f"{self} - notifying peers about {averaging_pb2.MessageCode.Name(code)}")
            self.terminate(exception=e)
            for peer_endpoint, mode in zip(self.ordered_group_endpoints, self.modes):
                if peer_endpoint != self.endpoint and mode != AveragingMode.CLIENT:
                    asyncio.create_task(self._send_error_to_peer(peer_endpoint, code))
            raise

    # TODO EVERYTHING BEFORE THIS SHOULD BE OKAY
    async def _communicate_with_peer(self, peer_endpoint: Endpoint):
        """ Send a part of local tensors and metadata to a single peer, receive the average for that part of tensors """
        if peer_endpoint == self.endpoint:
            sender_index = self.sender_endpoints.index(self.endpoint)
            for part_index, tensor_part in enumerate(self.parts_for_local_averaging):
                averaged_part = await self.tensor_part_reducer.accumulate_part(sender_index, part_index, tensor_part)
                self.tensor_part_container.append_averaged_part(sender_index, averaged_part)


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

    # TODO EVERYTHING AFTER THIS SHOULD BE OKAY
    async def rpc_aggregate_part(self, stream: AsyncIterator[averaging_pb2.AveragingData], context: grpc.ServicerContext
                                 ) -> AsyncIterator[averaging_pb2.AveragingData]:
        """ a peer sends us a part of his tensor; we should average it with other peers and return the difference """
        request: averaging_pb2.AveragingData = await anext(stream)
        reason_to_reject = self._check_reasons_to_reject(request)
        if reason_to_reject:
            yield reason_to_reject
            return

        elif request.code == averaging_pb2.PART_FOR_AVERAGING:
            try:
                stream = achain(aiter(request), stream)
                async for part_index, msg in aenumerate(stream):
                    # TODO async deserialize stream with a single BG thread
                    tensor_part = deserialize_torch_tensor(msg.tensor_part) #TODO BACKGROUND THREAD
                    averaged_part = await self.tensor_part_reducer.accumulate_part(
                        self.sender_to_index[request.endpoint], part_index, tensor_part)

                    serialized_delta = serialize_torch_tensor(averaged_part - tensor_part, self.compression_type)#TODO BACKGROUND THREAD
                    yield averaging_pb2.AveragingData(code=averaging_pb2.AVERAGED_PART, tensor_part=serialized_delta)

            except Exception as e:
                self.terminate(exception=e)
                yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
        else:
            error_code = averaging_pb2.MessageCode.Name(request.code)
            logger.debug(f"{self} - peer {request.endpoint} sent {error_code}, allreduce cannot continue")
            self.terminate(exception=AllreduceException(f"peer {request.endpoint} sent {error_code}."))
            yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)

    def _check_reasons_to_reject(self, request: averaging_pb2.AveragingData) -> Optional[averaging_pb2.AveragingData]:
        if request.group_id != self.group_id:
            return averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)
        elif self._future.cancelled():
            return averaging_pb2.AveragingData(code=averaging_pb2.CANCELLED)
        elif self._future.exception():
            return averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)

    async def _send_error_to_peer(self, peer_endpoint: Endpoint, code: averaging_pb2.MessageCode):
        stream = self._get_peer_stub(peer_endpoint).rpc_aggregate_part()
        await stream.write(averaging_pb2.AveragingData(group_id=self.group_id, endpoint=self.endpoint, code=code))
        await stream.done_writing()

    def terminate(self, *, cancel: bool = False, exception: Optional[BaseException] = None):
        assert not cancel or not exception, "finalize accepts either exception or cancel, but not both"
        if not self._future.done():
            logger.debug(f"{self} - {'cancelled' if cancel else exception or 'finished'}")
            if cancel:
                self._future.cancel()
            elif exception:
                self._future.set_exception(exception)
            else:
                self._future.set_result(None)
            self.tensor_part_container.terminate()
            self.tensor_part_reducer.terminate()
            return True
        else:
            logger.debug(f"{self} - could not finish: allreduce is already finished: {self._future}")
            return False


class AllReduceRunner(AllReduceProtocol, ):
    """
    A class that implements ButterflyAllReduceProtocol on top of a gRPC servicer
    """

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


