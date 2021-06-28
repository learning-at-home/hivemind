import asyncio
from typing import Sequence, Set, Dict, Tuple, Iterable, AsyncIterator, Any, Optional
from enum import Enum

import grpc
import torch

from hivemind.client.averaging.partition import TensorPartContainer, TensorPartReducer, AllreduceException
from hivemind.utils import Endpoint, get_logger, ChannelCache
from hivemind.utils.asyncio import anext, achain, aiter, aenumerate, async_map, azip
from hivemind.utils.compression import serialize_torch_tensor, deserialize_torch_tensor
from hivemind.proto import averaging_pb2_grpc, runtime_pb2, averaging_pb2

# flavour types
GroupID = bytes
logger = get_logger(__name__)


class AveragingMode(Enum):
    NODE = 0
    CLIENT = 1
    AUX = 2


class AllReduceRunner(averaging_pb2_grpc.DecentralizedAveragingServicer):
    """
    An internal class that runs butterfly AllReduce in a predefined group of averagers

    :note: this class returns **differences** between averaged and local tensors in order to improve numerical stability
    :param group_id: unique identifier of this specific all-reduce run
    :param tensors: local tensors that should be averaged with group-mates
    :param endpoint: your endpoint, must be included in ordered_group_endpoints
    :param ordered_group_endpoints: group endpoints ordered s.t. i-th endpoint is responsible for averaging i-th part
    :param peer_fractions: for each peer, a target fraction of vector elements that this peer should average
      (the actual number of values by peer will be nearly proportional, but there are no exact guarantees)
    :param modes: AveragingMode for each peer in ordered_group_endpoints (normal, client-only or auxiliary)
    :param weights: scaling coefficients for weighted averaging (default = equal weights for all non-aux peers)
    :param gathered: additional user-defined data collected from this group
    :param kwargs: additional paramters (e.g. part_size_bytes) will be passed to TensorPartContainer
    """

    def __init__(
            self, *, group_id: GroupID, tensors: Sequence[torch.Tensor], endpoint: Endpoint,
            ordered_group_endpoints: Sequence[Endpoint], peer_fractions: Tuple[float, ...],
            weights: Optional[Sequence[float]] = None, modes: Optional[Sequence[AveragingMode]] = None,
            gathered: Optional[Dict[Endpoint, Any]] = None, **kwargs):
        assert endpoint in ordered_group_endpoints, "endpoint is not a part of the group"
        modes = modes or [AveragingMode.CLIENT if frac == 0 else AveragingMode.NODE for frac in peer_fractions]
        weights = weights or [1 if mode != AveragingMode.AUX else 0 for mode in modes]
        assert len(weights) == len(modes) == len(ordered_group_endpoints), "lists have inconsistent length"
        assert any(mode != AveragingMode.CLIENT for mode in modes), "cannot run allreduce without reducers"
        for mode, frac, weight in zip(modes, peer_fractions, weights):
            assert mode != AveragingMode.CLIENT or frac == 0, "client-mode peer should have zero all-reduce fraction"
            assert mode != AveragingMode.AUX or weight == 0, "auxiliary peer should have zero averaging weight"

        self.group_id, self.endpoint, self.ordered_group_endpoints = group_id, endpoint, ordered_group_endpoints
        self.modes, self.peer_fractions, self.gathered = modes, peer_fractions, gathered
        self.endpoint_index = self.ordered_group_endpoints.index(self.endpoint)
        self._future = asyncio.Future()

        self.sender_endpoints, self.sender_weights = [], []
        for endpoint, weight, mode in zip(self.ordered_group_endpoints, weights, modes):
            if mode != AveragingMode.AUX:
                self.sender_endpoints.append(endpoint)
                self.sender_weights.append(weight)

        print('!!', kwargs)
        self.tensor_part_container = TensorPartContainer(tensors, peer_fractions, **kwargs)
        self.parts_for_local_averaging = self.tensor_part_container.get_raw_input_parts(
            self.ordered_group_endpoints.index(self.endpoint))
        self.tensor_part_reducer = TensorPartReducer(tuple(part.shape for part in self.parts_for_local_averaging),
                                                     len(self.sender_endpoints), self.sender_weights)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.endpoint}, group_size={self.group_size})"

    def __aiter__(self):
        return self.run()

    def __contains__(self, endpoint: Endpoint):
        return endpoint in self.ordered_group_endpoints

    @property
    def group_size(self):
        return len(self.ordered_group_endpoints)

    def _get_peer_stub(self, peer: Endpoint) -> averaging_pb2_grpc.DecentralizedAveragingStub:
        return ChannelCache.get_stub(peer, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)

    async def run(self) -> AsyncIterator[torch.Tensor]:
        """ run all-reduce, return average tensors """
        pending_tasks = set()
        try:
            if len(self.sender_endpoints) == 0:
                logger.debug(f"{self} - finished all-reduce early: all peers are auxiliaries ({self.modes})")
                self.finalize()

            elif self.endpoint in self.sender_endpoints:
                for endpoint, parts in zip(self.ordered_group_endpoints, self.tensor_part_container.num_parts_by_peer):
                    if parts != 0:
                        pending_tasks.add(asyncio.create_task(self._communicate_with_peer(endpoint)))

                async for averaged_tensor_delta in self.tensor_part_container.iterate_output_tensors():
                    yield averaged_tensor_delta  # delta = averaged_tensor - original_tensor
                self.finalize()

            else:  # auxiliary peer
                await self.tensor_part_reducer.finished.wait()
                self.finalize()

        except BaseException as e:
            self.finalize(exception=e)
            for task in pending_tasks:
                task.cancel()
            code = averaging_pb2.CANCELLED if isinstance(e, asyncio.CancelledError) else averaging_pb2.INTERNAL_ERROR
            logger.debug(f"{self} - notifying peers about {averaging_pb2.MessageCode.Name(code)}")
            for peer_endpoint, mode in zip(self.ordered_group_endpoints, self.modes):
                if peer_endpoint != self.endpoint and mode != AveragingMode.CLIENT:
                    asyncio.create_task(self._send_error_to_peer(peer_endpoint, code))
            raise

    async def _communicate_with_peer(self, peer_endpoint: Endpoint):
        """ Send a part of local tensors and metadata to a single peer, receive the average for that part of tensors """
        peer_index = self.ordered_group_endpoints.index(peer_endpoint)
        if peer_endpoint == self.endpoint:
            sender_index = self.sender_endpoints.index(peer_endpoint)
            for part_index, tensor_part in enumerate(self.parts_for_local_averaging):
                averaged_part = await self.tensor_part_reducer.accumulate_part(sender_index, part_index, tensor_part)
                self.tensor_part_container.register_processed_part(peer_index, part_index, averaged_part - tensor_part)

        else:
            loop = asyncio.get_event_loop()
            stream = self._get_peer_stub(peer_endpoint).rpc_aggregate_part()
            write_task = asyncio.create_task(self._write_to_peer(stream, peer_index))

            try:
                code = None
                async for part_index, msg in aenumerate(stream):
                    if code is None:
                        code = msg.code
                    averaged_part_delta = await loop.run_in_executor(None, deserialize_torch_tensor, msg.tensor_part)
                    self.tensor_part_container.register_processed_part(peer_index, part_index, averaged_part_delta)
                await write_task

                if code != averaging_pb2.AVERAGED_PART:
                    raise AllreduceException(f"peer {peer_endpoint} returned {averaging_pb2.MessageCode.Name(code)} "
                                             f"instead of {averaging_pb2.MessageCode.Name(averaging_pb2.AVERAGED_PART)}"
                                             f", allreduce failed")
            finally:
                if not write_task.done():
                    write_task.cancel()

    async def _write_to_peer(self, stream: grpc.aio.StreamStreamCall, peer_index: int):
        parts_aiter = self.tensor_part_container.iterate_input_parts_for(peer_index)
        first_part = await anext(parts_aiter)
        await stream.write(averaging_pb2.AveragingData(code=averaging_pb2.PART_FOR_AVERAGING,
                                                       group_id=self.group_id, endpoint=self.endpoint,
                                                       tensor_part=first_part))
        async for part in parts_aiter:
            await stream.write(averaging_pb2.AveragingData(tensor_part=part))

        await stream.done_writing()

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
                sender_index = self.sender_endpoints.index(request.endpoint)
                async for msg in self._accumulate_parts_streaming(achain(aiter(request), stream), sender_index):
                    yield msg

            except Exception as e:
                self.finalize(exception=e)
                yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
        else:
            error_code = averaging_pb2.MessageCode.Name(request.code)
            logger.debug(f"{self} - peer {request.endpoint} sent {error_code}, allreduce cannot continue")
            self.finalize(exception=AllreduceException(f"peer {request.endpoint} sent {error_code}."))
            yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)

    def _check_reasons_to_reject(self, request: averaging_pb2.AveragingData) -> Optional[averaging_pb2.AveragingData]:
        if request.group_id != self.group_id:
            return averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)
        elif self._future.cancelled():
            return averaging_pb2.AveragingData(code=averaging_pb2.CANCELLED)
        elif self._future.done():
            return averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)

    async def _accumulate_parts_streaming(self, stream: AsyncIterator[averaging_pb2.AveragingData], sender_index: int):
        loop = asyncio.get_event_loop()
        async for part_index, (tensor_part, part_compression) in aenumerate(
                async_map(lambda msg: (deserialize_torch_tensor(msg.tensor_part), msg.tensor_part.compression), stream,
                          max_prefetch=self.tensor_part_container.prefetch)):
            averaged_part = await self.tensor_part_reducer.accumulate_part(sender_index, part_index, tensor_part)

            serialized_delta = await loop.run_in_executor(
                None, lambda: serialize_torch_tensor(averaged_part - tensor_part, part_compression))
            yield averaging_pb2.AveragingData(code=averaging_pb2.AVERAGED_PART, tensor_part=serialized_delta)

    async def _send_error_to_peer(self, peer_endpoint: Endpoint, code: averaging_pb2.MessageCode):
        stream = self._get_peer_stub(peer_endpoint).rpc_aggregate_part()
        await stream.write(averaging_pb2.AveragingData(group_id=self.group_id, endpoint=self.endpoint, code=code))
        await stream.done_writing()

    def finalize(self, *, cancel: bool = False, exception: Optional[BaseException] = None):
        assert not cancel or not exception, "finalize accepts either exception or cancel, but not both"
        if not self._future.done():
            logger.debug(f"{self} - {'cancelled' if cancel else exception or 'finished'}")
            if cancel:
                self._future.cancel()
            elif exception:
                self._future.set_exception(exception)
            else:
                self._future.set_result(None)
            self.tensor_part_container.finalize()
            self.tensor_part_reducer.finalize()
            return True
        else:
            logger.debug(f"{self} - could not finish: allreduce is already finished: {self._future}")
            return False
