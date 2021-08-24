import asyncio
from enum import Enum
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple, Type

import torch

from hivemind.averaging.partition import AllreduceException, TensorPartContainer, TensorPartReducer
from hivemind.p2p import P2P, P2PContext, PeerID, ServicerBase, StubBase
from hivemind.proto import averaging_pb2
from hivemind.utils import get_logger
from hivemind.utils.asyncio import achain, aenumerate, afirst, aiter, amap_in_executor, anext
from hivemind.utils.compression import deserialize_torch_tensor, serialize_torch_tensor

# flavour types
GroupID = bytes
logger = get_logger(__name__)


class AveragingMode(Enum):
    NODE = 0
    CLIENT = 1
    AUX = 2


class AllReduceRunner(ServicerBase):
    """
    An internal class that runs butterfly AllReduce in a predefined group of averagers.

    This class inherits hivemind.p2p.ServicerBase, so it can be used as an RPCServicer for testing purposes without
    creating a full DecentralizedAverager.

    :note: this class returns **differences** between averaged and local tensors in order to improve numerical stability
    :param p2p: a hivemind.p2p.P2P instance used for communication with other peers
    :param servicer_type: a hivemind.p2p.ServicerBase subclass whose RPC signatures are used
      when requesting other peers. Typically, it is DecentralizedAverager, its derivative,
      or AllReduceRunner itself (for testing purposes).
    :param prefix: namespace for servicer's RPCs (typically, equal to prefix for group keys)
    :param group_id: unique identifier of this specific all-reduce run
    :param tensors: local tensors that should be averaged with groupmates
    :param tensors: local tensors that should be averaged with groupmates
    :param peer_id: your peer_id, must be included in ordered_peer_ids
    :param ordered_peer_ids: group peer_ids ordered s.t. i-th peer_id is responsible for averaging i-th part
    :param peer_fractions: for each peer, a target fraction of vector elements that this peer should average
      (the actual number of values by peer will be nearly proportional, but there are no exact guarantees)
    :param modes: AveragingMode for each peer in ordered_peer_ids (normal, client-only or auxiliary)
    :param weights: scaling coefficients for weighted averaging (default = equal weights for all non-aux peers)
    :param gathered: additional user-defined data collected from this group
    :param kwargs: additional paramters (e.g. part_size_bytes) will be passed to TensorPartContainer
    """

    def __init__(
        self,
        *,
        p2p: P2P,
        servicer_type: Type[ServicerBase],
        prefix: Optional[str],
        group_id: GroupID,
        tensors: Sequence[torch.Tensor],
        ordered_peer_ids: Sequence[PeerID],
        peer_fractions: Tuple[float, ...],
        weights: Optional[Sequence[float]] = None,
        modes: Optional[Sequence[AveragingMode]] = None,
        gathered: Optional[Dict[PeerID, Any]] = None,
        **kwargs,
    ):
        self._p2p = p2p
        self.peer_id = p2p.peer_id
        assert self.peer_id in ordered_peer_ids, "peer_id is not a part of the group"

        if not issubclass(servicer_type, ServicerBase):
            raise TypeError("`servicer_type` is expected to be a ServicerBase subclass")
        self._servicer_type = servicer_type
        self._prefix = prefix

        modes = modes or tuple(AveragingMode.CLIENT if frac == 0 else AveragingMode.NODE for frac in peer_fractions)
        weights = weights or tuple(int(mode != AveragingMode.AUX) for mode in modes)
        assert len(weights) == len(modes) == len(ordered_peer_ids), "lists have inconsistent length"
        assert any(mode != AveragingMode.CLIENT for mode in modes), "cannot run allreduce without reducers"
        for mode, frac, weight in zip(modes, peer_fractions, weights):
            assert mode != AveragingMode.CLIENT or frac == 0, "client-mode peer should have zero all-reduce fraction"
            assert mode != AveragingMode.AUX or weight == 0, "auxiliary peer should have zero averaging weight"

        self.group_id, self.ordered_peer_ids = group_id, ordered_peer_ids
        self.modes, self.peer_fractions, self.gathered = modes, peer_fractions, gathered

        self._future = asyncio.Future()

        self.sender_peer_ids, self.sender_weights = [], []
        for peer_id, weight, mode in zip(self.ordered_peer_ids, weights, modes):
            if mode != AveragingMode.AUX:
                self.sender_peer_ids.append(peer_id)
                self.sender_weights.append(weight)

        peer_id_index = self.ordered_peer_ids.index(self.peer_id)
        self.tensor_part_container = TensorPartContainer(tensors, peer_fractions, **kwargs)
        self.parts_for_local_averaging = self.tensor_part_container.get_raw_input_parts(peer_id_index)
        self.tensor_part_reducer = TensorPartReducer(
            tuple(part.shape for part in self.parts_for_local_averaging),
            len(self.sender_peer_ids),
            self.sender_weights,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.peer_id}, group_size={self.group_size})"

    def __aiter__(self):
        return self.run()

    def __contains__(self, peer_id: PeerID):
        return peer_id in self.ordered_peer_ids

    @property
    def group_size(self):
        return len(self.ordered_peer_ids)

    def _get_peer_stub(self, peer: PeerID) -> StubBase:
        return self._servicer_type.get_stub(self._p2p, peer, namespace=self._prefix)

    async def run(self) -> AsyncIterator[torch.Tensor]:
        """Run all-reduce, return differences between averaged and original tensors as they are computed"""
        pending_tasks = set()
        try:
            if len(self.sender_peer_ids) == 0:
                logger.debug(f"{self} - finished all-reduce early: all peers are auxiliaries ({self.modes})")
                self.finalize()

            elif self.peer_id in self.sender_peer_ids:
                for peer_id, parts in zip(self.ordered_peer_ids, self.tensor_part_container.num_parts_by_peer):
                    if parts != 0:
                        pending_tasks.add(asyncio.create_task(self._communicate_with_peer(peer_id)))

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
            raise

    async def _communicate_with_peer(self, peer_id: PeerID):
        """Send a part of local tensors and metadata to a single peer, receive the average for that part of tensors"""
        peer_index = self.ordered_peer_ids.index(peer_id)
        if peer_id == self.peer_id:
            sender_index = self.sender_peer_ids.index(peer_id)
            for part_index, tensor_part in enumerate(self.parts_for_local_averaging):
                averaged_part = await self.tensor_part_reducer.accumulate_part(sender_index, part_index, tensor_part)
                self.tensor_part_container.register_processed_part(peer_index, part_index, averaged_part - tensor_part)

        else:
            loop = asyncio.get_event_loop()
            code = None
            stream = self._get_peer_stub(peer_id).rpc_aggregate_part(self._generate_input_for_peer(peer_index))
            async for part_index, msg in aenumerate(stream):
                if code is None:
                    code = msg.code
                averaged_part_delta = await loop.run_in_executor(None, deserialize_torch_tensor, msg.tensor_part)
                self.tensor_part_container.register_processed_part(peer_index, part_index, averaged_part_delta)

            if code != averaging_pb2.AVERAGED_PART:
                raise AllreduceException(
                    f"peer {peer_id} returned {averaging_pb2.MessageCode.Name(code)} "
                    f"instead of {averaging_pb2.MessageCode.Name(averaging_pb2.AVERAGED_PART)}"
                    f", allreduce failed"
                )

    async def _generate_input_for_peer(self, peer_index: int) -> AsyncIterator[averaging_pb2.AveragingData]:
        parts_aiter = self.tensor_part_container.iterate_input_parts_for(peer_index)
        first_part = await anext(parts_aiter)
        yield averaging_pb2.AveragingData(
            code=averaging_pb2.PART_FOR_AVERAGING,
            group_id=self.group_id,
            tensor_part=first_part,
        )
        async for part in parts_aiter:
            yield averaging_pb2.AveragingData(tensor_part=part)

    async def rpc_aggregate_part(
        self, stream: AsyncIterator[averaging_pb2.AveragingData], context: P2PContext
    ) -> AsyncIterator[averaging_pb2.AveragingData]:
        """a peer sends us a part of his tensor; we should average it with other peers and return the difference"""
        request: averaging_pb2.AveragingData = await anext(stream)
        reason_to_reject = self._check_reasons_to_reject(request)
        if reason_to_reject:
            yield reason_to_reject
            return

        elif request.code == averaging_pb2.PART_FOR_AVERAGING:
            try:
                sender_index = self.sender_peer_ids.index(context.remote_id)
                async for msg in self._accumulate_parts_streaming(achain(aiter(request), stream), sender_index):
                    yield msg

            except Exception as e:
                self.finalize(exception=e)
                yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
        else:
            error_code = averaging_pb2.MessageCode.Name(request.code)
            logger.debug(f"{self} - peer {context.remote_id} sent {error_code}, allreduce cannot continue")
            self.finalize(exception=AllreduceException(f"peer {context.remote_id} sent {error_code}."))
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
            amap_in_executor(
                lambda msg: (deserialize_torch_tensor(msg.tensor_part), msg.tensor_part.compression),
                stream,
                max_prefetch=self.tensor_part_container.prefetch,
            )
        ):
            averaged_part = await self.tensor_part_reducer.accumulate_part(sender_index, part_index, tensor_part)

            serialized_delta = await loop.run_in_executor(
                None, lambda: serialize_torch_tensor(averaged_part - tensor_part, part_compression)
            )
            yield averaging_pb2.AveragingData(code=averaging_pb2.AVERAGED_PART, tensor_part=serialized_delta)

    async def _send_error_to_peer(self, peer_id: PeerID, code: averaging_pb2.MessageCode):
        error = averaging_pb2.AveragingData(group_id=self.group_id, code=code)
        # Coroutines are lazy, so we take the first item to start the couroutine's execution
        await afirst(self._get_peer_stub(peer_id).rpc_aggregate_part(aiter(error)))

    def finalize(self, *, cancel: bool = False, exception: Optional[BaseException] = None):
        """finish or terminate AllReduceRunner, propagate any errors / cancellations to peers."""
        assert not cancel or not exception, "finalize accepts either exception or cancel, but not both"
        pending_tasks = set()
        if cancel or exception:
            # propagate error to peers
            if cancel or isinstance(exception, asyncio.CancelledError):
                code = averaging_pb2.CANCELLED
            else:
                code = averaging_pb2.INTERNAL_ERROR
            logger.debug(f"{self} - notifying peers about {averaging_pb2.MessageCode.Name(code)}")
            for peer_id, mode in zip(self.ordered_peer_ids, self.modes):
                if peer_id != self.peer_id and mode != AveragingMode.CLIENT:
                    pending_tasks.add(asyncio.create_task(self._send_error_to_peer(peer_id, code)))

        if not self._future.done():
            if cancel:
                logger.debug(f"{self} - cancelled")
                self._future.cancel()
            elif exception:
                logger.debug(f"{self} - caught {exception}")
                self._future.set_exception(exception)
            else:
                logger.debug(f"{self} - finished")
                self._future.set_result(None)
            self.tensor_part_container.finalize()
            self.tensor_part_reducer.finalize()
            return pending_tasks
        else:
            logger.debug(f"{self} - could not finish: allreduce is already finished: {self._future}")
            return pending_tasks
