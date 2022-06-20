import asyncio
from enum import Enum
from typing import AsyncIterator, Optional, Sequence, Set, Tuple, Type

import torch

from hivemind.averaging.partition import AllreduceException, BannedException, TensorPartContainer, TensorPartReducer
from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.p2p import P2P, P2PContext, PeerID, ServicerBase, StubBase
from hivemind.proto import averaging_pb2
from hivemind.utils import get_logger
from hivemind.utils.asyncio import (
    achain,
    aiter_with_timeout,
    amap_in_executor,
    anext,
    as_aiter,
    attach_event_on_finished,
)

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
    :param weight: scalar weight of this peer's tensors in the average (doesn't need to sum up to 1)
    :param peer_id: your peer_id, must be included in ordered_peer_ids
    :param ordered_peer_ids: group peer_ids ordered s.t. i-th peer_id is responsible for averaging i-th part
    :param peer_fractions: for each peer, a target fraction of vector elements that this peer should average
      (the actual number of values by peer will be nearly proportional, but there are no exact guarantees)
    :param modes: AveragingMode for each peer in ordered_peer_ids (normal, client-only or auxiliary)
    :param sender_timeout: during all_reduce, any sender that fails to send tensor chunk within this many seconds from
      previous chunk will be marked as failed and excluded from averaging. default: equal to next_chunk_timeout
    :param reducer_timeout: during all_reduce, any reducer that fails to send results chunk within this many seconds
      from previous chunk will be marked as failed and excluded from averaging. default: 2 x sender_timeout
    :param kwargs: additional parameters (e.g. part_size_bytes) will be passed to TensorPartContainer
    :note: Full-mode peers send and receive tensor parts concurrently, assuming a full-duplex TCP stream. In turn,
      non-averaging peers receive results only after they finish sending, which helps them avoid
      throughput issues in case of asymmetric high-latency connections (e.g. ACK compression).
    """

    def __init__(
        self,
        *,
        p2p: P2P,
        servicer_type: Type[ServicerBase],
        prefix: Optional[str],
        group_id: GroupID,
        tensors: Sequence[torch.Tensor],
        weight: Optional[float] = None,
        ordered_peer_ids: Sequence[PeerID],
        peer_fractions: Tuple[float, ...],
        modes: Optional[Sequence[AveragingMode]] = None,
        sender_timeout: Optional[float] = None,
        reducer_timeout: Optional[float] = None,
        **kwargs,
    ):
        self._p2p = p2p
        self.peer_id = p2p.peer_id
        assert self.peer_id in ordered_peer_ids, "peer_id is not a part of the group"
        if reducer_timeout is not None and (sender_timeout is None or reducer_timeout <= sender_timeout):
            raise ValueError(
                "If reducer_timeout is enabled, sender_timeout must be shorter than reducer_timeout. "
                "Otherwise, there is a chance that reducers will be banned while they await senders."
            )

        if not issubclass(servicer_type, ServicerBase):
            raise TypeError("`servicer_type` is expected to be a ServicerBase subclass")
        self._servicer_type = servicer_type
        self._prefix = prefix

        modes = modes or tuple(AveragingMode.CLIENT if frac == 0 else AveragingMode.NODE for frac in peer_fractions)
        assert len(modes) == len(ordered_peer_ids), "lists have inconsistent length"
        assert any(mode != AveragingMode.CLIENT for mode in modes), "cannot run allreduce without reducers"
        for mode, frac in zip(modes, peer_fractions):
            assert mode != AveragingMode.CLIENT or frac == 0, "client-mode peer should have zero all-reduce fraction"

        self.group_id, self.ordered_peer_ids = group_id, ordered_peer_ids
        self.modes, self.peer_fractions = modes, peer_fractions

        if weight is None:
            weight = float(modes[self.ordered_peer_ids.index(self.peer_id)] != AveragingMode.AUX)
        self.weight = weight

        self._future = asyncio.Future()

        self.sender_peer_ids = []
        for peer_id, mode in zip(self.ordered_peer_ids, modes):
            if mode != AveragingMode.AUX:
                self.sender_peer_ids.append(peer_id)

        self.sender_timeout, self.reducer_timeout = sender_timeout, reducer_timeout
        self.all_senders_started = asyncio.Event()
        self.banned_senders: Set[PeerID] = set()  # peers that did not send data by next_chunk_timeout
        self.banlock = asyncio.Lock()

        self.active_senders: Set[PeerID] = set()  # peers that began sending data via rpc_aggregate_part
        if self.peer_id in self.sender_peer_ids:
            self.active_senders.add(self.peer_id)
        if len(self.active_senders) == len(self.sender_peer_ids):
            self.all_senders_started.set()

        peer_id_index = self.ordered_peer_ids.index(self.peer_id)
        self.tensor_part_container = TensorPartContainer(tensors, peer_fractions, return_deltas=True, **kwargs)
        self.parts_for_local_averaging = self.tensor_part_container.get_raw_input_parts(peer_id_index)
        self.tensor_part_reducer = TensorPartReducer(
            tuple(part.shape for part in self.parts_for_local_averaging),
            len(self.sender_peer_ids),
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

    def should_delay_results(self, peer_id: PeerID) -> bool:
        return self.peer_fractions[self.ordered_peer_ids.index(peer_id)] == 0

    async def run(self) -> AsyncIterator[torch.Tensor]:
        """Run all-reduce, return differences between averaged and original tensors as they are computed"""
        pending_tasks = set()

        if self.tensor_part_container.num_parts_by_peer[self.ordered_peer_ids.index(self.peer_id)] != 0:
            pending_tasks.add(asyncio.create_task(self._handle_missing_senders()))

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

        finally:
            for task in pending_tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as inner_exc:
                    logger.debug(f"Task {task} failed with {inner_exc}", exc_info=True)

    async def _handle_missing_senders(self):
        """Detect senders that should have sent tensors for averaging, but did not send anything within timeout"""
        try:
            await asyncio.wait_for(self.all_senders_started.wait(), self.sender_timeout)
        except asyncio.TimeoutError:
            for peer_id in self.sender_peer_ids:
                if peer_id not in self.active_senders and peer_id not in self.banned_senders:
                    await self._ban_sender(peer_id)

    async def _communicate_with_peer(self, peer_id: PeerID):
        """Send a part of local tensors and metadata to a single peer, receive the average for that part of tensors"""
        peer_index = self.ordered_peer_ids.index(peer_id)
        if peer_id == self.peer_id:
            sender_index = self.sender_peer_ids.index(peer_id)
            for part_index, tensor_part in enumerate(self.parts_for_local_averaging):
                averaged_part = await self.tensor_part_reducer.accumulate_part(
                    sender_index, part_index, tensor_part, weight=self.weight
                )
                self.tensor_part_container.register_processed_part(peer_index, part_index, averaged_part - tensor_part)

        else:
            try:
                done_sending = asyncio.Event()
                inputs_aiter = attach_event_on_finished(self._generate_input_for_peer(peer_index), done_sending)
                stream = await self._get_peer_stub(peer_id).rpc_aggregate_part(inputs_aiter)

                if self.should_delay_results(self.peer_id):
                    await done_sending.wait()

                part_index = 0

                def _try_deserialize(msg):
                    if msg.code != averaging_pb2.AVERAGED_PART:
                        raise AllreduceException(f"{peer_id} sent {averaging_pb2.MessageCode.Name(msg.code)}")
                    return deserialize_torch_tensor(msg.tensor_part), msg

                async for delta, msg in amap_in_executor(
                    _try_deserialize,
                    aiter_with_timeout(stream, self.reducer_timeout),
                    max_prefetch=self.tensor_part_container.prefetch,
                ):
                    self.tensor_part_container.register_processed_part(peer_index, part_index, delta)
                    part_index += 1

                if part_index != self.tensor_part_container.num_parts_by_peer[peer_index]:
                    raise AllreduceException(
                        f"peer {peer_id} sent {part_index} parts, but we expected "
                        f"{self.tensor_part_container.num_parts_by_peer[peer_index]}"
                    )
            except BaseException as e:
                if isinstance(e, Exception):
                    logger.debug(f"Caught {repr(e)} when communicating to {peer_id}", exc_info=True)
                self.tensor_part_container.register_failed_reducer(peer_index)
                raise

    async def _generate_input_for_peer(self, peer_index: int) -> AsyncIterator[averaging_pb2.AveragingData]:
        parts_aiter = self.tensor_part_container.iterate_input_parts_for(peer_index)
        first_part = await anext(parts_aiter)
        yield averaging_pb2.AveragingData(
            code=averaging_pb2.PART_FOR_AVERAGING,
            group_id=self.group_id,
            tensor_part=first_part,
            weight=self.weight,
        )
        async for part in parts_aiter:
            yield averaging_pb2.AveragingData(tensor_part=part, weight=self.weight)

    async def rpc_aggregate_part(
        self, stream: AsyncIterator[averaging_pb2.AveragingData], context: P2PContext
    ) -> AsyncIterator[averaging_pb2.AveragingData]:
        """a peer sends us a part of his tensor; we should average it with other peers and return the difference"""
        sender_index = self.sender_peer_ids.index(context.remote_id)
        self.active_senders.add(context.remote_id)
        if len(self.active_senders) == len(self.sender_peer_ids):
            self.all_senders_started.set()

        try:
            request: averaging_pb2.AveragingData = await asyncio.wait_for(anext(stream), self.sender_timeout)
            reason_to_reject = self._check_reasons_to_reject(request, context)
            if reason_to_reject:
                yield reason_to_reject
                return

            elif request.code == averaging_pb2.PART_FOR_AVERAGING:
                stream = aiter_with_timeout(achain(as_aiter(request), stream), self.sender_timeout)
                if not self.should_delay_results(context.remote_id):
                    async for msg in self._accumulate_parts_streaming(stream, sender_index):
                        yield msg

                else:
                    done_receiving = asyncio.Event()
                    delayed_results = asyncio.Queue()

                    async def _accumulate_parts():
                        try:
                            async for msg in self._accumulate_parts_streaming(
                                attach_event_on_finished(stream, done_receiving), sender_index
                            ):
                                delayed_results.put_nowait(msg)
                        finally:
                            delayed_results.put_nowait(None)

                    accumulate_task = asyncio.create_task(_accumulate_parts())

                    await done_receiving.wait()

                    while True:
                        next_result = await delayed_results.get()
                        if next_result is None:
                            break
                        yield next_result
                    await accumulate_task

            else:
                yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
                raise AllreduceException(f"{context.remote_id} sent {averaging_pb2.MessageCode.Name(request.code)}")

        except BaseException as e:
            await self._ban_sender(context.remote_id)
            if isinstance(e, Exception):
                logger.debug(f"Caught {repr(e)} when communicating with {context.remote_id}", exc_info=True)
                yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
            else:
                raise  # CancelledError, StopIteration and similar

    async def _ban_sender(self, peer_id: PeerID):
        async with self.banlock:
            if peer_id not in self.banned_senders:
                self.banned_senders.add(peer_id)
                self.tensor_part_reducer.on_sender_failed(self.sender_peer_ids.index(peer_id))

    def _check_reasons_to_reject(
        self, request: averaging_pb2.AveragingData, context: P2PContext
    ) -> Optional[averaging_pb2.AveragingData]:
        if request.group_id != self.group_id:
            return averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)
        elif self._future.cancelled():
            return averaging_pb2.AveragingData(code=averaging_pb2.CANCELLED)
        elif self._future.done():
            return averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
        elif context.remote_id not in self.sender_peer_ids:
            return averaging_pb2.AveragingData(code=averaging_pb2.PROTOCOL_VIOLATION)

    async def _accumulate_parts_streaming(self, stream: AsyncIterator[averaging_pb2.AveragingData], sender_index: int):
        part_index = 0
        try:
            loop = asyncio.get_event_loop()
            async for tensor_part, weight, part_compression in amap_in_executor(
                lambda msg: (deserialize_torch_tensor(msg.tensor_part), msg.weight, msg.tensor_part.compression),
                stream,
                max_prefetch=self.tensor_part_container.prefetch,
            ):
                try:
                    averaged_part = await self.tensor_part_reducer.accumulate_part(
                        sender_index, part_index, tensor_part, weight=weight
                    )
                    part_index += 1
                except BannedException:
                    logger.debug(f"Sender {sender_index} is already banned")
                    break  # sender was banned, we no longer need to aggregate it

                serialized_delta = await loop.run_in_executor(
                    None, lambda: serialize_torch_tensor(averaged_part - tensor_part, part_compression)
                )
                yield averaging_pb2.AveragingData(code=averaging_pb2.AVERAGED_PART, tensor_part=serialized_delta)
        finally:
            if part_index != self.tensor_part_reducer.num_parts:
                await self._ban_sender(self.sender_peer_ids[sender_index])

    def finalize(self, *, cancel: bool = False, exception: Optional[BaseException] = None):
        """finish or terminate AllReduceRunner, propagate any errors / cancellations to peers."""
        assert not cancel or not exception, "finalize accepts either exception or cancel, but not both"
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
        else:
            logger.debug(f"{self} - attempted to finalize allreduce that is already finished: {self._future}")
