"""
Auxiliary data structures for AllReduceRunner
"""
import asyncio
from collections import deque
from typing import AsyncIterable, AsyncIterator, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch

from hivemind.compression import CompressionBase, CompressionInfo, NoCompression
from hivemind.proto import runtime_pb2
from hivemind.utils import amap_in_executor, as_aiter, get_logger

T = TypeVar("T")
DEFAULT_PART_SIZE_BYTES = 2**19
logger = get_logger(__name__)


class TensorPartContainer:
    """
    Auxiliary data structure for averaging, responsible for splitting tensors into parts and reassembling them.
    The class is designed to avoid excessive memory allocation and run all heavy computation in background

    :param tensors: local tensors to be split and aggregated
    :param peer_fractions: for each peer, a target fraction of vector elements that this peer should average
    :param compression: optionally compress tensors with this compression algorithm before sending them to peers
    :param part_size_bytes: greedily split tensors into parts of up to this many bytes (after compression)
    :param tensor_infos: CompressionInfo for each respective tensor; this determines how the tensor will be comressed
    :param return_deltas: if True, output tensors are differences (aggregated tensor - local tensor)
    :param prefetch: when compressing, pre-compute this many compressed tensors in background
    """

    def __init__(
        self,
        tensors: Sequence[torch.Tensor],
        peer_fractions: Sequence[float],
        compression: CompressionBase = NoCompression(),
        part_size_bytes: int = DEFAULT_PART_SIZE_BYTES,
        tensor_infos: Optional[Sequence[CompressionInfo]] = None,
        return_deltas: bool = True,
        prefetch: int = 1,
    ):
        if tensor_infos is None:
            tensor_infos = tuple(CompressionInfo.from_tensor(x, key=i) for i, x in enumerate(tensors))
        assert len(tensor_infos) == len(tensors), "compression types do not match the number of tensors"
        self.local_tensors, self.peer_fractions, self.group_size = tensors, peer_fractions, len(peer_fractions)
        self.compression, self.part_size_bytes, self.tensor_infos = compression, part_size_bytes, tensor_infos
        self.total_size = sum(tensor.numel() for tensor in tensors)
        self.failed_size = 0
        self.return_deltas = return_deltas
        self.prefetch = prefetch

        self._input_parts_by_peer = [deque() for _ in range(self.group_size)]
        self._output_parts_by_peer = [deque() for _ in range(self.group_size)]
        self._inputs_consumed_by_peer = [False for _ in range(self.group_size)]
        self._output_part_available = [asyncio.Event() for _ in range(self.group_size)]
        self._outputs_registered_by_peer = [0 for _ in range(self.group_size)]
        self._outputs_consumed = False
        self.finished = asyncio.Event()
        self.num_parts_by_tensor = []

        # split tensor parts in proportion to target_size_by_peer
        current_length = 0
        current_peer_index = 0
        pivots = (np.cumsum(peer_fractions) / np.sum(peer_fractions) * self.total_size).astype(np.int64)
        pivots[-1] = self.total_size

        for tensor, info in zip(self.local_tensors, self.tensor_infos):
            bytes_per_value = tensor.element_size() * compression.estimate_compression_ratio(info)
            part_size_values = int(part_size_bytes / bytes_per_value)
            tensor_parts = tensor.detach().view(-1).split(part_size_values)
            self.num_parts_by_tensor.append(len(tensor_parts))
            for part_index, part in enumerate(tensor_parts):
                part_info = info.get_part(part_index, part_size_values)
                if current_length + len(part) > pivots[current_peer_index]:
                    # switch to next peer; if a part lands between parts of two or
                    # more peers, assign that part to the peer with highest intersection
                    prev_peer_index = current_peer_index
                    peer_intersections = [pivots[current_peer_index] - current_length]
                    while current_length + len(part) > pivots[current_peer_index]:
                        current_peer_index += 1
                        current_peer_part_end = min(current_length + len(part), pivots[current_peer_index])
                        peer_intersections.append(current_peer_part_end - pivots[current_peer_index - 1])
                    assigned_peer_index = prev_peer_index + np.argmax(peer_intersections)
                    self._input_parts_by_peer[assigned_peer_index].append((part, part_info))
                else:
                    self._input_parts_by_peer[current_peer_index].append((part, part_info))
                current_length += len(part)

        assert current_length == self.total_size
        self.num_parts_by_peer = tuple(len(parts) for parts in self._input_parts_by_peer)

    @torch.no_grad()
    def get_raw_input_parts(self, peer_index: int) -> Tuple[torch.Tensor, ...]:
        """get non-serialized tensor parts for a peer at a given index"""
        assert not self._inputs_consumed_by_peer[peer_index], "input parts of a given peer are already deallocated."
        self._inputs_consumed_by_peer[peer_index] = True
        input_parts = tuple(part for part, compression in self._input_parts_by_peer[peer_index])
        return input_parts

    @torch.no_grad()
    async def iterate_input_parts_for(self, peer_index: int) -> AsyncIterator[runtime_pb2.Tensor]:
        """iterate serialized tensor parts for a peer at a given index. Run serialization in background."""
        assert not self._inputs_consumed_by_peer[peer_index], "input parts of a given peer are already deallocated."
        self._inputs_consumed_by_peer[peer_index] = True
        parts_aiter = as_aiter(*self._input_parts_by_peer[peer_index])
        async for serialized_part in amap_in_executor(
            lambda x_and_info: self.compression.compress(*x_and_info), parts_aiter, max_prefetch=self.prefetch
        ):
            yield serialized_part

    def register_processed_part(self, peer_index: int, part_index: int, part: torch.Tensor):
        """
        register next-in-line part of results received from a given peer for use in iterate_output_tensors
        depending on the algorithm, processed part is an average, difference from average or another aggregation
        """
        if part_index != self._outputs_registered_by_peer[peer_index]:
            raise ValueError(
                f"Could not register part #{part_index} from peer #{peer_index}, "
                f" expected part index: {self._outputs_registered_by_peer[peer_index]}"
            )
        self._output_parts_by_peer[peer_index].append(part)
        self._outputs_registered_by_peer[peer_index] += 1
        self._output_part_available[peer_index].set()

    def register_failed_reducer(self, peer_index: int):
        """
        a given peer failed to aggregate a certain part, use our local part instead, keep track of failed parts
        """
        for part_index in range(self._outputs_registered_by_peer[peer_index], self.num_parts_by_peer[peer_index]):
            part_and_info = self._input_parts_by_peer[peer_index][part_index]
            part_result_or_delta = torch.zeros_like(part_and_info[0]) if self.return_deltas else part_and_info[0]
            self.register_processed_part(peer_index, part_index, part_result_or_delta)
            self.failed_size += part_result_or_delta.numel()

    async def iterate_output_tensors(self) -> AsyncIterable[torch.Tensor]:
        """iterate over the outputs of averaging (whether they are average, delta or other aggregation result)"""
        assert not self._outputs_consumed, "output tensors are already iterated and no longer available."
        self._outputs_consumed = True
        peer_index = num_parts_processed = 0
        for tensor_index in range(len(self.local_tensors)):
            tensor_parts = []
            while len(tensor_parts) < self.num_parts_by_tensor[tensor_index]:
                if num_parts_processed >= self.num_parts_by_peer[peer_index]:
                    num_parts_processed = 0
                    peer_index += 1
                    continue
                if not self._output_parts_by_peer[peer_index]:
                    self._output_part_available[peer_index].clear()
                    await self._output_part_available[peer_index].wait()
                    if self.finished.is_set():
                        raise AllreduceException("All-reduce was terminated during iteration")

                tensor_parts.append(self._output_parts_by_peer[peer_index].popleft())
                num_parts_processed += 1
            tensor = torch.cat(tensor_parts)
            del tensor_parts
            yield tensor.reshape(self.local_tensors[tensor_index].shape)

    def __del__(self):
        self.finalize()

    def finalize(self):
        """terminate all iterators, delete intermediate data"""
        if not self.finished.is_set():
            for peer_index in range(self.group_size):
                self._inputs_consumed_by_peer[peer_index] = True
                self._output_part_available[peer_index].set()
                self._input_parts_by_peer[peer_index].clear()
                self._output_parts_by_peer[peer_index].clear()
            if self.failed_size != 0:
                logger.warning(f"Averaging: received {(1. - self.failed_size / self.total_size) * 100:.1f}% results")
            self._outputs_consumed = True
            self.finished.set()


class TensorPartReducer:
    """
    Auxiliary data structure responsible for running asynchronous all-reduce
    :param part_shapes: a sequence of shapes of torch tensors that will be averaged by this reducer
    :param num_senders: total number of peers in a given all-reduce group that will send gradients
    :note: even if local peer is not sending data, local parts will be used for shape information
    """

    def __init__(self, part_shapes: Sequence[torch.Size], num_senders: int):
        self.part_shapes, self.num_senders, self.num_parts = part_shapes, num_senders, len(part_shapes)
        self.current_part_index = -1  # index in local_parts of the part that should be loaded next
        self.current_part_accumulated_from = 0  # number of peers from which the current part was accumulated
        self.accumulator = None  # this will contain the sum of current tensor part from group peers
        self.denominator = 0.0  # total weight accumulated from all peers for current part
        self.current_part_future = asyncio.Future()
        self.finished = asyncio.Event()

        self.num_parts_received = [0 for _ in range(self.num_senders)]
        self.sender_failed_after = [float("inf") for _ in range(self.num_senders)]
        self.num_current_senders = self.num_senders

        self.reset_accumulators()

    def reset_accumulators(self):
        """(re)create averaging buffers for the next part in line, prepopulate with local tensor part"""
        assert self.current_part_accumulated_from == self.num_current_senders or self.current_part_index == -1
        if self.current_part_index >= self.num_parts - 1:
            self.finalize()
            return

        self.current_part_index += 1
        self.current_part_accumulated_from = 0
        self.current_part_future = asyncio.Future()
        self.num_current_senders = sum(
            self.current_part_index < failed_index for failed_index in self.sender_failed_after
        )
        self.accumulator = torch.zeros(self.part_shapes[self.current_part_index])
        self.denominator = 0.0

    async def accumulate_part(
        self, sender_index: int, part_index: int, tensor_part: torch.Tensor, weight: float = 1.0
    ) -> torch.Tensor:
        """Add vector part to accumulator, wait for all other vectors to be added, then return the average part"""
        assert 0 <= sender_index < self.num_senders, "invalid sender index"
        assert 0 <= part_index < self.num_parts, "invalid part index"
        self.num_parts_received[sender_index] += 1

        while part_index > self.current_part_index:
            # wait for previous parts to finish processing ...
            await asyncio.wait({self.current_part_future, self.finished.wait()}, return_when=asyncio.FIRST_COMPLETED)
            if self.finished.is_set():
                raise AllreduceException(f"attempted to aggregate part in a finalized {self.__class__.__name__}")

        if self.sender_failed_after[sender_index] != float("inf"):
            raise BannedException(f"sender {sender_index} was banned in background")
        assert part_index == self.current_part_index

        current_part_future = self.current_part_future

        if part_index < self.sender_failed_after[sender_index]:
            self.accumulator.add_(tensor_part, alpha=weight)
            self.current_part_accumulated_from += 1
            self.denominator += weight
            self.check_current_part_finished()
        return await current_part_future

    def on_sender_failed(self, sender_index: int):
        """Exclude that sender's data for averaging any parts that it did not submit yet."""
        self.sender_failed_after[sender_index] = self.num_parts_received[sender_index]
        if self.finished.is_set():
            return
        if self.current_part_index == self.num_parts_received[sender_index]:
            self.num_current_senders -= 1
            self.check_current_part_finished()

    def check_current_part_finished(self):
        assert self.current_part_accumulated_from <= self.num_current_senders
        if self.current_part_accumulated_from == self.num_current_senders:
            self.current_part_future.set_result(self.accumulator.div_(self.denominator))
            self.reset_accumulators()

    def finalize(self):
        if not self.finished.is_set():
            if hasattr(self, "current_part_future"):
                self.current_part_future.cancel()
                del self.accumulator
            self.finished.set()

            if self.num_parts != 0 and self.num_senders != 0:
                parts_expected = self.num_parts * self.num_senders
                parts_received = sum(self.num_parts_received)
                if parts_expected != parts_received:
                    logger.warning(f"Reducer: received {parts_received / parts_expected * 100:.1f}% of input tensors")

    def __del__(self):
        self.finalize()


class AllreduceException(Exception):
    """A special exception that is raised when allreduce can't continue normally (e.g. disconnected/protocol error)"""


class BannedException(AllreduceException):
    """An exception that indicates that a given sender was banned and will no longer be aggregated"""
