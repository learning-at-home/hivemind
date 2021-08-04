"""
Auxiliary data structures for AllReduceRunner
"""
import asyncio
from collections import deque
from typing import AsyncIterable, AsyncIterator, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch

from hivemind.proto.runtime_pb2 import CompressionType, Tensor
from hivemind.utils.asyncio import amap_in_executor
from hivemind.utils.compression import get_nbytes_per_value, serialize_torch_tensor

T = TypeVar("T")
DEFAULT_PART_SIZE_BYTES = 2 ** 19


class TensorPartContainer:
    """
    Auxiliary data structure for averaging, responsible for splitting tensors into parts and reassembling them.
    The class is designed to avoid excessive memory allocation and run all heavy computation in background
    :param tensors: local tensors to be split and aggregated
    :param peer_fractions: for each peer, a target fraction of vector elements that this peer should average
    :param compression_type: optionally compress tensors with this compression algorithm before sending them to peers
    :param part_size_bytes: greedily split tensors into parts of up to this many bytes (after compression)
    :param prefetch: when compressing, pre-compute this many compressed tensors in background
    """

    def __init__(
        self,
        tensors: Sequence[torch.Tensor],
        peer_fractions: Sequence[float],
        compression_type: Union["CompressionType", Sequence["CompressionType"]] = CompressionType.NONE,
        part_size_bytes: int = DEFAULT_PART_SIZE_BYTES,
        prefetch: int = 1,
    ):
        if not isinstance(compression_type, Sequence):
            compression_type = [compression_type] * len(tensors)
        assert len(compression_type) == len(tensors), "compression types do not match the number of tensors"
        self.local_tensors, self.peer_fractions, self.group_size = tensors, peer_fractions, len(peer_fractions)
        self.compression_type, self.part_size_bytes, self.prefetch = compression_type, part_size_bytes, prefetch
        self.total_size = sum(tensor.numel() for tensor in tensors)
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

        for tensor, tensor_compression in zip(self.local_tensors, compression_type):
            part_size_values = int(part_size_bytes / get_nbytes_per_value(tensor.dtype, tensor_compression))
            tensor_parts = tensor.detach().view(-1).split(part_size_values)
            self.num_parts_by_tensor.append(len(tensor_parts))
            for part in tensor_parts:
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
                    self._input_parts_by_peer[assigned_peer_index].append((part, tensor_compression))
                else:
                    self._input_parts_by_peer[current_peer_index].append((part, tensor_compression))
                current_length += len(part)

        assert current_length == self.total_size
        self.num_parts_by_peer = tuple(len(parts) for parts in self._input_parts_by_peer)

    @torch.no_grad()
    def get_raw_input_parts(self, peer_index: int) -> Tuple[torch.Tensor, ...]:
        """get non-serialized tensor parts for a peer at a given index"""
        assert not self._inputs_consumed_by_peer[peer_index], "input parts of a given peer are already deallocated."
        self._inputs_consumed_by_peer[peer_index] = True
        input_parts = tuple(part for part, compression in self._input_parts_by_peer[peer_index])
        self._input_parts_by_peer[peer_index].clear()
        return input_parts

    @torch.no_grad()
    async def iterate_input_parts_for(self, peer_index: int) -> AsyncIterator[Tensor]:
        """iterate serialized tensor parts for a peer at a given index. Run serialization in background."""
        assert not self._inputs_consumed_by_peer[peer_index], "input parts of a given peer are already deallocated."
        self._inputs_consumed_by_peer[peer_index] = True

        async def _aiterate_parts():
            for _ in range(self.num_parts_by_peer[peer_index]):
                yield self._input_parts_by_peer[peer_index].popleft()

        async for serialized_part in amap_in_executor(
            lambda x_and_compr: serialize_torch_tensor(*x_and_compr), _aiterate_parts(), max_prefetch=self.prefetch
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
                        raise AllreduceException("All-reduce was terminated during iteration.")

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
                self._input_parts_by_peer[peer_index].clear()
                self._output_parts_by_peer[peer_index].clear()
                self._output_part_available[peer_index].set()
            self._outputs_consumed = True
            self.finished.set()


class TensorPartReducer:
    """
    Auxiliary data structure responsible for running asynchronous all-reduce
    :param part_shapes: a sequence of shapes of torch tensors that will be averaged by this reducer
    :param num_senders: total number of peers in a given all-reduce group that will send gradients
    :param weights: relative importance of each sender, used for weighted average (default = equal weights)
    :note: even if local peer is not sending data, local parts will be used for shape information
    """

    def __init__(self, part_shapes: Sequence[torch.Size], num_senders: int, weights: Optional[Sequence[float]] = None):
        self.part_shapes, self.num_senders, self.num_parts = part_shapes, num_senders, len(part_shapes)
        self.weights = tuple(weights or (1 for _ in range(num_senders)))
        assert len(self.weights) == self.num_senders, "The number of weights is inconsistent with num_senders"
        assert all(isinstance(weight, (int, float)) for weight in self.weights)
        self.current_part_index = -1  # index in local_parts of the part that should be loaded next
        self.current_part_accumulated_from = 0  # number of peers from which the current part was accumulated
        self.accumulator = None  # this will contain the sum of current tensor part from group peers
        self.denominator = 0.0  # total weight accumulated from all peers for current part
        self.current_part_future = asyncio.Future()
        self.finished = asyncio.Event()
        self.reset_accumulators()

    def reset_accumulators(self):
        """(re)create averaging buffers for the next part in line, prepopulate with local tensor part"""
        assert self.current_part_accumulated_from == self.num_senders or self.current_part_index == -1
        if self.current_part_index >= self.num_parts - 1:
            self.finalize()
            return

        self.current_part_index += 1
        self.current_part_accumulated_from = 0
        self.current_part_future = asyncio.Future()
        self.accumulator = torch.zeros(self.part_shapes[self.current_part_index])
        self.denominator = 0.0

    async def accumulate_part(self, sender_index: int, part_index: int, tensor_part: torch.Tensor) -> torch.Tensor:
        """Add vector part to accumulator, wait for all other vectors to be added, then return the average part"""
        assert 0 <= sender_index < self.num_senders, "invalid sender index"
        assert 0 <= part_index < self.num_parts, "invalid part index"

        while part_index > self.current_part_index:
            # wait for previous parts to finish processing ...
            await asyncio.wait({self.current_part_future, self.finished.wait()}, return_when=asyncio.FIRST_COMPLETED)
            if self.finished.is_set():
                raise AllreduceException(f"attempted to aggregate part in a finalized {self.__class__.__name__}")
        assert part_index == self.current_part_index

        current_part_future = self.current_part_future

        self.accumulator.add_(tensor_part, alpha=self.weights[sender_index])
        self.denominator += self.weights[sender_index]
        self.current_part_accumulated_from += 1

        assert self.current_part_accumulated_from <= self.num_senders
        if self.current_part_accumulated_from == self.num_senders:
            current_part_future.set_result(self.accumulator.div_(self.denominator))
            self.reset_accumulators()
        return await current_part_future

    def finalize(self):
        if not self.finished.is_set():
            if hasattr(self, "current_part_future"):
                self.current_part_future.cancel()
                del self.accumulator
            self.finished.set()

    def __del__(self):
        self.finalize()


class AllreduceException(Exception):
    """A special exception that is raised when allreduce can't continue normally (e.g. disconnected/protocol error)"""
