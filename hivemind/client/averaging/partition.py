"""
Auxiliary data structures for AllReduceProtocol and AllReduceRunner
"""
import asyncio
from typing import Sequence, Awaitable, AsyncIterable, Tuple, Optional, TypeVar

import torch
import numpy as np
from hivemind.proto import runtime_pb2
from hivemind.utils.compression import serialize_torch_tensor, deserialize_torch_tensor, get_nbytes_per_value
from collections import deque


# TODO implement
# - stream serializer in background thread
# - per-tensor compression
T = TypeVar('T')


class TensorPartContainer:
    """
    Auxiliary data structure for averaging, responsible for splitting tensors into parts and re-assembling them.
    The class is designed to avoid excessive memory allocation and run all heavy computation in background
    """

    def __init__(self, tensors: Sequence[torch.Tensor], peer_fractions: Sequence[float],
                 compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
                 part_size_bytes: int = 2 ** 20):
        self.local_tensors, self.part_sizes, self.part_size_bytes = tensors, peer_fractions, part_size_bytes
        self.group_size = len(peer_fractions)
        self.tensor_sizes = [tensor.numel() for tensor in tensors]
        self.compression_type = compression_type
        self.total_size = sum(self.tensor_sizes)
        self.num_parts_by_peer, self.num_parts_by_tensor = [], []
        self._input_parts_by_peer = [deque() for _ in range(self.group_size)]
        self._output_parts_by_peer = [deque() for _ in range(self.group_size)]
        self._inputs_consumed_by_peer = [False for _ in range(self.group_size)]
        self._output_part_available = [asyncio.Event() for _ in range(self.group_size)]
        self._outputs_consumed = False
        self.finished = asyncio.Event()

        # split tensor parts in proportion to target_size_by_peer
        current_length = 0
        current_peer_index = 0
        pivots = np.cumsum(peer_fractions) / np.sum(peer_fractions) * self.total_size
        pivots = np.concatenate([pivots.astype(np.int64)[:-1], [self.total_size]])

        for tensor in self.local_tensors:
            part_size_values = int(part_size_bytes / get_nbytes_per_value(tensor.dtype, self.compression_type))
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
                    self._input_parts_by_peer[assigned_peer_index].append(part)
                else:
                    self._input_parts_by_peer[current_peer_index].append(part)
                current_length += len(part)
        assert current_length == self.total_size
        for current_peer_index in range(self.group_size):
            self.num_parts_by_peer.append(len(self._input_parts_by_peer[current_peer_index]))

    @torch.no_grad()
    def get_raw_input_parts(self, peer_index: int) -> Tuple[torch.Tensor, ...]:
        """ get non-serialized tensor parts for a peer at a given index """
        assert not self._inputs_consumed_by_peer[peer_index], "input parts of a given peer are already deallocated."
        self._inputs_consumed_by_peer[peer_index] = True
        input_parts = tuple(self._input_parts_by_peer[peer_index])
        self._input_parts_by_peer[peer_index].clear()
        return input_parts

    @torch.no_grad()
    async def iterate_input_parts_for(self, peer_index: int) -> runtime_pb2.Tensor:
        """ iterate serialized tensor parts for a peer at a given index. Run serialization in background. """
        assert not self._inputs_consumed_by_peer[peer_index], "input parts of a given peer are already deallocated."
        self._inputs_consumed_by_peer[peer_index] = True
        if not self._input_parts_by_peer[peer_index]:
            return
        loop = asyncio.get_event_loop()

        def _serialize_next_part() -> Awaitable[runtime_pb2.Tensor]:
            next_part = self._input_parts_by_peer[peer_index].popleft()
            return loop.run_in_executor(None, lambda: serialize_torch_tensor(next_part, self.compression_type))

        prefetch_next_part = _serialize_next_part()
        for i in range(len(self._input_parts_by_peer[peer_index])):
            next_part = await self._this_or_termination(prefetch_next_part)
            prefetch_next_part = _serialize_next_part()
            yield next_part
        yield await self._this_or_termination(prefetch_next_part)

    def append_averaged_part(self, peer_index: int, part: runtime_pb2.Tensor):
        """ register next-in-line part of results received from a given peer  """
        self._output_parts_by_peer[peer_index].append(
            asyncio.get_event_loop().run_in_executor(None, deserialize_torch_tensor, part))
        self._output_part_available[peer_index].set()

    async def iterate_output_tensors(self) -> AsyncIterable[torch.Tensor]:
        """ iterate over the outputs of averaging (whether they are average, delta or other aggregation result) """
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
                    await self._this_or_termination(self._output_part_available[peer_index].wait())

                next_part = await self._this_or_termination(self._output_parts_by_peer[peer_index].popleft())
                tensor_parts.append(next_part)
                num_parts_processed += 1
            tensor = torch.cat(tensor_parts)
            del tensor_parts
            yield tensor.reshape(self.local_tensors[tensor_index].shape)

    def finalize(self):
        """ terminate all iterators, delete intermediate data """
        for peer_index in range(self.group_size):
            self._inputs_consumed_by_peer[peer_index] = True
            self._input_parts_by_peer[peer_index].clear()
            self._output_parts_by_peer[peer_index].clear()
            self._output_part_available[peer_index].set()
        self._outputs_consumed = True
        self.finished.set()

    async def _this_or_termination(self, coro: Awaitable[T]) -> T:
        await asyncio.wait({coro, self.finished.wait()}, return_when=asyncio.FIRST_COMPLETED)
        if self.finished.is_set():
            raise ValueError(f"attempted to request part from a finalized {self.__class__.__name__}")
        return await coro


class TensorPartReducer:
    """
    Auxiliary data structure responsible for running asynchronous all-reduce
    :param part_shapes: a sequence of shapes of torch tensors that will be averaged by this reducer
    :param num_senders: total number of peers in a given all-reduce group that will send gradients
    :note: even if local peer is not sending data, local parts will be used for shape information
    """
    current_part_index: int = -1  # index in local_parts of the part that should be loaded next
    current_part_accumulated_from: int = 0  # number of peers from which the current part was accumulated
    accumulator: torch.Tensor  # sum of current tensor part from group peers
    denominator: float  # total weight accumulated from all peers for current part
    current_part_future: asyncio.Future  # this future will be set with the current averaged part, once it is ready

    def __init__(self, part_shapes: Sequence[torch.Size], num_senders: int,
                 weights: Optional[Sequence[float]] = None):
        self.part_shapes, self.num_senders, self.num_parts = part_shapes, num_senders, len(part_shapes)
        self.weights = tuple(weights or (1 for _ in range(num_senders)))
        assert len(self.weights) == self.num_senders, "The number of weights is inconsistent with num_senders"
        for weight in self.weights:
            assert isinstance(weight, (int, float)) and weight > 0, "averaging weights must be a non-negative int/float"
        self.finished = asyncio.Event()
        self.reset_accumulators()

    def reset_accumulators(self):
        """ (re)create averaging buffers for the part part in line, pre-populates with local tensor part """
        assert self.current_part_accumulated_from == self.num_senders or self.current_part_index == -1
        if self.current_part_index >= self.num_parts - 1:
            self.finished.set()
            return

        self.current_part_index += 1
        self.current_part_accumulated_from = 0
        self.current_part_future = asyncio.Future()
        self.accumulator = torch.zeros(self.part_shapes[self.current_part_index])
        self.denominator = 0.0

    async def accumulate_part(self, sender_index: int, part_index: int, tensor_part: torch.Tensor) -> torch.Tensor:
        """ Add vector part to accumulator, wait for all other vectors to be added, then return the average part """
        assert 0 <= sender_index < self.num_senders, 'invalid sender index'
        assert 0 <= part_index < self.num_parts, "invalid part index"

        while part_index > self.current_part_index:
            # wait for previous parts to finish processing ...
            await self._this_or_termination(self.current_part_future)
        assert part_index == self.current_part_index

        current_part_future = self.current_part_future

        self.accumulator.add_(tensor_part, alpha=self.weights[sender_index])
        self.denominator += self.weights[sender_index]
        self.current_part_accumulated_from += 1

        assert self.current_part_accumulated_from <= self.num_senders
        if self.current_part_accumulated_from == self.num_senders:
            current_part_future.set_result(self.accumulator.div_(self.denominator))
            self.reset_accumulators()
        return await self._this_or_termination(current_part_future)

    def __await__(self):
        return self.finished.wait().__await__()

    def finalize(self):
        del self.accumulator
        self.current_part_future.cancel()
        self.finished.set()

    async def _this_or_termination(self, coro: Awaitable[T]) -> T:
        await asyncio.wait({coro, self.finished.wait()}, return_when=asyncio.FIRST_COMPLETED)
        if self.finished.is_set():
            raise ValueError(f"attempted to request part from a finalized {self.__class__.__name__}")
        return await coro

