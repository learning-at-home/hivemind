"""
Auxiliary data structures for AllReduceProtocol and AllReduceRunner
"""
import asyncio
from typing import Sequence, Awaitable, AsyncIterable, Tuple, Optional

import torch
import numpy as np
from hivemind.proto import runtime_pb2
from hivemind.utils.compression import serialize_torch_tensor, deserialize_torch_tensor, get_nbytes_per_value
from collections import deque


class TensorPartContainer:
    """
    Auxiliary data structure for averaging, responsible for splitting tensors into parts and re-assembling them.
    The class is designed to avoid excessive memory allocation and run all heavy computation in background
    """

    def __init__(self, tensors: Sequence[torch.Tensor], target_size_by_peer: Sequence[float],
                 compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
                 part_size_bytes: int = 2 ** 20):
        self.local_tensors, self.part_sizes, self.part_size_bytes = tensors, target_size_by_peer, part_size_bytes
        self.group_size = len(target_size_by_peer)
        self.tensor_sizes = [tensor.numel() for tensor in tensors]
        self.compression_type = compression_type
        self.total_size = sum(self.tensor_sizes)
        self.num_parts_by_peer, self.num_parts_by_tensor = [], []
        self._input_parts_by_peer = [deque() for _ in range(self.group_size)]
        self._output_parts_by_peer = [deque() for _ in range(self.group_size)]
        self._inputs_consumed_by_peer = [False for _ in range(self.group_size)]
        self._output_part_available = [asyncio.Event() for _ in range(self.group_size)]
        self._outputs_consumed = False

        # split tensor parts in proportion to target_size_by_peer
        current_length = 0
        current_peer_index = 0
        pivots = np.cumsum(target_size_by_peer) / np.sum(target_size_by_peer) * self.total_size
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
            next_part = await prefetch_next_part
            prefetch_next_part = _serialize_next_part()
            yield next_part
        yield await prefetch_next_part

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
                    await self._output_part_available[peer_index].wait()
                tensor_parts.append(await self._output_parts_by_peer[peer_index].popleft())
                num_parts_processed += 1
            tensor = torch.cat(tensor_parts)
            del tensor_parts
            yield tensor.reshape(self.local_tensors[tensor_index].shape)


class TensorPartReducer:
    """
    Auxiliary data structure responsible for running asynchronous all-reduce
    :param local_parts: a sequence of local torch tensors that will be averaged with other peers
    :param index: index of local peer among non-aux peers in a given all-reduce group, aux peers should use None
    :param num_senders: total number of peers in a given all-reduce group that will send gradients
    :note: if sender_index is None, local parts will only be used for shape information
    """
    next_part_index: int = 0  # index in local_parts of the part that should be loaded next
    accumulator: torch.Tensor  # sum of current tensor part from group peers
    denominator: float  # total weight accumulated from all peers for current part
    current_part_accumulated_from: int  # number of peers from which the current part was accumulated (including self)

    def __init__(self, local_parts: Sequence[torch.Tensor], num_senders: int, index: Optional[int],
                 weights: Optional[Sequence[float]] = None):
        self.local_parts, self.num_senders, self.index = local_parts, num_senders, index
        self.num_parts_accumulated = [0 for _ in range(num_senders)]
        self.weights = tuple(weights or (1 for _ in range(num_senders)))
        assert len(weights) == self.num_senders

    def prepare_next_part(self):
        """ create averaging buffers for the part part in line, pre-populates with local tensor part """
        self.accumulator = torch.zeros_like(self.local_parts[self.next_part_index])
        self.denominator = 0.0

        if self.index is not None:
            self.accumulator.add_(self.local_parts[self.next_part_index], alpha=self.weights[])
            self.denominator += self.weights[self.index]
            self.num_parts_accumulated[self.peer_index] += 1
            self.current_part_accumulated_from = 1

    async def accumulate_part(self, peer_index, remote_part: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        """ Add vector part to accumulator, wait for all other vectors to be added, then return the average part """
        assert self.num_parts_accumulated[peer_index] < len(self.local_parts), "peer has already accumulated all parts"
        assert 0 < peer_index < self.group_size and peer_index != self.peer_index
        #TODO peer modes, maybe use num_senders

        assert source in self.local_tensor_parts, "unexpected source, not a part of current group"
        assert source not in self.current_part_accumulated_from, "duplicate source, already received that part"
        assert self.peer_modes[
                   self.endpoint] != AveragingMode.CLIENT, f"{self.endpoint} is in AveragingMode.client mode"
        assert isinstance(weight, (int, float)) and weight > 0, "averaging weights must be a non-negative int/float"

        logger.debug(f"{self} - accumulating tensor part from {source}")
        self.accumulator.add_(remote_part, alpha=weight)
        self.denominator += weight
        self.current_part_accumulated_from.add(source)

        assert len(self.current_part_accumulated_from) <= self.num_senders
        if len(self.current_part_accumulated_from) == self.num_senders:
            average_result = self.accumulator.div_(self.denominator)
            self.averaged_part.set_result(average_result)

            if self.peer_modes[self.endpoint] == AveragingMode.AUX:
                self.future.set_result(None)  # auxiliary mode has finished averaging
            else:
                self.register_averaged_part(self.endpoint, average_result)

        return await self.averaged_part

