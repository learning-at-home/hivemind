import asyncio
from typing import Sequence, Awaitable, AsyncIterable

import torch
import numpy as np
from hivemind.proto import runtime_pb2
from hivemind.utils.compression import serialize_torch_tensor, deserialize_torch_tensor, get_nbytes_per_value
from collections import deque


class TensorPartition:
    """
    Auxiliary data structure for averaging, responsible for splitting tensors into parts and re-assembling them.
    The class is designed to avoid excessive memory allocation and run all heavy computation in background
    """

    def __init__(self, tensors: Sequence[torch.Tensor], part_sizes: Sequence[float],
                 compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
                 chunk_size_bytes: int = 2 ** 20):
        self.local_tensors, self.part_sizes, self.chunk_size_bytes = tensors, part_sizes, chunk_size_bytes
        self.tensor_sizes = [tensor.numel() for tensor in tensors]
        self.compression_type = compression_type
        self.total_size = sum(self.tensor_sizes)
        self._input_chunks_by_peer = [deque() for _ in range(self.num_peers)]
        self._output_chunks_by_peer = [deque() for _ in range(self.num_peers)]
        self._inputs_consumed_by_peer = [False for _ in range(self.num_peers)]
        self._output_chunks_available = [asyncio.Event() for _ in range(self.num_peers)]
        self._num_chunks_by_peer, self._num_chunks_by_tensor = [], []
        self._outputs_consumed = False

        # split chunks in proportion to part_sizes
        current_length = 0
        current_peer_index = 0
        pivots = np.cumsum(part_sizes) / np.sum(part_sizes) * self.total_size
        pivots = np.concatenate([pivots.astype(np.int64)[:-1], [self.total_size]])

        for tensor in self.local_tensors:
            chunk_size_values = int(chunk_size_bytes / get_nbytes_per_value(tensor.dtype, self.compression_type))
            tensor_chunks = tensor.detach().view(-1).split(chunk_size_values)
            self._num_chunks_by_tensor.append(len(tensor_chunks))
            for chunk in tensor_chunks:
                if current_length + len(chunk) > pivots[current_peer_index]:
                    # switch to next peer; if a chunk lands between parts of two or
                    # more peers, assign that chunk to the peer with highest intersecton
                    prev_peer_index = current_peer_index
                    peer_intersections = [pivots[current_peer_index] - current_length]
                    while current_length + len(chunk) > pivots[current_peer_index]:
                        current_peer_index += 1
                        current_peer_chunk_end = min(current_length + len(chunk), pivots[current_peer_index])
                        peer_intersections.append(current_peer_chunk_end - pivots[current_peer_index - 1])
                    assigned_peer_index = prev_peer_index + np.argmax(peer_intersections)
                    self._input_chunks_by_peer[assigned_peer_index].append(chunk)
                else:
                    self._input_chunks_by_peer[current_peer_index].append(chunk)
                current_length += len(chunk)
        assert current_length == self.total_size
        for current_peer_index in range(self.num_peers):
            self._num_chunks_by_peer.append(len(self._input_chunks_by_peer[current_peer_index]))

    @property
    def num_peers(self) -> int:
        return len(self.part_sizes)

    @torch.no_grad()
    async def iterate_input_chunks(self, peer_index: int) -> runtime_pb2.Tensor:
        """ get tensor chunks that should be averaged by a peer at a given index. Run serialization in background. """
        assert not self._inputs_consumed_by_peer[peer_index], "input chunks of a given peer are already deallocated."
        self._inputs_consumed_by_peer[peer_index] = True
        if not self._input_chunks_by_peer[peer_index]:
            return
        loop = asyncio.get_event_loop()

        def _serialize_next_chunk() -> Awaitable[runtime_pb2.Tensor]:
            next_chunk = self._input_chunks_by_peer[peer_index].popleft()
            return loop.run_in_executor(None, lambda: serialize_torch_tensor(next_chunk, self.compression_type))

        prefetch_next_chunk = _serialize_next_chunk()
        for i in range(len(self._input_chunks_by_peer[peer_index])):
            next_chunk = await prefetch_next_chunk
            prefetch_next_chunk = _serialize_next_chunk()
            yield next_chunk
        yield await prefetch_next_chunk

    def append_averaged_chunk(self, peer_index: int, chunk: runtime_pb2.Tensor):
        """ register next-in-line chunk of results received from a given peer  """
        self._output_chunks_by_peer[peer_index].append(
            asyncio.get_event_loop().run_in_executor(None, deserialize_torch_tensor, chunk))
        self._output_chunks_available[peer_index].set()

    async def iterate_output_tensors(self) -> AsyncIterable[torch.Tensor]:
        """ iterate over the outputs of averaging (whether they are average, delta or other aggregation result) """
        assert not self._outputs_consumed, "output tensors are already iterated and no longer available."
        self._outputs_consumed = True
        peer_index = num_chunks_processed = 0
        for tensor_index in range(len(self.local_tensors)):
            tensor_chunks = []
            while len(tensor_chunks) < self._num_chunks_by_tensor[tensor_index]:
                if num_chunks_processed >= self._num_chunks_by_peer[peer_index]:
                    num_chunks_processed = 0
                    peer_index += 1
                    continue
                if not self._output_chunks_by_peer[peer_index]:
                    self._output_chunks_available[peer_index].clear()
                    await self._output_chunks_available[peer_index].wait()
                tensor_chunks.append(await self._output_chunks_by_peer[peer_index].popleft())
                num_chunks_processed += 1
            tensor = torch.cat(tensor_chunks)
            del tensor_chunks
            yield tensor.reshape(self.local_tensors[tensor_index].shape)
