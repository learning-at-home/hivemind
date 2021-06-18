from typing import Sequence

import torch
import numpy as np


class TensorPartContainer:
    """
    Auxiliary object that splits model tensors into parts for all-reduce and combines them back together
    """

    # TODO current state:
    # - it splits parts. Hopefully correctly
    # - need some kind of interface to combine parts back together

    def __init__(self, tensors: Sequence[torch.Tensor], part_sizes: Sequence[float], chunk_size: int = 2 ** 16):
        self.tensors, self.part_sizes, self.chunk_size = tensors, part_sizes, chunk_size
        self.tensor_sizes = [tensor.numel() for tensor in tensors]
        self.total_size = sum(self.tensor_sizes)
        self.chunks_by_peer = []
        self.num_chunks_by_tensor = []

        # split chunks in proportion to part_sizes
        current_length = 0
        current_peer_index = 0
        current_peer_chunks = []
        pivots = np.cumsum(part_sizes) / np.sum(part_sizes) * self.total_size
        pivots = np.concatenate([pivots.astype(np.int64)[:-1], [self.total_size]])

        for tensor in self.tensors:
            tensor_chunks = tensor.view(-1).split(chunk_size)
            self.num_chunks_by_tensor.append(len(tensor_chunks))
            for chunk in tensor_chunks:
                if current_length + len(chunk) > pivots[current_peer_index]:
                    self.chunks_by_peer.append(current_peer_chunks)
                    current_peer_chunks = []
                    current_peer_index += 1
                    if current_length + len(chunk) // 2 > pivots[current_peer_index]:
                        current_peer_chunks.append(chunk)
                    else:
                        self.chunks_by_peer[-1].append(chunk)
                else:
                    current_peer_chunks.append(chunk)
                current_length += len(chunk)

        self.chunks_by_peer.append(current_peer_chunks)
        assert len(self.chunks_by_peer) == self.group_size

    @property
    def group_size(self) -> int:
        return len(self.part_sizes)
