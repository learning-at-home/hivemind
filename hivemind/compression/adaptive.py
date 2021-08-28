from typing import Optional

import torch

from hivemind import TensorDescriptor
from hivemind.compression.base import Compression
from hivemind.proto import runtime_pb2


class SizeAdaptiveCompressionStrategy(Compression):
    """Apply compression strategy 1 if tensor has more than :threshold: elements and strategy 2 otherwise"""

    def __init__(self, threshold: int, compression1: Compression, compression2: Compression):
        self.threshold, self.compression1, self.compression2 = threshold, compression1, compression2

    def __call__(
        self, tensor: torch.Tensor, tensor_descr: Optional[TensorDescriptor] = None, **kwargs
    ) -> runtime_pb2.Tensor:
        numel = tensor_descr.numel() if tensor_descr is not None else tensor.numel()
        compression = self.compression1 if numel > self.threshold else self.compression2
        return compression(tensor, tensor_descr=tensor_descr, **kwargs)
