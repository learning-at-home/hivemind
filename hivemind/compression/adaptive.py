from typing import Optional, Mapping, Union, Sequence

import torch

import hivemind
from hivemind.compression.base import CompressionBase, CompressionInfo, UID
from hivemind.proto import runtime_pb2


class AdaptiveCompressionBase(CompressionBase):
    def restore(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        return hivemind.compression.deserialize_torch_tensor(serialized_tensor)


class SizeAdaptiveCompression(AdaptiveCompressionBase):
    """Apply compression strategy 1 if tensor has more than :threshold: elements and strategy 2 otherwise"""

    def __init__(self, threshold: int, small: CompressionBase, large: CompressionBase):
        self.threshold, self.small, self.large = threshold, small, large

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        compression = self.small if info.descriptor.numel() > self.threshold else self.large
        return compression.estimate_compression_ratio(info)

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        compression = self.small if info.descriptor.numel() > self.threshold else self.large
        return compression.compress(tensor, info=info, allow_inplace=allow_inplace)


class PerTensorCompression(AdaptiveCompressionBase):
    """Manually specify the compression strategy depending on tensor's uid"""

    def __init__(self, tensor_compressions: Union[Sequence[CompressionBase], Mapping[UID, CompressionBase]]):
        self.tensor_compressions = tensor_compressions

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        return self.tensor_compressions[info.uid].estimate_compression_ratio(info)

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        return self.tensor_compressions[info.uid].compress(tensor, info=info, allow_inplace=allow_inplace)
