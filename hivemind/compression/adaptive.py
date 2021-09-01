from abc import ABC
from typing import Mapping, Union, Sequence

import torch

import hivemind
from hivemind.compression.base import CompressionBase, UID, CompressionInfo, NoCompression, TensorRole
from hivemind.proto import runtime_pb2


class AdaptiveCompressionBase(CompressionBase, ABC):
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


class RoleAdaptiveCompression(AdaptiveCompressionBase):
    """Compress a tensor based on its role in training. Any non-specified compressions will use the "default" option"""

    def __init__(self, *,
                 activation: CompressionBase = None,
                 parameter: CompressionBase = None,
                 gradient: CompressionBase = None,
                 optimizer: CompressionBase = None,
                 default: CompressionBase = NoCompression()):
        self.role_compressions = {
            TensorRole.ACTIVATION: activation or default,
            TensorRole.PARAMETER: parameter or default,
            TensorRole.GRADIENT: gradient or default,
            TensorRole.OPTIMIZER: optimizer or default,
            TensorRole.UNSPECIFIED: default,
        }

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        return self.role_compressions[info.role].estimate_compression_ratio(info)

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        return self.role_compressions[info.role].compress(tensor, info=info, allow_inplace=allow_inplace)


class PerTensorCompression(AdaptiveCompressionBase):
    """Manually specify the compression strategy depending on tensor's uid"""

    def __init__(self, tensor_compressions: Union[Sequence[CompressionBase], Mapping[UID, CompressionBase]]):
        self.tensor_compressions = tensor_compressions

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        return self.tensor_compressions[info.uid].estimate_compression_ratio(info)

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        return self.tensor_compressions[info.uid].compress(tensor, info=info, allow_inplace=allow_inplace)
