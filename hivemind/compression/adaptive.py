from abc import ABC, abstractmethod
from typing import Mapping, Sequence, Union

import torch

from hivemind.compression.base import CompressionBase, CompressionInfo, Key, NoCompression, TensorRole
from hivemind.compression.serialization import deserialize_torch_tensor
from hivemind.proto import runtime_pb2


class AdaptiveCompressionBase(CompressionBase, ABC):
    @abstractmethod
    def choose_compression(self, info: CompressionInfo) -> CompressionBase:
        ...

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        return self.choose_compression(info).estimate_compression_ratio(info)

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        return self.choose_compression(info).compress(tensor, info=info, allow_inplace=allow_inplace)

    def extract(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        return deserialize_torch_tensor(serialized_tensor)


class SizeAdaptiveCompression(AdaptiveCompressionBase):
    """Apply compression strategy 1 if tensor has more than :threshold: elements and strategy 2 otherwise"""

    def __init__(self, threshold: int, less: CompressionBase, greater_equal: CompressionBase):
        self.threshold, self.less, self.greater_equal = threshold, less, greater_equal

    def choose_compression(self, info: CompressionInfo) -> CompressionBase:
        return self.greater_equal if info.descriptor.numel() >= self.threshold else self.less


class RoleAdaptiveCompression(AdaptiveCompressionBase):
    """Compress a tensor based on its role in training. Any non-specified compressions will use the "default" option"""

    def __init__(
        self,
        *,
        activation: CompressionBase = None,
        parameter: CompressionBase = None,
        gradient: CompressionBase = None,
        optimizer: CompressionBase = None,
        default: CompressionBase = NoCompression()
    ):
        self.role_compressions = {
            TensorRole.ACTIVATION: activation or default,
            TensorRole.PARAMETER: parameter or default,
            TensorRole.GRADIENT: gradient or default,
            TensorRole.OPTIMIZER: optimizer or default,
            TensorRole.UNSPECIFIED: default,
        }

    def choose_compression(self, info: CompressionInfo) -> CompressionBase:
        return self.role_compressions[info.role]


class PerTensorCompression(AdaptiveCompressionBase):
    """Manually specify the compression strategy depending on tensor key"""

    def __init__(self, tensor_compressions: Union[Sequence[CompressionBase], Mapping[Key, CompressionBase]]):
        self.tensor_compressions = tensor_compressions

    def choose_compression(self, info: CompressionInfo) -> CompressionBase:
        return self.tensor_compressions[info.key]
