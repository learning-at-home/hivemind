from shap.explainers.gradient import torch

from hivemind.compression.base import Compression, CompressionInfo
from hivemind.proto import runtime_pb2


class NoCompression(Compression):
    """A dummy compression strategy that preserves the original tensor as is."""
    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        return runtime_pb2.Tensor()

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        return 1.0

    def restore(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        array = np.frombuffer(serialized_tensor.buffer, dtype=np.dtype(serialized_tensor.dtype))
        return torch.as_tensor(array).reshape(serialized_tensor.size)
