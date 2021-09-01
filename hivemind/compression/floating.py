import math

import numpy as np
import torch

from hivemind.compression.base import CompressionBase, CompressionInfo
from hivemind.proto import runtime_pb2


class Float16Compression(CompressionBase):
    compression_type = runtime_pb2.CompressionType.FLOAT16
    FP16_MIN, FP16_MAX = torch.finfo(torch.float16).min, torch.finfo(torch.float16).max

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        dtype_name = tensor.numpy().dtype.name
        tensor = tensor.detach().cpu().float()
        tensor = tensor if allow_inplace else tensor.clone()
        tensor = tensor.clamp_(self.FP16_MIN, self.FP16_MAX).to(torch.float16)
        return runtime_pb2.Tensor(
            compression=self.compression_type,
            buffer=tensor.numpy().tobytes(),
            size=tensor.shape,
            dtype=dtype_name,
            requires_grad=tensor.requires_grad,
        )

    def extract(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        original_dtype = np.dtype(serialized_tensor.dtype)
        array = np.frombuffer(serialized_tensor.buffer, dtype=np.float16)
        return torch.as_tensor(np.asarray(array, dtype=original_dtype)).reshape(tuple(serialized_tensor.size))

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        return 16.0 / get_num_bits(info.descriptor.dtype)


class ScaledFloat16Compression(Float16Compression):
    """A compression strategy that applies mean-std scaling over last axis before casting to float16"""

    compression_type = runtime_pb2.CompressionType.MEANSTD_16BIT
    FP32_BYTES = torch.finfo(torch.float32).bits // 8
    FP32_EPS = torch.finfo(torch.float32).eps

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        dtype_name = tensor.numpy().dtype.name
        tensor = tensor.detach().cpu().float()
        tensor = tensor if allow_inplace else tensor.clone()
        means = torch.mean(tensor, dim=-1, keepdim=True)
        tensor.sub_(means)
        stds = tensor.norm(dim=-1, keepdim=True) / math.sqrt(tensor.shape[-1])
        stds.clamp_min_(self.FP32_EPS)
        tensor.div_(stds)
        tensor = tensor.clamp_(self.FP16_MIN, self.FP16_MAX).to(torch.float16)

        data = b"".join((tensor.numpy().tobytes(), means.float().numpy().tobytes(), stds.float().numpy().tobytes()))

        return runtime_pb2.Tensor(
            compression=self.compression_type,
            buffer=data,
            size=tensor.shape,
            dtype=dtype_name,
            requires_grad=tensor.requires_grad,
        )

    def extract(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        stats_shape = list(serialized_tensor.size)
        stats_shape[-1] = 1
        stats_count = np.prod(stats_shape)
        means_offset = len(serialized_tensor.buffer) - 2 * stats_count * self.FP32_BYTES
        stds_offset = len(serialized_tensor.buffer) - stats_count * self.FP32_BYTES

        array = np.frombuffer(serialized_tensor.buffer, dtype=np.float16, count=np.prod(serialized_tensor.size))
        means = np.frombuffer(serialized_tensor.buffer, dtype=np.float32, offset=means_offset, count=stats_count)
        stds = np.frombuffer(serialized_tensor.buffer, dtype=np.float32, offset=stds_offset, count=stats_count)

        means = torch.as_tensor(means).reshape(stats_shape)
        stds = torch.as_tensor(stds).reshape(stats_shape)
        tensor = torch.as_tensor(np.asarray(array, dtype=serialized_tensor.dtype)).reshape(
            list(serialized_tensor.size)
        )
        return tensor.mul_(stds).add_(means)


def get_num_bits(dtype: torch.dtype) -> int:
    if dtype == torch.bool:
        return 8  # see https://github.com/pytorch/pytorch/issues/41571
    elif dtype.is_floating_point:
        return torch.finfo(dtype).bits
    else:
        try:
            return torch.iinfo(dtype).bits
        except TypeError:
            raise TypeError(f"Could not infer size for tensor type {dtype}")
