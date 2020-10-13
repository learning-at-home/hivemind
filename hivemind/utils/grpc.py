"""
Utilities for running GRPC services: compile protobuf, patch legacy versions, etc
"""

import numpy as np
import torch

from hivemind.proto import runtime_pb2
from hivemind.proto.runtime_pb2 import CompressionType

FP16_MAX = 65_504


def serialize_torch_tensor(tensor: torch.Tensor, compression_type=CompressionType.NONE, 
                           allow_inplace=False) -> runtime_pb2.Tensor:
    if compression_type == CompressionType.MEANSTD_LAST_AXIS_FLOAT16:
        assert tensor.dtype == torch.float32

        tensor = tensor if allow_inplace else tensor.clone()
        means = torch.mean(tensor, dim=-1, keepdim=True)
        tensor.sub_(means)

        stds = torch.square(tensor).sum(dim=-1, keepdim=True).div_(tensor.shape[-1]).sqrt_()
        tensor.div_(stds)
        tensor = tensor.clamp_(-FP16_MAX, FP16_MAX).to(torch.float16)

        data = b''.join((tensor.numpy().tobytes(), means.numpy().tobytes(), stds.numpy().tobytes()))

        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=data,
            size=tensor.shape,
            dtype='compressed_float32',
            requires_grad=tensor.requires_grad)
    elif compression_type == CompressionType.FLOAT16:
        assert tensor.dtype == torch.float32

        tensor = tensor if allow_inplace else tensor.clone()
        tensor = tensor.clamp_(-FP16_MAX, FP16_MAX).to(torch.float16)

        data = tensor.numpy().tobytes()

        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=data,
            size=tensor.shape,
            dtype='clamped_float32',
            requires_grad=tensor.requires_grad)
    else:
        array = tensor.numpy()
        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=array.tobytes(),
            size=array.shape,
            dtype=array.dtype.name,
            requires_grad=tensor.requires_grad)

    return proto


def deserialize_torch_tensor(serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
    # TODO avoid copying the array (need to silence pytorch warning, because array is not writable)
    if serialized_tensor.compression == CompressionType.NONE:
        array = np.frombuffer(serialized_tensor.buffer, dtype=np.dtype(serialized_tensor.dtype)).copy()
        tensor = torch.as_tensor(array).view(*serialized_tensor.size).requires_grad_(serialized_tensor.requires_grad)
    elif serialized_tensor.compression == CompressionType.MEANSTD_LAST_AXIS_FLOAT16:
        stats_size = list(serialized_tensor.size)
        stats_size[-1] = 1
        stats_count = np.prod(stats_size)
        means, stds = serialized_tensor.buffer[-8*stats_count:-4*stats_count], serialized_tensor.buffer[-4*stats_count:]
        means = torch.as_tensor(np.frombuffer(means, dtype=np.float32).copy()).view(*stats_size)
        stds = torch.as_tensor(np.frombuffer(stds, dtype=np.float32).copy()).view(*stats_size)
        array = np.frombuffer(serialized_tensor.buffer[:-8 * stats_count], dtype=np.float16).copy()
        tensor = torch.as_tensor(array).to(torch.float32).view(*serialized_tensor.size).mul_(stds).add_(means)
    elif serialized_tensor.compression == CompressionType.FLOAT16:
        array = np.frombuffer(serialized_tensor.buffer, dtype=np.float16).copy()
        tensor = torch.as_tensor(array).view(*serialized_tensor.size)\
            .to(torch.float32).requires_grad_(serialized_tensor.requires_grad)
    else:
        raise ValueError(f"Unknown compression type: {serialized_tensor.compression}")
    return tensor
