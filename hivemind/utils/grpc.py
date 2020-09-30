"""
Utilities for running GRPC services: compile protobuf, patch legacy versions, etc
"""

import numpy as np
import torch

from hivemind.proto import runtime_pb2
from hivemind.proto.runtime_pb2 import CompressionType


def serialize_torch_tensor(tensor: torch.Tensor, compression_type=CompressionType.NONE, allow_inplace=False) -> runtime_pb2.Tensor:
    if compression_type == CompressionType.MEANSTD_LAST_AXIS_FLOAT16:
        assert tensor.dtype == torch.float32
        FP16_MAX = 65_504

        tensor = tensor if allow_inplace else tensor.clone()
        means = torch.mean(tensor, dim=-1, keepdim=True)
        tensor.sub_(means)

        stds = torch.square(tensor).sum(dim=-1, keepdim=True).div_(tensor.shape[-1]).sqrt_()
        tensor.div_(stds)
        tensor.clamp_(-FP16_MAX, FP16_MAX).to(torch.float16)

        data = tensor.numpy().tobytes() + means.numpy().tobytes() + stds.numpy().tobytes()

        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=data,
            size=tensor.shape,
            dtype='compressed_float16',
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


def deserialize_torch_tensor(tensor: runtime_pb2.Tensor) -> torch.Tensor:
    # TODO avoid copying the array (need to silence pytorch warning,x because array is not writable)
    if tensor.compression == CompressionType.NONE:
        array = np.frombuffer(tensor.buffer, dtype=np.dtype(tensor.dtype)).copy()
        array.reshape(tuple(tensor.size))
    elif tensor.compression == CompressionType.MEANSTD_LAST_AXIS_FLOAT16:
        means, stds = tensor.buffer[-8*tensor.size[-1]:-4*tensor.size[-1]], tensor.buffer[-4*tensor.size[-1]:]
        means = torch.as_tensor(np.frombuffer(means))
        stds = torch.as_tensor(np.frombuffer(stds))
        array = np.frombuffer(tensor.buffer[:-8*tensor.size[-1]], dtype=np.float16).astype(np.float32)
        array.reshape(tuple(tensor.size))
        array *= stds
        array += means
    return torch.as_tensor(array).requires_grad_(tensor.requires_grad)
