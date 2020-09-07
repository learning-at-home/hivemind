"""
Utilities for running GRPC services: compile protobuf, patch legacy versions, etc
"""

import numpy as np
import torch

from hivemind.proto import runtime_pb2
from hivemind.proto.runtime_pb2 import CompressionType


def serialize_torch_tensor(tensor: torch.Tensor, compression_type=CompressionType.NONE) -> runtime_pb2.Tensor:
    array = tensor.numpy()
    if compression_type == CompressionType.MEANSTD_LAST_AXIS_FLOAT16:
        assert array.dtype == np.float32

        mean = array.mean()
        std = array.std()
        normalized = (array - mean) / std

        data = array.astype(np.float16).tobytes() + np.array([mean, std], dtype=np.float32).tobytes()

        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=data,
            size=array.shape,
            dtype=array.dtype.name,
            requires_grad=tensor.requires_grad)
    else:
        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=array.tobytes(),
            size=array.shape,
            dtype=array.dtype.name,
            requires_grad=tensor.requires_grad)

    return proto


def deserialize_torch_tensor(tensor: runtime_pb2.Tensor) -> torch.Tensor:
    # TODO avoid copying the array (need to silence pytorch warning, because array is not writable)
    if tensor.compression == CompressionType.NONE:
        array = np.frombuffer(tensor.buffer, dtype=np.dtype(tensor.dtype)).copy()
    elif tensor.compression == CompressionType.MEANSTD_LAST_AXIS_FLOAT16:
        mean, std = np.frombuffer(tensor.buffer[-8:], dtype=np.float32)
        array = np.frombuffer(tensor.buffer[:-8], dtype=np.float16).astype(np.float32).copy()
        array *= std
        array += mean
    return torch.as_tensor(array).view(tuple(tensor.size)).requires_grad_(tensor.requires_grad)
