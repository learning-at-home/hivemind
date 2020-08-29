"""
Utilities for running GRPC services: compile protobuf, patch legacy versions, etc
"""

import numpy as np
import torch

from hivemind.proto import runtime_pb2


def serialize_torch_tensor(tensor: torch.Tensor, compression_type="None") -> runtime_pb2.Tensor:
    # TODO change the format of the compression_type argument
    array = tensor.numpy()
    if compression_type == "None":
        proto = runtime_pb2.Tensor(
            compression=runtime_pb2.Tensor.NONE,
            buffer=array.tobytes(),
            size=array.shape,
            dtype=array.dtype.name,
            requires_grad=tensor.requires_grad)
    else:
        assert array.dtype == np.float32
        proto = runtime_pb2.Tensor(
            compression=runtime_pb2.Tensor.HALFPRECISION,
            buffer=array.astype(np.float16).tobytes(),
            size=array.shape,
            dtype=array.dtype.name,
            requires_grad=tensor.requires_grad)

    return proto


def deserialize_torch_tensor(tensor: runtime_pb2.Tensor) -> torch.Tensor:
    # TODO avoid copying the array (need to silence pytorch warning, because array is not writable)
    if tensor.compression == runtime_pb2.Tensor.NONE:
        array = np.frombuffer(tensor.buffer, dtype=np.dtype(tensor.dtype)).copy()
    elif tensor.compression == runtime_pb2.Tensor.HALFPRECISION:
        array = np.frombuffer(tensor.buffer, dtype=np.float16).astype(np.float32).copy()
    return torch.as_tensor(array).view(tuple(tensor.size)).requires_grad_(tensor.requires_grad)
