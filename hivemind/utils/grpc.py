"""
Utilities for running GRPC services: compile protobuf, patch legacy versions, etc
"""

import numpy as np
import torch

from hivemind.proto import runtime_pb2


def serialize_torch_tensor(tensor: torch.Tensor) -> runtime_pb2.Tensor:
    array = tensor.numpy()
    proto = runtime_pb2.Tensor(
        buffer=array.tobytes(),
        size=array.shape,
        dtype=array.dtype.name,
        requires_grad=tensor.requires_grad)
    return proto


def deserialize_torch_tensor(tensor: runtime_pb2.Tensor) -> torch.Tensor:
    # TODO avoid copying the array (need to silence pytorch warning, because array is not writable)
    array = np.frombuffer(tensor.buffer, dtype=np.dtype(tensor.dtype)).copy()
    return torch.as_tensor(array).view(tuple(tensor.size)).requires_grad_(tensor.requires_grad)
