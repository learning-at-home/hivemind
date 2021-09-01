import dataclasses
from abc import ABC
from enum import Enum, auto
from typing import Optional, Union, Any

import numpy as np
import torch

from hivemind.proto import runtime_pb2
from hivemind.utils.tensor_descr import TensorDescriptor

UID = Any


class TensorRole(Enum):
    ACTIVATION = auto()
    PARAMETER = auto()
    GRADIENT = auto()
    OPTIMIZER = auto()
    UNSPECIFIED = auto()


@dataclasses.dataclass(frozen=True)
class CompressionInfo:
    """Auxiliary data structure that contains information about the tensor that determines how it is compressed"""

    uid: UID  # name or index of the tensor from named parameters, optimizer state dict or i/o structure
    descriptor: TensorDescriptor  # data structure that defines shape, dtype, layout and device information
    role: TensorRole = TensorRole.UNSPECIFIED  # which role does the tensor play with respect to the model
    part_index: int = 0  # if tensor is sliced into parts, this represents the index within one tensor
    part_size: Optional[int] = None  # if tensor is sliced into parts, this is the _maximum_ number of values per part

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, uid: UID = None, descriptor: TensorDescriptor = None, **kwargs):
        return cls(uid, descriptor or TensorDescriptor.from_tensor(tensor), **kwargs)


class CompressionBase(ABC):
    """A base class that applies compression algorithm to a pytorch tensor"""

    compression_type: runtime_pb2.CompressionType

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        """
        Applies compression algorithm to a tensor based on their meta-parameters

        :param tensor: a pytorch tensor to compress; depending on the applicaiton, x is either a full tensor or part
        :param info: meta-information about the tensor; if partitioning is used, this still describes the full tensor
        :param allow_inplace: if True, compression can (but doesn't have to) to modify tensor in-place for efficiency
        :returns: a protobuf message that encodes the tensor
        """
        raise NotImplementedError()

    def extract(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        """Create a pytorch tensor from the serialized outputs of .compress"""
        raise NotImplementedError()

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        """Estimate the compression ratio without doing the actual compression; lower ratio = better compression"""
        raise NotImplementedError()


class NoCompression(CompressionBase):
    """A dummy compression strategy that preserves the original tensor as is."""

    compression_type = runtime_pb2.CompressionType.NONE

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        array = tensor.numpy()
        return runtime_pb2.Tensor(
            compression=self.compression_type,
            buffer=array.tobytes(),
            size=array.shape,
            dtype=array.dtype.name,
            requires_grad=tensor.requires_grad,
        )

    def extract(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        array = np.frombuffer(serialized_tensor.buffer, dtype=np.dtype(serialized_tensor.dtype))
        return torch.as_tensor(array).reshape(tuple(serialized_tensor.size))

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        return 1.0
