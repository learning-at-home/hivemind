import dataclasses
import os
import warnings
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Optional

import numpy as np
import torch

from hivemind.proto import runtime_pb2
from hivemind.utils.tensor_descr import TensorDescriptor

# While converting read-only NumPy arrays into PyTorch tensors, we don't make extra copies for efficiency
warnings.filterwarnings("ignore", message="The given NumPy array is not writable", category=UserWarning)

USE_LEGACY_BFLOAT16 = bool(int(os.environ.get("USE_LEGACY_BFLOAT16", 1)))

Key = Any


class TensorRole(Enum):
    ACTIVATION = auto()
    PARAMETER = auto()
    GRADIENT = auto()
    OPTIMIZER = auto()
    UNSPECIFIED = auto()


@dataclasses.dataclass(frozen=True)
class CompressionInfo:
    """Auxiliary data structure that contains information about the tensor that determines how it is compressed"""

    key: Key  # name or index of the tensor from named parameters, optimizer state dict or i/o structure
    descriptor: TensorDescriptor  # data structure that defines shape, dtype, layout and device information
    role: TensorRole = TensorRole.UNSPECIFIED  # which role does the tensor play with respect to the model
    part_index: int = 0  # if tensor is sliced into parts, this represents the index within one tensor
    part_size: Optional[int] = None  # if tensor is sliced into parts, this is the _maximum_ number of values per part

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, key: Key = None, descriptor: TensorDescriptor = None, **kwargs):
        return cls(key, descriptor or TensorDescriptor.from_tensor(tensor), **kwargs)

    def get_part(self, part_index: int, part_size: Optional[int]):
        return CompressionInfo(self.key, self.descriptor, self.role, part_index=part_index, part_size=part_size)


class CompressionBase(ABC):
    """A base class that applies compression algorithm to a pytorch tensor"""

    compression_type: runtime_pb2.CompressionType

    @abstractmethod
    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        """
        Applies compression algorithm to a tensor based on their meta-parameters

        :param tensor: a pytorch tensor to compress; depending on the application, it is a full tensor or a part
        :param info: meta-information about the tensor; if partitioning is used, this still describes the full tensor
        :param allow_inplace: if True, compression can (but doesn't have to) to modify tensor in-place for efficiency
        :returns: a protobuf message that encodes the tensor
        """
        ...

    @abstractmethod
    def extract(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        """Create a pytorch tensor from the serialized outputs of .compress"""
        ...

    @abstractmethod
    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        """Estimate the compression ratio without doing the actual compression; lower ratio = better compression"""
        ...

    def __repr__(self):
        return f"hivemind.{self.__class__.__name__}()"


class NoCompression(CompressionBase):
    """A dummy compression strategy that preserves the original tensor as is."""

    compression_type = runtime_pb2.CompressionType.NONE

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        tensor = tensor.detach()
        shape = tensor.shape
        dtype_name = str(tensor.dtype).lstrip("torch.")
        raw_data = tensor
        if tensor.dtype == torch.bfloat16:
            if USE_LEGACY_BFLOAT16:
                raw_data = tensor.to(torch.float32)
            else:
                typed_storage = tensor.storage()
                storage = typed_storage.untyped() if hasattr(typed_storage, "untyped") else typed_storage._untyped()
                raw_data = torch.tensor(storage, dtype=torch.int8)

        return runtime_pb2.Tensor(
            compression=self.compression_type,
            buffer=raw_data.numpy().tobytes(),
            size=shape,
            dtype=dtype_name,
            requires_grad=tensor.requires_grad,
        )

    def extract(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        shape = torch.Size(serialized_tensor.size)
        if serialized_tensor.dtype == "bfloat16":
            if len(serialized_tensor.buffer) // shape.numel() == 4:  # legacy mode: convert to fp32
                array = np.frombuffer(serialized_tensor.buffer, dtype=np.float32)
                tensor = torch.as_tensor(array, dtype=torch.bfloat16)
            else:  # efficient mode: send bfloat16 data directly
                storage_type = torch.TypedStorage if hasattr(torch, "TypedStorage") else torch._TypedStorage
                storage = storage_type.from_buffer(serialized_tensor.buffer, byte_order="little", dtype=torch.bfloat16)
                tensor = torch.as_tensor(storage, dtype=torch.bfloat16)
        else:
            array = np.frombuffer(serialized_tensor.buffer, dtype=np.dtype(serialized_tensor.dtype))
            tensor = torch.as_tensor(array)
        return tensor.reshape(shape)

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        return 1.0
