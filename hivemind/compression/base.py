import dataclasses
from typing import Union, Optional

import torch
from enum import Enum, auto

from hivemind.utils.tensor_descr import TensorDescriptor
from hivemind.proto import runtime_pb2


class TensorRole(Enum):
    ACTIVATION = auto()
    PARAMETER = auto()
    GRADIENT = auto()
    OPTIMIZER = auto()
    UNSPECIFIED = auto()


@dataclasses.dataclass(frozen=True)
class CompressionInfo:
    """Auxiliary data structure that contains information about the tensor that determines how it is compressed"""
    key: Union[int, str]  # name or index of the tensor from named parameters, optimizer state dict or i/o structure
    descriptor: TensorDescriptor  # data structure that defines shape, dtype, layout and device information
    role: TensorRole = TensorRole.UNSPECIFIED  # which role does the tensor play with respect to the model
    chunk_index: int = 0  # if tensor is sliced into chunks, this represents the index within one tensor
    chunk_size: Optional[int] = None  # if tensor is sliced into chunks, this is the number of values per chunk

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, descriptor: Optional[TensorDescriptor] = None, **kwargs):
        return cls(tensor, descriptor or TensorDescriptor.from_tensor(tensor), **kwargs)


class Compression:
    """ A base class that applies compression algorithm to a pytorch tensor """
    compression_type: runtime_pb2.CompressionType

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        """
        Applies compression algorithm to a tensor based on their meta-parameters

        :param tensor: a pytorch tensor to compress; depending on the applicaiton, x is either a full tensor or a chunk
        :param info: meta-information about the tensor; if chunking is used, this must still describe the full tensor
        :param allow_inplace: if True, compression can (but doesn't have to) to modify tensor in-place for efficiency
        :returns: a protobuf message that encodes the tensor
        """
        raise NotImplementedError()

    def restore(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        """ Create a pytorch tensor from the serialized outputs of .compress """
        raise NotImplementedError()

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        """Estimate the compression ratio without doing the actual compression; lower ratio = better compression"""
        raise NotImplementedError()
