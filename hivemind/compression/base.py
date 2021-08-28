from __future__ import annotations
from typing import Any, Optional

import torch
from enum import Enum, auto

import runtime_pb2
from hivemind import TensorDescriptor


class Compression:
    """ A base class that applies compression algorithm to a pytorch tensor """
    def compress(self, tensor: torch.Tensor, key: Any = None, role: TensorRole = TensorRole.UNSPECIFIED,
                 tensor_index: int = 0, chunk_index: int = 0, tensor_descr: Optional[TensorDescriptor] = None,
                 allow_inplace: bool = False, **kwargs) -> runtime_pb2.Tensor:
        """
        Applies compression algorithm to a tensor based on their meta-parameters

        :param tensor: a pytorch tensor to compress; depending on the applicaiton, x is either a full tensor or a chunk
        :param key: name or index of the tensor from named parameters or optimizer state dict
        :param role: which role does the tensor play in the training process (activation/param/grad/opt/etc)
        :param tensor_index: global tensor index in the compressed structure
        :param chunk_index: if tensor is sliced into chunks, this represents the index within one tensor
        :param tensor_descr: if tensor is sliced into chunks, this will be the description of a full tensor
        :param allow_inplace: if True, the algorithm is allowed (but not required) to modify x in-place for efficiency
        :note: if tensor is not sliced into chunks, chunk_index will be always set to 0
        :returns: a protobuf message that encodes the tensor
        """
        raise NotImplementedError()

    def restore(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        """ Create a pytorch tensor from the serialized outputs of .compress """
        raise NotImplementedError()


class TensorRole(Enum):
    ACTIVATION = auto()
    PARAMETER = auto()
    GRADIENT = auto()
    OPTIMIZER = auto()
    UNSPECIFIED = auto()
