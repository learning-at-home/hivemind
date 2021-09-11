from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import torch

from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.serializer import MSGPackSerializer

DUMMY_BATCH_SIZE = 3  # used for dummy runs only

warnings.filterwarnings("ignore", "CUDA initialization*", category=UserWarning)


# ^-- cures https://github.com/pytorch/pytorch/issues/47038


@dataclass(init=True, repr=True, frozen=True)
class DescriptorBase:
    pass


@dataclass(init=True, repr=True, frozen=True)
class TensorDescriptor(DescriptorBase):
    size: tuple
    dtype: torch.dtype = None
    layout: torch.layout = torch.strided
    device: torch.device = None
    requires_grad: bool = False
    pin_memory: bool = False
    compression: CompressionType = CompressionType.NONE

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.size

    def numel(self) -> int:
        return int(np.prod(self.size))

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorDescriptor:
        return cls(
            tensor.shape, tensor.dtype, tensor.layout, tensor.device, tensor.requires_grad, _safe_check_pinned(tensor)
        )

    def make_empty(self, **kwargs):
        properties = asdict(self)
        properties.update(kwargs)
        properties.pop("compression")
        return torch.empty(**properties)


def _str_to_torch_type(name: str, torch_type: type):
    try:
        value = getattr(torch, name.split(".")[-1])
    except AttributeError:
        raise ValueError("Invalid dtype")
    if not isinstance(value, torch_type):
        raise ValueError(f"Invalid dtype: expected {torch_type}, got: {type(value)}")

    return value


@MSGPackSerializer.ext_serializable(0x51)
@dataclass(repr=True, frozen=True)
class BatchTensorDescriptor(TensorDescriptor):
    """torch.Tensor with a variable 0-th dimension, used to describe batched data"""

    def __init__(self, *instance_size, **kwargs):  # compatibility: allow initializing with *size
        if len(instance_size) == 1 and isinstance(instance_size[0], (list, tuple, torch.Size)):
            instance_size = instance_size[0]  # we were given size as the only parameter instead of *parameters
        super().__init__((None, *instance_size), **kwargs)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, compression=CompressionType.NONE) -> BatchTensorDescriptor:
        return cls(
            *tensor.shape[1:],
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
            pin_memory=_safe_check_pinned(tensor),
            compression=compression if tensor.is_floating_point() else CompressionType.NONE,
        )

    def make_empty(self, *batch_size: int, **kwargs) -> torch.Tensor:
        assert self.shape[0] is None, "Make sure 0-th dimension is not specified (set to None)"
        return super().make_empty(size=(*batch_size, *self.shape[1:]), **kwargs)

    def packb(self) -> bytes:
        obj_dict = asdict(self)
        obj_dict.pop("device")

        obj_dict.update(
            dtype=str(self.dtype),
            layout=str(self.layout),
            device_type=self.device.type,
            device_index=self.device.index,
        )

        return MSGPackSerializer.dumps(obj_dict)

    @classmethod
    def unpackb(cls, raw: bytes) -> BatchTensorDescriptor:
        obj_dict = MSGPackSerializer.loads(raw)

        device = torch.device(obj_dict.pop("device_type"), obj_dict.pop("device_index"))

        obj_dict.update(
            dtype=_str_to_torch_type(obj_dict["dtype"], torch.dtype),
            layout=_str_to_torch_type(obj_dict["layout"], torch.layout),
            device=device,
        )

        size = obj_dict.pop("size")[1:]

        return BatchTensorDescriptor(*size, **obj_dict)


def _safe_check_pinned(tensor: torch.Tensor) -> bool:
    """Check whether or not a tensor is pinned. If torch cannot initialize cuda, returns False instead of error."""
    try:
        return torch.cuda.is_available() and tensor.is_pinned()
    except RuntimeError:
        return False
