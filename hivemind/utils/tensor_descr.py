import warnings
from dataclasses import dataclass, asdict

import torch

from hivemind.proto.runtime_pb2 import CompressionType

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
    def shape(self):
        return self.size

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(tensor.shape, tensor.dtype, tensor.layout, tensor.device, tensor.requires_grad, safe_check_pinned(tensor))

    def make_empty(self, **kwargs):
        properties = asdict(self)
        properties.update(kwargs)
        return torch.empty(**properties)


@dataclass(repr=True, frozen=True)
class BatchTensorDescriptor(TensorDescriptor):
    """ torch Tensor with a variable 0-th dimension, used to describe batched data """

    def __init__(self, *instance_size, **kwargs):  # compatibility: allow initializing with *size
        if len(instance_size) == 1 and isinstance(instance_size[0], (list, tuple, torch.Size)):
            instance_size = instance_size[0]  # we were given size as the only parameter instead of *parameters
        super().__init__((None, *instance_size), **kwargs)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, compression=CompressionType.NONE):
        return cls(*tensor.shape[1:], dtype=tensor.dtype, layout=tensor.layout,
                   device=tensor.device, requires_grad=tensor.requires_grad,
                   pin_memory=safe_check_pinned(tensor),
                   compression=compression if tensor.is_floating_point() else CompressionType.NONE)

    def make_empty(self, batch_size, **kwargs):
        assert self.shape[0] is None, "Make sure 0-th dimension is not specified (set to None)"
        return super().make_empty(size=(batch_size, *self.shape[1:]), **kwargs)

    
def safe_check_pinned(tensor: torch.Tensor) -> bool:
    """ Check whether or not a tensor is pinned. If torch cannot initialize cuda, returns False instead of error. """
    try:
        return torch.cuda.is_available() and tensor.is_pinned()
    except RuntimeError:
        return False
