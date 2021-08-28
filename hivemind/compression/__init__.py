"""
Compression strategies that reduce the network communication in .averaging, .optim and .moe
"""

import warnings
from typing import Optional, Dict, Union

import torch

from hivemind.proto import runtime_pb2
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.compression.base import Compression, CompressionInfo
from hivemind.compression.simple import NoCompression, Float16Compression, ScaledFloat16Compression
from hivemind.compression.quantization import Uniform8BitQuantization, Quantile8BitQuantization


warnings.filterwarnings("ignore", message="The given NumPy array is not writeable", category=UserWarning)


BASE_COMPRESSION_TYPES: Dict[str, Compression] = dict(
    NONE=NoCompression(), FLOAT16=Float16Compression(), MEANSTD_16BIT=ScaledFloat16Compression(),
    QUANTILE_8BIT=Quantile8BitQuantization(), UNIFORM_8BIT=Uniform8BitQuantization()
)

for key in runtime_pb2.CompressionType.keys():
    assert key in BASE_COMPRESSION_TYPES, f"Compression type {key} does not have a registered deserializer."
    assert BASE_COMPRESSION_TYPES[key].compression_type, f"Compression strategy for {key} has inconsistent type"


def serialize_torch_tensor(tensor: torch.Tensor,
                           compression_type: Union[CompressionType, Compression] = CompressionType.NONE,
                           info: Optional[CompressionInfo] = None,
                           allow_inplace: bool = False,
                           **kwargs) -> runtime_pb2.Tensor:
    """Serialize a given tensor into a protobuf message using the specified compression strategy"""
    assert tensor.device == torch.device("cpu")
    if not isinstance(compression_type, Compression):
        compression_type = BASE_COMPRESSION_TYPES[CompressionType.Name(compression_type)]
    info = info or CompressionInfo.from_tensor(tensor, **kwargs)
    return compression_type.compress(tensor, info, allow_inplace)


def deserialize_torch_tensor(serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
    """Restore a pytorch tensor from a protobuf message"""
    compression_type = BASE_COMPRESSION_TYPES[CompressionType.Name(serialized_tensor.compression_type)]
    return compression_type.restore(serialized_tensor).requires_grad_(serialized_tensor.requires_grad)
