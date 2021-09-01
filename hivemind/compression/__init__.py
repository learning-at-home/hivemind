"""
Compression strategies that reduce the network communication in .averaging, .optim and .moe
"""

import warnings
from typing import Dict, Optional

import torch

from hivemind.compression.base import CompressionBase, CompressionInfo, NoCompression
from hivemind.compression.quantization import Quantile8BitQuantization, Uniform8BitQuantization
from hivemind.compression.floating import Float16Compression, ScaledFloat16Compression
from hivemind.proto import runtime_pb2
from hivemind.proto.runtime_pb2 import CompressionType

warnings.filterwarnings("ignore", message="The given NumPy array is not writeable", category=UserWarning)


BASE_COMPRESSION_TYPES: Dict[str, CompressionBase] = dict(
    NONE=NoCompression(),
    FLOAT16=Float16Compression(),
    MEANSTD_16BIT=ScaledFloat16Compression(),
    QUANTILE_8BIT=Quantile8BitQuantization(),
    UNIFORM_8BIT=Uniform8BitQuantization(),
)

for key in CompressionType.keys():
    assert key in BASE_COMPRESSION_TYPES, f"Compression type {key} does not have a registered deserializer."
    actual_compression_type = BASE_COMPRESSION_TYPES[key].compression_type
    assert (
        CompressionType.Name(actual_compression_type) == key
    ), f"Compression strategy for {key} has inconsistent type"


def serialize_torch_tensor(
    tensor: torch.Tensor,
    compression_type: CompressionType = CompressionType.NONE,
    info: Optional[CompressionInfo] = None,
    allow_inplace: bool = False,
    **kwargs,
) -> runtime_pb2.Tensor:
    """Serialize a given tensor into a protobuf message using the specified compression strategy"""
    assert tensor.device == torch.device("cpu")
    compression = BASE_COMPRESSION_TYPES[CompressionType.Name(compression_type)]
    info = info or CompressionInfo.from_tensor(tensor, **kwargs)
    return compression.compress(tensor, info, allow_inplace)


def deserialize_torch_tensor(serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
    """Restore a pytorch tensor from a protobuf message"""
    compression = BASE_COMPRESSION_TYPES[CompressionType.Name(serialized_tensor.compression)]
    return compression.restore(serialized_tensor).requires_grad_(serialized_tensor.requires_grad)
