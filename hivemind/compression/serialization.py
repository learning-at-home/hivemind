from __future__ import annotations

from typing import AsyncIterator, Dict, Iterable, List, Optional

import torch

from hivemind.compression.base import CompressionBase, CompressionInfo, NoCompression
from hivemind.compression.floating import Float16Compression, ScaledFloat16Compression
from hivemind.compression.quantization import BlockwiseQuantization, Quantile8BitQuantization, Uniform8BitQuantization
from hivemind.proto import runtime_pb2
from hivemind.utils.streaming import combine_from_streaming

_BASE_COMPRESSION_TYPES: Dict[str, CompressionBase] = dict(
    NONE=NoCompression(),
    FLOAT16=Float16Compression(),
    MEANSTD_16BIT=ScaledFloat16Compression(),
    QUANTILE_8BIT=Quantile8BitQuantization(),
    UNIFORM_8BIT=Uniform8BitQuantization(),
    BLOCKWISE_8BIT=BlockwiseQuantization(),
)

for key in runtime_pb2.CompressionType.keys():
    assert key in _BASE_COMPRESSION_TYPES, f"Compression type {key} does not have a registered deserializer"
    actual_compression_type = _BASE_COMPRESSION_TYPES[key].compression_type
    assert (
        runtime_pb2.CompressionType.Name(actual_compression_type) == key
    ), f"Compression strategy for {key} has inconsistent type"


def serialize_torch_tensor(
    tensor: torch.Tensor,
    compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
    info: Optional[CompressionInfo] = None,
    allow_inplace: bool = False,
    **kwargs,
) -> runtime_pb2.Tensor:
    """Serialize a given tensor into a protobuf message using the specified compression strategy"""
    assert tensor.device == torch.device("cpu")
    compression = _BASE_COMPRESSION_TYPES[runtime_pb2.CompressionType.Name(compression_type)]
    info = info or CompressionInfo.from_tensor(tensor, **kwargs)
    return compression.compress(tensor, info, allow_inplace)


def deserialize_torch_tensor(serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
    """Restore a pytorch tensor from a protobuf message"""
    compression = _BASE_COMPRESSION_TYPES[runtime_pb2.CompressionType.Name(serialized_tensor.compression)]
    return compression.extract(serialized_tensor).requires_grad_(serialized_tensor.requires_grad)


async def deserialize_tensor_stream(
    stream: AsyncIterator[Iterable[runtime_pb2.Tensor]],
) -> List[torch.Tensor]:
    """Async wrapper of combine_from_streaming that combines tensors from a stream of parts and deserializes them"""

    tensors = []
    tensor_parts = []

    async for parts in stream:
        for part in parts:
            if part.dtype and tensor_parts:
                tensors.append(deserialize_torch_tensor(combine_from_streaming(tensor_parts)))
                tensor_parts = []

            tensor_parts.append(part)
    if tensor_parts:
        tensors.append(deserialize_torch_tensor(combine_from_streaming(tensor_parts)))

    return tensors
