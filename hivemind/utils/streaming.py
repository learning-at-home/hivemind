"""
Utilities for streaming tensors
"""

from __future__ import annotations

from typing import Iterable, Iterator

from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)

STREAMING_CHUNK_SIZE_BYTES = 2**16


def split_for_streaming(
    serialized_tensor: runtime_pb2.Tensor,
    chunk_size_bytes: int = STREAMING_CHUNK_SIZE_BYTES,
) -> Iterator[runtime_pb2.Tensor]:
    """Split serialized_tensor into multiple chunks for streaming"""
    buffer = memoryview(serialized_tensor.buffer)
    num_chunks = len(range(0, len(buffer), chunk_size_bytes))
    yield runtime_pb2.Tensor(
        compression=serialized_tensor.compression,
        buffer=buffer[:chunk_size_bytes].tobytes(),
        chunks=num_chunks,
        size=serialized_tensor.size,
        dtype=serialized_tensor.dtype,
        requires_grad=serialized_tensor.requires_grad,
    )
    for chunk_start in range(chunk_size_bytes, len(buffer), chunk_size_bytes):
        yield runtime_pb2.Tensor(buffer=buffer[chunk_start : chunk_start + chunk_size_bytes].tobytes())


def combine_from_streaming(stream: Iterable[runtime_pb2.Tensor]) -> runtime_pb2.Tensor:
    """Restore a result of split_into_chunks into a single serialized tensor"""
    stream = iter(stream)
    first_chunk = next(stream)
    serialized_tensor = runtime_pb2.Tensor()
    serialized_tensor.CopyFrom(first_chunk)
    buffer_chunks = [first_chunk.buffer]
    for tensor_part in stream:
        buffer_chunks.append(tensor_part.buffer)
    serialized_tensor.buffer = b"".join(buffer_chunks)
    return serialized_tensor
