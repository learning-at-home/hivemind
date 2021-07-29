import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from hivemind.proto import runtime_pb2
from hivemind.proto.runtime_pb2 import CompressionType

FP32_EPS = 1e-06
NUM_BYTES_FLOAT32 = 4
NUM_BYTES_FLOAT16 = 2
NUM_BITS_QUANTILE_COMPRESSION = 8
NUM_COMPRESSION_QUANTILES = 2 ** NUM_BITS_QUANTILE_COMPRESSION
UNIFORM_BUCKETS_STD_RANGE = 6
FP16_MAX = 65_504
UINT8_RANGE = 256

COMPRESSION_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("QUANTILE_COMPRESSION_THREADS", 128)))

warnings.filterwarnings("ignore", message="The given NumPy array is not writeable", category=UserWarning)


def _quantile_encode_approx(tensor: torch.Tensor, n_bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n_bins = 2 ** n_bits
    borders = torch.as_tensor(_quantile_qq_approximation(tensor.numpy(), n_bins + 1)[1:-1])
    quant_weight = torch.clamp_(torch.bucketize(tensor, borders), 0, n_bins - 1)
    lookup = average_buckets(tensor, quant_weight, n_bins)
    return quant_weight, lookup


def average_buckets(tensor: torch.Tensor, quant_weight: torch.Tensor, n_bins: int):
    bin_sums = torch.zeros(n_bins).scatter_add_(0, quant_weight.flatten().long(), tensor.flatten())
    bin_counts = torch.clamp_min_(torch.bincount(quant_weight.flatten(), minlength=n_bins), 1)
    lookup = bin_sums / bin_counts
    return lookup


def _quantile_qq_approximation(array: np.array, n_quantiles: int, min_chunk_size: int = 10 ** 5) -> np.ndarray:
    """Estimate uniform quantiles of data using quantile-of-quantiles. Runs in parallel."""
    if not array.data.c_contiguous and array.data.f_contiguous:
        array = array.T
    array = np.ascontiguousarray(array.reshape(-1))
    quantiles = np.linspace(0.0, 1.0, num=n_quantiles, dtype=array.dtype)
    chunk_size = _get_chunk_size(len(array), min_chunk_size)
    num_chunks = (len(array) - 1) // chunk_size + 1
    partition_quantiles = np.empty((num_chunks, len(quantiles)), dtype=array.dtype)

    jobs = []
    for i in range(num_chunks):
        chunk = slice(chunk_size * i, chunk_size * (i + 1))
        jobs.append(COMPRESSION_EXECUTOR.submit(np.quantile, array[chunk], quantiles, out=partition_quantiles[i]))

    for job in jobs:
        job.result()
    return np.quantile(partition_quantiles, quantiles)


def _get_chunk_size(num_elements: int, min_chunk_size: int) -> int:
    """Adjust chunk_size to minimize imbalance between chunk sizes"""
    if min_chunk_size >= num_elements:
        return min_chunk_size
    leftover_elements = num_elements % min_chunk_size
    num_chunks = num_elements // min_chunk_size
    return min_chunk_size + (leftover_elements - 1) // num_chunks + 1


def _uint8_uniform_buckets_encode(tensor: torch.Tensor, range_in_sigmas: float):
    offset = UINT8_RANGE // 2
    shift = tensor.mean()
    scale = range_in_sigmas * tensor.std() / UINT8_RANGE

    quant_weight = torch.quantize_per_tensor(tensor - shift, scale, offset, torch.quint8).int_repr()
    lookup = average_buckets(tensor, quant_weight, UINT8_RANGE)
    return quant_weight, lookup


def serialize_torch_tensor(
    tensor: torch.Tensor, compression_type=CompressionType.NONE, allow_inplace=False
) -> runtime_pb2.Tensor:
    assert tensor.device == torch.device("cpu")
    if compression_type == CompressionType.MEANSTD_16BIT:
        assert tensor.dtype == torch.float32

        tensor = tensor if allow_inplace else tensor.clone()
        means = torch.mean(tensor, dim=-1, keepdim=True)
        tensor.sub_(means)

        stds = torch.square(tensor).sum(dim=-1, keepdim=True).div_(tensor.shape[-1]).sqrt_()
        stds.clamp_min_(FP32_EPS)
        tensor.div_(stds)
        tensor = tensor.clamp_(-FP16_MAX, FP16_MAX).to(torch.float16)

        data = b"".join((tensor.numpy().tobytes(), means.numpy().tobytes(), stds.numpy().tobytes()))

        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=data,
            size=tensor.shape,
            dtype="compressed_float32",
            requires_grad=tensor.requires_grad,
        )
    elif compression_type == CompressionType.FLOAT16:
        assert tensor.dtype == torch.float32

        tensor = tensor if allow_inplace else tensor.clone()
        tensor = tensor.clamp_(-FP16_MAX, FP16_MAX).to(torch.float16)

        data = tensor.numpy().tobytes()

        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=data,
            size=tensor.shape,
            dtype="clamped_float32",
            requires_grad=tensor.requires_grad,
        )
    elif compression_type == CompressionType.NONE:
        array = tensor.numpy()
        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=array.tobytes(),
            size=array.shape,
            dtype=array.dtype.name,
            requires_grad=tensor.requires_grad,
        )
    elif compression_type in (CompressionType.QUANTILE_8BIT, CompressionType.UNIFORM_8BIT):
        assert tensor.dtype == torch.float32

        if compression_type == CompressionType.QUANTILE_8BIT:
            quantized, lookup = _quantile_encode_approx(tensor.detach(), NUM_BITS_QUANTILE_COMPRESSION)
        elif compression_type == CompressionType.UNIFORM_8BIT:
            quantized, lookup = _uint8_uniform_buckets_encode(tensor.detach(), UNIFORM_BUCKETS_STD_RANGE)
        data = b"".join((lookup.numpy().tobytes(), quantized.numpy().astype(np.uint8).tobytes()))

        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=data,
            size=tensor.shape,
            dtype="compressed_float32",
            requires_grad=tensor.requires_grad,
        )
    else:
        raise ValueError(f"Unknown compression type: {compression_type}")

    return proto


def construct_torch_tensor(array: np.ndarray, size: Sequence, dtype: Optional[torch.dtype] = None):
    """Helper conversion function that handles edge case with scalar deserialization"""
    if size:
        return torch.as_tensor(array, dtype=dtype).view(*size)
    else:
        return torch.as_tensor(array, dtype=dtype)


def deserialize_torch_tensor(serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
    if serialized_tensor.compression == CompressionType.NONE:
        array = np.frombuffer(serialized_tensor.buffer, dtype=np.dtype(serialized_tensor.dtype))
        tensor = construct_torch_tensor(array, serialized_tensor.size)

    elif serialized_tensor.compression == CompressionType.MEANSTD_16BIT:
        stats_size = list(serialized_tensor.size)
        stats_size[-1] = 1
        stats_count = np.prod(stats_size)

        means = serialized_tensor.buffer[-2 * NUM_BYTES_FLOAT32 * stats_count : -NUM_BYTES_FLOAT32 * stats_count]
        stds = serialized_tensor.buffer[-NUM_BYTES_FLOAT32 * stats_count :]
        means = construct_torch_tensor(np.frombuffer(means, dtype=np.float32), stats_size)
        stds = construct_torch_tensor(np.frombuffer(stds, dtype=np.float32), stats_size)

        array = np.frombuffer(serialized_tensor.buffer[: -8 * stats_count], dtype=np.float16)
        tensor = construct_torch_tensor(array, serialized_tensor.size, torch.float32).mul_(stds).add_(means)

    elif serialized_tensor.compression == CompressionType.FLOAT16:
        array = np.frombuffer(serialized_tensor.buffer, dtype=np.float16)
        tensor = construct_torch_tensor(array, serialized_tensor.size, torch.float32)

    elif serialized_tensor.compression in (CompressionType.QUANTILE_8BIT, CompressionType.UNIFORM_8BIT):
        if serialized_tensor.compression == CompressionType.QUANTILE_8BIT:
            lookup_size = NUM_COMPRESSION_QUANTILES * NUM_BYTES_FLOAT32
        else:
            lookup_size = UINT8_RANGE * NUM_BYTES_FLOAT32
        lookup = serialized_tensor.buffer[:lookup_size]
        quantized = serialized_tensor.buffer[lookup_size:]
        lookup = torch.as_tensor(np.frombuffer(lookup, dtype=np.float32))
        quantized = np.frombuffer(quantized, dtype=np.uint8)
        quantized = construct_torch_tensor(quantized, serialized_tensor.size, dtype=torch.int64)
        tensor = lookup[quantized]

    else:
        raise ValueError(f"Unknown compression type: {serialized_tensor.compression}")

    tensor.requires_grad_(serialized_tensor.requires_grad)
    return tensor


def get_nbytes_per_value(dtype: torch.dtype, compression: CompressionType) -> int:
    """returns the number of bytes per value for a given tensor (excluding metadata)"""
    if compression in (CompressionType.QUANTILE_8BIT, CompressionType.UNIFORM_8BIT):
        return 1
    elif compression in (CompressionType.FLOAT16, CompressionType.MEANSTD_16BIT):
        return 2
    elif compression == CompressionType.NONE:
        return torch.finfo(dtype).bits // 8
    else:
        raise NotImplementedError(f"Unknown compression type: {CompressionType.Name(compression)}")
