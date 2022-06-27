import importlib.util
import math
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np
import torch

if importlib.util.find_spec("bitsandbytes") is not None:
    from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise

from hivemind.compression.base import CompressionBase, CompressionInfo
from hivemind.proto import runtime_pb2

EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("QUANTIZATION_THREADS", 128)))


class Quantization(CompressionBase, ABC):
    codebook_dtype, indices_dtype = np.float32, np.uint8

    @abstractmethod
    def quantize(self, tensor: torch.Tensor, allow_inplace: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Convert tensor into a pair of (indices, codebook)"""
        ...

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        quantized, codebook = self.quantize(tensor.detach(), allow_inplace=allow_inplace)
        return runtime_pb2.Tensor(
            compression=self.compression_type,
            buffer=b"".join((np.int64(len(codebook)).tobytes(), codebook.tobytes(), quantized.tobytes())),
            size=tensor.shape,
            dtype=tensor.numpy().dtype.name,
            requires_grad=tensor.requires_grad,
        )

    def extract(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        codebook_size = int(np.frombuffer(serialized_tensor.buffer, count=1, dtype=np.int64))
        codebook = np.frombuffer(serialized_tensor.buffer, offset=8, count=codebook_size, dtype=self.codebook_dtype)
        quantized = np.frombuffer(serialized_tensor.buffer, offset=8 + codebook.nbytes, dtype=self.indices_dtype)
        quantized = torch.as_tensor(quantized, dtype=torch.int64).reshape(tuple(serialized_tensor.size))
        codebook = torch.as_tensor(np.asarray(codebook, dtype=serialized_tensor.dtype))
        return codebook[quantized]

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        return self.n_bits / torch.finfo(info.descriptor.dtype).bits

    @property
    def n_bits(self):
        return self.indices_dtype(1).itemsize * 8

    @property
    def n_bins(self):
        return 2**self.n_bits


class Uniform8BitQuantization(Quantization):
    RANGE_IN_SIGMAS: int = 6
    compression_type = runtime_pb2.UNIFORM_8BIT

    def quantize(self, tensor: torch.Tensor, allow_inplace: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        offset = self.n_bins // 2
        shift = tensor.mean()
        centered_tensor = tensor.sub_(shift) if allow_inplace else tensor - shift
        std_unbiased = centered_tensor.norm() / math.sqrt(centered_tensor.numel() - 1)
        scale = self.RANGE_IN_SIGMAS * std_unbiased / self.n_bins
        quantized = torch.quantize_per_tensor(centered_tensor, scale, offset, torch.quint8).int_repr()
        lookup = average_buckets(tensor, quantized, self.n_bins)
        return np.asarray(quantized, dtype=self.indices_dtype), np.asarray(lookup, dtype=self.codebook_dtype)


class Quantile8BitQuantization(Quantization):
    compression_type = runtime_pb2.QUANTILE_8BIT

    def quantize(self, tensor: torch.Tensor, allow_inplace: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        tensor = tensor.detach().float()
        borders = torch.as_tensor(quantile_qq_approximation(tensor.numpy(), self.n_bins + 1)[1:-1])
        quantized = torch.clamp_(torch.bucketize(tensor, borders), 0, self.n_bins - 1)
        codebook = average_buckets(tensor, quantized, self.n_bins)
        return quantized.numpy().astype(np.uint8), codebook.numpy()


def average_buckets(tensor: torch.Tensor, quant_weight: torch.Tensor, n_bins: int):
    """Return the average value in each bucket"""
    bin_sums = torch.zeros(n_bins).scatter_add_(0, quant_weight.flatten().long(), tensor.flatten())
    bin_counts = torch.clamp_min_(torch.bincount(quant_weight.flatten(), minlength=n_bins), 1)
    lookup = bin_sums / bin_counts
    return lookup


def get_chunk_size(num_elements: int, min_chunk_size: int) -> int:
    """Adjust chunk_size to minimize imbalance between chunk sizes"""
    if min_chunk_size >= num_elements:
        return min_chunk_size
    leftover_elements = num_elements % min_chunk_size
    num_chunks = num_elements // min_chunk_size
    return min_chunk_size + (leftover_elements - 1) // num_chunks + 1


def quantile_qq_approximation(array: np.ndarray, n_quantiles: int, min_chunk_size: int = 10**5) -> np.ndarray:
    """Estimate uniform quantiles of data using quantile-of-quantiles. Runs in parallel."""
    if not array.data.c_contiguous and array.data.f_contiguous:
        array = array.T
    array = np.ascontiguousarray(array.reshape(-1))
    quantiles = np.linspace(0.0, 1.0, num=n_quantiles, dtype=array.dtype)
    chunk_size = get_chunk_size(len(array), min_chunk_size)
    num_chunks = (len(array) - 1) // chunk_size + 1
    partition_quantiles = np.empty((num_chunks, len(quantiles)), dtype=array.dtype)

    jobs = []
    for i in range(num_chunks):
        chunk = slice(chunk_size * i, chunk_size * (i + 1))
        jobs.append(EXECUTOR.submit(np.quantile, array[chunk], quantiles, out=partition_quantiles[i]))

    for job in jobs:
        job.result()
    return np.quantile(partition_quantiles, quantiles)


class BlockwiseQuantization(Quantization):
    compression_type = runtime_pb2.BLOCKWISE_8BIT
    codebook_dtype, indices_dtype = np.float32, np.uint8

    def quantize(
        self, tensor: torch.Tensor, allow_inplace: bool = False
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        try:
            quantized, (absmax, codebook) = quantize_blockwise(tensor)
        except NameError:
            raise ImportError(
                "BlockwiseQuantization requires bitsandbytes to function. "
                "Please install it using the following guide: "
                "https://github.com/facebookresearch/bitsandbytes#requirements--installation"
            )
        return quantized.numpy(), (absmax.numpy(), codebook.numpy())

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        quantized, (absmax, codebook) = self.quantize(tensor.detach(), allow_inplace=allow_inplace)

        serialized_data = (
            np.int64(len(absmax)).tobytes(),
            np.int64(len(codebook)).tobytes(),
            absmax.tobytes(),
            codebook.tobytes(),
            quantized.tobytes(),
        )

        return runtime_pb2.Tensor(
            buffer=b"".join(serialized_data),
            size=tensor.shape,
            requires_grad=tensor.requires_grad,
            dtype=tensor.numpy().dtype.name,
            compression=self.compression_type,
        )

    def extract(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        absmax_size = int(np.frombuffer(serialized_tensor.buffer, count=1, dtype=np.int64))
        codebook_size = int(np.frombuffer(serialized_tensor.buffer, offset=8, count=1, dtype=np.int64))
        absmax = np.frombuffer(serialized_tensor.buffer, offset=16, count=absmax_size, dtype=self.codebook_dtype)
        codebook = np.frombuffer(
            serialized_tensor.buffer, offset=16 + absmax.nbytes, count=codebook_size, dtype=self.codebook_dtype
        )
        quantized = np.frombuffer(
            serialized_tensor.buffer, offset=16 + absmax.nbytes + codebook.nbytes, dtype=self.indices_dtype
        )

        absmax = torch.as_tensor(absmax)
        codebook = torch.as_tensor(codebook)
        quantized = torch.as_tensor(quantized).reshape(tuple(serialized_tensor.size))
        try:
            return dequantize_blockwise(quantized, (absmax, codebook))
        except NameError:
            raise ImportError(
                "BlockwiseQuantization requires bitsandbytes to function. "
                "Please install it using the following guide: "
                "https://github.com/facebookresearch/bitsandbytes#requirements--installation"
            )
