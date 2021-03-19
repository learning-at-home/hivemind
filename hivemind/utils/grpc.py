"""
Utilities for running GRPC services: compile protobuf, patch legacy versions, etc
"""

SIZE_OF_FLOAT32 = 4
SIZE_OF_FLOAT16 = 2
NUM_BITS_QUANTILE_COMPRESSION = 8
NUM_QUANTILES_QUANTILE_COMPRESSION = 2 ** NUM_BITS_QUANTILE_COMPRESSION

from __future__ import annotations

import os
import threading
from typing import NamedTuple, Tuple, Optional, Union, Any, Dict, TypeVar, Type, Iterator, Iterable

import grpc
import numpy as np
import torch

from hivemind import run_in_background
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger
from hivemind.utils.networking import Endpoint
from hivemind.utils.timed_storage import TimedStorage, get_dht_time, ValueWithExpiration

logger = get_logger(__name__)

Stub = TypeVar("Stub")

GRPC_KEEPALIVE_OPTIONS = (
    ('grpc.keepalive_time_ms', 60 * 1000),
    ('grpc.keepalive_timeout_ms', 60 * 1000),
    ('grpc.keepalive_permit_without_calls', True),
    ('grpc.http2.max_pings_without_data', 0),
    ('grpc.http2.min_time_between_pings_ms', 30 * 1000),
    ('grpc.http2.min_ping_interval_without_data_ms', 10 * 1000),
)


class ChannelInfo(NamedTuple):
    target: Endpoint
    aio: bool
    options: Tuple[Tuple[str, str], ...]
    credentials: Optional[grpc.ChannelCredentials]
    compression: Optional[grpc.Compression]


class ChannelCache(TimedStorage[ChannelInfo, Tuple[Union[grpc.Channel, grpc.aio.Channel], Dict]]):
    """
    A process-wide cache of gRPC channels, supports both normal and aio channels, secure/insecure channels, etc
    Based on grpcio internal channel cache by Richard Belleville and Lidi Zheng (thanks!)
    Unlike TimedStorage, ChannelCache actively evicts stale channels even if the cache is not accessed
    Unlike grpc._simple_stubs.ChannelCache, this implementation supports aio and does not forcibly close active channels
    """
    MAXIMUM_CHANNELS = int(os.environ.get("GRPC_PYTHON_MANAGED_CHANNEL_MAXIMUM", 4096))
    EVICTION_PERIOD_SECONDS = float(os.environ.get("GRPC_PYTHON_MANAGED_CHANNEL_EVICTION_SECONDS", 10 * 60))
    logger.debug(f"Eviction period = {EVICTION_PERIOD_SECONDS}s, max channels = {MAXIMUM_CHANNELS}")

    _singleton: Optional[ChannelCache] = None
    _singleton_pid: int = os.getpid()
    _lock: threading.RLock = threading.RLock()
    _update_eviction_evt: threading.Event = threading.Event()

    def __init__(self, _created_as_singleton=False):
        assert _created_as_singleton, f"Please use {self.__class__.__name__}.get_singleton()"
        super().__init__(maxsize=self.MAXIMUM_CHANNELS)
        self._is_active = True
        self._nearest_expiration_time = float('inf')
        self._eviction_thread = threading.Thread(target=self._evict_stale_channels_in_background, daemon=True)
        self._eviction_thread.start()

    @classmethod
    def get_singleton(cls):
        """ Get or create the channel cache for the current process """
        with cls._lock:
            if cls._singleton is None or cls._singleton_pid != os.getpid():
                if cls._singleton is not None:
                    cls._singleton._stop_background_thread()
                cls._singleton, cls._singleton_pid = cls(_created_as_singleton=True), os.getpid()
            return cls._singleton

    @classmethod
    def get_stub(cls, target: Endpoint, stub_type: Type[Stub], *, aio: bool, options: Tuple[Tuple[str, Any]] = (),
                 channel_credentials: Optional[grpc.ChannelCredentials] = None,
                 compression: Optional[grpc.Compression] = None) -> Stub:
        """
        Create a grpc channel with given options or reuse pre-existing one

        :param target: the recipient's address and port
        :param stub_type: a gRPC stub (client) to be instantiated
        :param aio: if True, returns grpc.Channel, otherwise returns grpc.aio.Channel
        :param options: see https://grpc.github.io/grpc/core/group__grpc__arg__keys.html
        :param channel_credentials: if specified, create a secure channel usin these credentials (default = insecure)
        :param compression: see https://github.com/grpc/grpc/tree/master/examples/python/compression
        """
        cache = cls.get_singleton()
        with cls._lock:
            key = ChannelInfo(target, aio, tuple(options), channel_credentials, compression)
            entry: ValueWithExpiration = super(cls, cache).get(key)

            if entry is not None:
                channel, stubs = entry.value
            else:
                channel = cls._create_channel(*key)
                stubs = {}

            channel._channel.check_connectivity_state(True)

            if stub_type not in stubs:
                stubs[stub_type] = stub_type(channel)

            # either cache channel or update expiration of an existing channel
            expiration_time = get_dht_time() + cls.EVICTION_PERIOD_SECONDS
            super(cls, cache).store(key, (channel, stubs), expiration_time)

            if expiration_time < cache._nearest_expiration_time:
                cache._nearest_expiration_time = expiration_time
                cls._update_eviction_evt.set()

            return stubs[stub_type]

    @classmethod
    def _create_channel(cls, target: Endpoint, aio: bool, extra_options: Tuple[Tuple[str, Any], ...],
                        channel_credentials: Optional[grpc.ChannelCredentials],
                        compression: Optional[grpc.Compression]) -> Union[grpc.Channel, grpc.aio.Channel]:
        namespace = grpc.aio if aio else grpc

        options = extra_options + GRPC_KEEPALIVE_OPTIONS

        if channel_credentials is None:
            logger.debug(f"Creating insecure {namespace} channel with options '{options}' "
                         f"and compression '{compression}'")
            return namespace.insecure_channel(target, options=options, compression=compression)
        else:
            logger.debug(f"Creating secure {namespace} channel with credentials '{channel_credentials}', "
                         f"options '{options}' and compression '{compression}'")
            return namespace.secure_channel(target, credentials=channel_credentials,
                                            options=options, compression=compression)

    def _evict_stale_channels_in_background(self):
        while self._is_active:
            now = get_dht_time()
            time_to_wait = max(0.0, self._nearest_expiration_time - now)
            interrupted_early = self._update_eviction_evt.wait(time_to_wait if time_to_wait != float('inf') else None)
            if interrupted_early:
                self._update_eviction_evt.clear()
                continue

            with self._lock:
                self._remove_outdated()
                _, entry = super().top()
                self._nearest_expiration_time = entry.expiration_time if entry is not None else float('inf')

    def _stop_background_thread(self):
        with self._lock:
            self._is_active = False
            self._update_eviction_evt.set()

    def store(self, *args, **kwargs) -> ValueError:
        raise ValueError(f"Please use {self.__class__.__name__}.get_stub to get or create stubs")

    def get(self, *args, **kwargs) -> ValueError:
        raise ValueError(f"Please use {self.__class__.__name__}.get_stub to get or create stubs")

    def top(self) -> ValueError:
        raise ValueError(f"Please use {self.__class__.__name__}.get_stub to get or create stubs")


def quantile_encode_torch_approx(tensor: torch.Tensor, n_bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n_bins = 2 ** n_bits
    borders = torch.as_tensor(quantile_qq_approximation(tensor.numpy(), n_bins + 1)[:-1])
    quant_weight = torch.clamp_(torch.bucketize(tensor, borders) - 1, 0, n_bins - 1)
    bin_sums = torch.zeros(n_bins).scatter_add_(0, quant_weight.flatten(), tensor.flatten())
    bin_counts = torch.clamp_min_(torch.bincount(quant_weight.flatten(), minlength=n_bins), 1)
    lookup = bin_sums / bin_counts
    return quant_weight, lookup


def quantile_qq_approximation(array: np.array, n_quantiles: int, min_chunk_size: int = 10 ** 5, **kwargs) -> np.ndarray:
    """ Estimate uniform quantiles of data using quantile-of-quantiles. Runs in parallel. """
    if not array.data.c_contiguous and array.data.f_contiguous:
        array = array.T
    array = np.ascontiguousarray(array.reshape(-1))
    quantiles = np.linspace(0., 1., num=n_quantiles, dtype=array.dtype)
    chunk_size = get_chunk_size(len(array), min_chunk_size)
    num_chunks = (len(array) - 1) // chunk_size + 1
    partition_quantiles = np.empty((num_chunks, len(quantiles)), dtype=array.dtype)

    jobs = []
    for i in range(num_chunks):
        chunk = slice(chunk_size * i, chunk_size * (i + 1))
        jobs.append(run_in_background(
            np.quantile, array[chunk], quantiles, out=partition_quantiles[i], **kwargs))

    for job in jobs:
        job.result()
    return np.quantile(partition_quantiles, quantiles, **kwargs)


def get_chunk_size(num_elements: int, min_chunk_size: int) -> int:
    """ Adjust chunk_size to minimize imbalance between chunk sizes """
    if min_chunk_size >= num_elements:
        return min_chunk_size
    leftover_elements = num_elements % min_chunk_size
    num_chunks = num_elements // min_chunk_size
    return min_chunk_size + (leftover_elements - 1) // num_chunks + 1


FP16_MAX = 65_504


def serialize_torch_tensor(tensor: torch.Tensor, compression_type=CompressionType.NONE,
                           allow_inplace=False) -> runtime_pb2.Tensor:
    assert tensor.device == torch.device('cpu')
    if compression_type == CompressionType.MEANSTD_LAST_AXIS_FLOAT16:
        assert tensor.dtype == torch.float32

        tensor = tensor if allow_inplace else tensor.clone()
        means = torch.mean(tensor, dim=-1, keepdim=True)
        tensor.sub_(means)

        stds = torch.square(tensor).sum(dim=-1, keepdim=True).div_(tensor.shape[-1]).sqrt_()
        tensor.div_(stds)
        tensor = tensor.clamp_(-FP16_MAX, FP16_MAX).to(torch.float16)

        data = b''.join((tensor.numpy().tobytes(), means.numpy().tobytes(), stds.numpy().tobytes()))

        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=data,
            size=tensor.shape,
            dtype='compressed_float32',
            requires_grad=tensor.requires_grad)
    elif compression_type == CompressionType.FLOAT16:
        assert tensor.dtype == torch.float32

        tensor = tensor if allow_inplace else tensor.clone()
        tensor = tensor.clamp_(-FP16_MAX, FP16_MAX).to(torch.float16)

        data = tensor.numpy().tobytes()

        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=data,
            size=tensor.shape,
            dtype='clamped_float32',
            requires_grad=tensor.requires_grad)
    elif compression_type == CompressionType.NONE:
        array = tensor.numpy()
        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=array.tobytes(),
            size=array.shape,
            dtype=array.dtype.name,
            requires_grad=tensor.requires_grad)
    elif compression_type == CompressionType.QUANTILE_8BIT:
        assert tensor.dtype == torch.float32

        quantized, lookup = quantile_encode_torch_approx(tensor.detach(), NUM_BITS_QUANTILE_COMPRESSION)
        data = b''.join((lookup.numpy().tobytes(), quantized.numpy().astype(np.uint8).tobytes()))

        proto = runtime_pb2.Tensor(
            compression=compression_type,
            buffer=data,
            size=tensor.shape,
            dtype='compressed_float32',
            requires_grad=tensor.requires_grad)
    else:
        raise ValueError(f"Unknown compression type: {compression_type}")

    return proto


def deserialize_torch_tensor(serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
    # TODO avoid copying the array (need to silence pytorch warning, because array is not writable)
    if serialized_tensor.compression == CompressionType.NONE:
        array = np.frombuffer(serialized_tensor.buffer, dtype=np.dtype(serialized_tensor.dtype)).copy()
        tensor = torch.as_tensor(array).view(*serialized_tensor.size)
    elif serialized_tensor.compression == CompressionType.MEANSTD_LAST_AXIS_FLOAT16:
        stats_size = list(serialized_tensor.size)
        stats_size[-1] = 1
        stats_count = np.prod(stats_size)
        means = serialized_tensor.buffer[-2 * SIZE_OF_FLOAT32 * stats_count: -SIZE_OF_FLOAT32 * stats_count]
        stds = serialized_tensor.buffer[-SIZE_OF_FLOAT32 * stats_count:]
        means = torch.as_tensor(np.frombuffer(means, dtype=np.float32).copy()).view(*stats_size)
        stds = torch.as_tensor(np.frombuffer(stds, dtype=np.float32).copy()).view(*stats_size)
        array = np.frombuffer(serialized_tensor.buffer[:-2 * SIZE_OF_FLOAT32 * stats_count], dtype=np.float16).copy()
        tensor = torch.as_tensor(array, dtype=torch.float32).view(*serialized_tensor.size).mul_(stds).add_(means)
    elif serialized_tensor.compression == CompressionType.FLOAT16:
        array = np.frombuffer(serialized_tensor.buffer, dtype=np.float16).copy()
        tensor = torch.as_tensor(array, dtype=torch.float32).view(*serialized_tensor.size)
    elif serialized_tensor.compression == CompressionType.QUANTILE_8BIT:
        lookup = serialized_tensor.buffer[:NUM_QUANTILES_QUANTILE_COMPRESSION * SIZE_OF_FLOAT32]
        quantized = serialized_tensor.buffer[NUM_QUANTILES_QUANTILE_COMPRESSION * SIZE_OF_FLOAT32:]
        lookup = torch.as_tensor(np.frombuffer(lookup, dtype=np.float32).copy())
        quantized = torch.as_tensor(np.frombuffer(quantized, dtype=np.uint8).copy(),
                                    dtype=torch.int64).view(*serialized_tensor.size)
        tensor = lookup[quantized]
    else:
        raise ValueError(f"Unknown compression type: {serialized_tensor.compression}")

    tensor.requires_grad_(serialized_tensor.requires_grad)
    return tensor


def split_for_streaming(serialized_tensor: runtime_pb2.Tensor, chunk_size_bytes: int) -> Iterator[runtime_pb2.Tensor]:
    """ Split serialized_tensor into multiple chunks for gRPC streaming """
    buffer = memoryview(serialized_tensor.buffer)
    num_chunks = len(range(0, len(buffer), chunk_size_bytes))
    yield runtime_pb2.Tensor(
        compression=serialized_tensor.compression, buffer=buffer[:chunk_size_bytes].tobytes(), chunks=num_chunks,
        size=serialized_tensor.size, dtype=serialized_tensor.dtype, requires_grad=serialized_tensor.requires_grad)
    for chunk_start in range(chunk_size_bytes, len(buffer), chunk_size_bytes):
        yield runtime_pb2.Tensor(buffer=buffer[chunk_start: chunk_start + chunk_size_bytes].tobytes())


def combine_from_streaming(stream: Iterable[runtime_pb2.Tensor]) -> runtime_pb2.Tensor:
    """ Restore a result of split_into_chunks into a single serialized tensor """
    stream = iter(stream)
    first_chunk = next(stream)
    serialized_tensor = runtime_pb2.Tensor()
    serialized_tensor.CopyFrom(first_chunk)
    buffer_chunks = [first_chunk.buffer]
    for tensor_part in stream:
        buffer_chunks.append(tensor_part.buffer)
    serialized_tensor.buffer = b''.join(buffer_chunks)
    return serialized_tensor
