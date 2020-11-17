"""
Utilities for running GRPC services: compile protobuf, patch legacy versions, etc
"""
from __future__ import annotations
import os
import threading
from typing import NamedTuple, Sequence, Tuple, Optional, Union, Any

import grpc
import grpc.experimental
import numpy as np
import torch

from hivemind.proto import runtime_pb2
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.timed_storage import TimedStorage, get_dht_time
from hivemind.utils.networking import Endpoint
from hivemind.utils.logging import get_logger

logger = get_logger(__file__)


class ChannelInfo(NamedTuple):
    target: Endpoint
    aio: bool
    options: Tuple[Tuple[str, str], ...]
    credentials: Optional[grpc.ChannelCredentials]
    compression: Optional[grpc.Compression]


class ChannelCache(TimedStorage[ChannelInfo, Union[grpc.Channel, grpc.aio.Channel]]):
    """
    A process-wide cache of gRPC channels, supports both normal and aio channels, secure/insecure channels, etc
    Based on grpcio internal channel cache by Richard Belleville and Lidi Zheng (thanks!)
    Unlike TimedStorage, ChannelCache actively evicts stale channels even if the cache is not accessed
    Unlike grpc._simple_stubs.ChannelCache, this implementation supports aio and does not forcibly close active channels
    """
    EVICTION_PERIOD_SECONDS = os.environ.get("GRPC_PYTHON_MANAGED_CHANNEL_EVICTION_SECONDS", 10 * 60)
    MAXIMUM_CHANNELS = os.environ.get("GRPC_PYTHON_MANAGED_CHANNEL_MAXIMUM", 1024)
    logger.debug(f"Eviction period = {EVICTION_PERIOD_SECONDS}s, max channels = {MAXIMUM_CHANNELS}")

    _singleton: Optional[ChannelCache] = None
    _singleton_pid: int = os.getpid()
    _lock: threading.RLock = threading.RLock()
    _new_top_evt: threading.Event = threading.Event()
    _eviction_thread: threading.Thread

    def __init__(self):
        super().__init__(maxsize=self.MAXIMUM_CHANNELS)
        self._eviction_thread = threading.Thread(target=self._close_stale_channels_in_background, daemon=True)
        self._eviction_thread.start()

    @classmethod
    def get_singleton(cls):
        with cls._lock:
            if cls._singleton is None or cls._singleton_pid != os.getpid():
                cls._singleton, cls._singleton_pid = cls(), os.getpid()
            return cls._singleton

    @classmethod
    def get_channel(cls, target: Endpoint, *, aio: bool, options: Sequence[Tuple[str, Any], ...] = (),
                    channel_credentials: Optional[grpc.ChannelCredentials] = None,
                    compression: Optional[grpc.Compression] = None) -> Union[grpc.Channel, grpc.aio.Channel]:
        """
        Create a grpc channel with given options or reuse pre-existing one

        :param target: the recipient's address and port
        :param aio: if True, returns grpc.Channel, otherwise returns grpc.aio.Channel
        :param options: see https://grpc.github.io/grpc/core/group__grpc__arg__keys.html
        :param channel_credentials: if specified, create a secure channel usin these credentials (default = insecure)
        :param compression: see https://github.com/grpc/grpc/tree/master/examples/python/compression
        """
        cache = cls.get_singleton()
        with cls._lock:
            key = ChannelInfo(target, aio, tuple(options), channel_credentials, compression)
            channel, _ = super(cls, cache).get(key) or (None, None)
            if channel is None:
                channel = cls._create_channel(*key)

            # either cache channel or update expiration of an existing channel
            super(cls, cache).store(key, channel, get_dht_time() + cls.EVICTION_PERIOD_SECONDS)

            new_top_key, _ = super(cls, cache).top()
            if key is new_top_key:
                cls._new_top_evt.set()

            return channel

    @classmethod
    def _create_channel(cls, target: Endpoint, aio: bool, options: Sequence[Tuple[str, Any], ...],
                        channel_credentials: Optional[grpc.ChannelCredentials],
                        compression: Optional[grpc.Compression]) -> Union[grpc.Channel, grpc.aio.Channel]:
        namespace = grpc.aio if aio else grpc
        if channel_credentials is grpc.experimental.insecure_channel_credentials():
            logger.debug(f"Creating insecure {namespace} channel with options '{options}' " +
                         f"and compression '{compression}'")
            return grpc.insecure_channel(target, options=options, compression=compression)
        else:
            logger.debug(f"Creating secure {namespace} channel with credentials '{channel_credentials}', "
                         + f"options '{options}' and compression '{compression}'")
            return namespace.secure_channel(target, credentials=channel_credentials,
                                            options=options, compression=compression)

    @classmethod
    def _close_stale_channels_in_background(cls):
        while True:
            try:
                cache = cls.get_singleton()
                now = get_dht_time()

                with cls._lock:
                    cache._remove_outdated()
                    with cache.freeze():
                        if len(cache) > 0:
                            _, (_, nearest_exiration) = super(cls, cache).top()
                        else:
                            nearest_exiration = float('inf')

                time_to_wait = max(0.0, nearest_exiration - now)
                cls._new_top_evt.wait(time_to_wait if time_to_wait != float('inf') else None)
            except Exception as e:
                logger.exception(e)

    def store(self, *args, **kwargs) -> ValueError:
        raise ValueError(f"Please use {self.__class__.__name__}.get_channel to get/create channels")

    def get(self, *args, **kwargs) -> ValueError:
        raise ValueError(f"Please use {self.__class__.__name__}.get_channel to get/create channels")

    def top(self) -> ValueError:
        raise ValueError(f"Please use {self.__class__.__name__}.get_channel to get/create channels")




FP16_MAX = 65_504


def serialize_torch_tensor(tensor: torch.Tensor, compression_type=CompressionType.NONE,
                           allow_inplace=False) -> runtime_pb2.Tensor:
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
    else:
        raise ValueError(f"Unknown compression type: {compression_type}")

    return proto


def deserialize_torch_tensor(serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
    # TODO avoid copying the array (need to silence pytorch warning, because array is not writable)
    if serialized_tensor.compression == CompressionType.NONE:
        array = np.frombuffer(serialized_tensor.buffer, dtype=np.dtype(serialized_tensor.dtype)).copy()
        tensor = torch.as_tensor(array).view(*serialized_tensor.size).requires_grad_(serialized_tensor.requires_grad)
    elif serialized_tensor.compression == CompressionType.MEANSTD_LAST_AXIS_FLOAT16:
        stats_size = list(serialized_tensor.size)
        stats_size[-1] = 1
        stats_count = np.prod(stats_size)
        means, stds = serialized_tensor.buffer[-8*stats_count:-4*stats_count], serialized_tensor.buffer[-4*stats_count:]
        means = torch.as_tensor(np.frombuffer(means, dtype=np.float32).copy()).view(*stats_size)
        stds = torch.as_tensor(np.frombuffer(stds, dtype=np.float32).copy()).view(*stats_size)
        array = np.frombuffer(serialized_tensor.buffer[:-8 * stats_count], dtype=np.float16).copy()
        tensor = torch.as_tensor(array).to(torch.float32).view(*serialized_tensor.size).mul_(stds).add_(means)
    elif serialized_tensor.compression == CompressionType.FLOAT16:
        array = np.frombuffer(serialized_tensor.buffer, dtype=np.float16).copy()
        tensor = torch.as_tensor(array).view(*serialized_tensor.size)\
            .to(torch.float32).requires_grad_(serialized_tensor.requires_grad)
    else:
        raise ValueError(f"Unknown compression type: {serialized_tensor.compression}")
    return tensor
