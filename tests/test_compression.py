import multiprocessing as mp
from ctypes import c_int32

import pytest
import torch
import torch.nn as nn

import hivemind
from hivemind.compression import (
    CompressionBase,
    CompressionInfo,
    Float16Compression,
    NoCompression,
    PerTensorCompression,
    RoleAdaptiveCompression,
    SizeAdaptiveCompression,
    Uniform8BitQuantization,
    deserialize_torch_tensor,
    serialize_torch_tensor,
)
from hivemind.compression.adaptive import AdaptiveCompressionBase
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.streaming import combine_from_streaming, split_for_streaming

from test_utils.dht_swarms import launch_dht_instances


@pytest.mark.forked
def test_tensor_compression(size=(128, 128, 64), alpha=5e-08, beta=0.0008):
    torch.manual_seed(0)
    X = torch.randn(*size)
    assert torch.allclose(deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.NONE)), X)
    error = deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.MEANSTD_16BIT)) - X
    assert error.square().mean() < alpha
    error = deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.FLOAT16)) - X
    assert error.square().mean() < alpha
    error = deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.QUANTILE_8BIT)) - X
    assert error.square().mean() < beta
    error = deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.UNIFORM_8BIT)) - X
    assert error.square().mean() < beta
    error = deserialize_torch_tensor(serialize_torch_tensor(X, CompressionType.BLOCKWISE_8BIT)) - X
    assert error.square().mean() < beta

    zeros = torch.zeros(5, 5)
    for compression_type in CompressionType.values():
        assert deserialize_torch_tensor(serialize_torch_tensor(zeros, compression_type)).isfinite().all()


def _check(tensor, compression, rtol=1e-5, atol=1e-8, chunk_size=30 * 1024):
    serialized_tensor = serialize_torch_tensor(tensor, compression)
    chunks = list(split_for_streaming(serialized_tensor, chunk_size))
    assert len(chunks) == (len(serialized_tensor.buffer) - 1) // chunk_size + 1
    restored = combine_from_streaming(chunks)
    result = deserialize_torch_tensor(restored)
    assert torch.allclose(result, tensor, rtol=rtol, atol=atol)
    assert result.dtype == tensor.dtype


@pytest.mark.forked
def test_serialize_tensor():
    tensor = torch.randn(512, 12288)
    for chunk_size in [1024, 64 * 1024, 64 * 1024 + 1, 10**9]:
        _check(tensor, CompressionType.NONE, chunk_size=chunk_size)

    _check(tensor, CompressionType.FLOAT16, rtol=0.0, atol=1e-2)
    _check(torch.randint(0, 100, (512, 1, 1)), CompressionType.NONE)
    _check(torch.tensor(1.0), CompressionType.NONE)
    _check(torch.tensor(1.0), CompressionType.FLOAT16)


@pytest.mark.parametrize("use_legacy_bfloat16", [True, False])
@pytest.mark.forked
def test_serialize_bfloat16(use_legacy_bfloat16: bool):
    hivemind.compression.base.USE_LEGACY_BFLOAT16 = use_legacy_bfloat16
    tensor = torch.randn(4096, 16, dtype=torch.bfloat16)
    _check(tensor, CompressionType.NONE)
    _check(tensor, CompressionType.BLOCKWISE_8BIT, rtol=0.1, atol=0.01, chunk_size=1024)


@pytest.mark.forked
def test_allreduce_compression():
    """this test ensures that compression works correctly when multiple tensors have different compression types"""

    tensors1 = [torch.linspace(0, 500, 1000) ** 0.5, torch.randn(1000)]
    tensors2 = [torch.linspace(300, 800, 1000) ** 0.5, torch.randn(1000)]
    results = {}

    FLOAT16, UINT8 = Float16Compression(), Uniform8BitQuantization()

    for compression_type_pair in [(FLOAT16, FLOAT16), (FLOAT16, UINT8), (UINT8, FLOAT16), (UINT8, UINT8)]:
        dht_instances = launch_dht_instances(2)
        averager1 = hivemind.averaging.DecentralizedAverager(
            [x.clone() for x in tensors1],
            dht=dht_instances[0],
            compression=PerTensorCompression(compression_type_pair),
            client_mode=True,
            target_group_size=2,
            prefix="mygroup",
            start=True,
        )
        averager2 = hivemind.averaging.DecentralizedAverager(
            [x.clone() for x in tensors2],
            dht=dht_instances[1],
            compression=PerTensorCompression(compression_type_pair),
            target_group_size=2,
            prefix="mygroup",
            start=True,
        )

        for future in averager1.step(wait=False), averager2.step(wait=False):
            future.result()

        with averager1.get_tensors() as averaged_tensors:
            results[compression_type_pair] = averaged_tensors

        for instance in [averager1, averager2] + dht_instances:
            instance.shutdown()

    assert torch.allclose(results[UINT8, FLOAT16][0], results[UINT8, UINT8][0])
    assert torch.allclose(results[UINT8, FLOAT16][1], results[FLOAT16, FLOAT16][1])
    assert torch.allclose(results[UINT8, UINT8][1], results[FLOAT16, UINT8][1])
    assert torch.allclose(results[FLOAT16, UINT8][0], results[FLOAT16, FLOAT16][0])

    assert not torch.allclose(results[UINT8, FLOAT16][1], results[UINT8, UINT8][1])
    assert not torch.allclose(results[UINT8, FLOAT16][0], results[FLOAT16, FLOAT16][0])
    assert not torch.allclose(results[UINT8, UINT8][0], results[FLOAT16, UINT8][0])
    assert not torch.allclose(results[FLOAT16, UINT8][1], results[FLOAT16, FLOAT16][1])

    reference = [(tensors1[i] + tensors2[i]) / 2 for i in range(len(tensors1))]
    for i in range(2):
        assert 0 < torch.mean(torch.square(results[FLOAT16, FLOAT16][i] - reference[i])).item() <= 1e-5
        assert 1e-5 < torch.mean(torch.square(results[UINT8, UINT8][i] - reference[i])).item() <= 1e-2


class TrackedCompression(AdaptiveCompressionBase):
    def __init__(self, compression: CompressionBase):
        self.compression = compression
        self.mp_counter, self.mp_part_size = mp.Value(c_int32, 0), mp.Value(c_int32, 0)
        super().__init__()

    def choose_compression(self, info: CompressionInfo) -> CompressionBase:
        return self.compression

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False):
        self.mp_counter.value += 1
        if info.part_size is not None:
            self.mp_part_size.value = max(self.mp_part_size.value, info.part_size)
        return self.compression.compress(tensor, info=info, allow_inplace=allow_inplace)


def make_params():
    return [
        nn.Parameter(x)
        for x in (
            torch.randn([]),
            torch.randn(1),
            torch.randn(100),
            torch.randn(1_000),
            torch.randn(5_000),
            torch.randn(10_000),
        )
    ]


@pytest.mark.forked
def test_adaptive_compression():
    UINT8 = TrackedCompression(Uniform8BitQuantization())
    FLOAT16 = TrackedCompression(Float16Compression())
    FLOAT32 = TrackedCompression(NoCompression())
    STATE_FP16 = TrackedCompression(Float16Compression())
    STATE_FP32 = TrackedCompression(NoCompression())

    averaging_compression_adaptive = RoleAdaptiveCompression(
        parameter=FLOAT16,
        gradient=SizeAdaptiveCompression(threshold=1_000, less=FLOAT16, greater_equal=UINT8),
        optimizer=FLOAT32,
        default=FLOAT32,
    )

    state_compression_adaptive = SizeAdaptiveCompression(
        threshold=500,
        less=STATE_FP32,
        greater_equal=STATE_FP16,
    )

    averager1 = hivemind.TrainingAverager(
        opt=torch.optim.Adam(make_params()),
        average_parameters=True,
        average_gradients=True,
        average_opt_statistics=("exp_avg",),
        compression=averaging_compression_adaptive,
        state_compression=state_compression_adaptive,
        prefix="test_avgr",
        target_group_size=2,
        part_size_bytes=5_000,
        start=True,
        dht=hivemind.DHT(start=True),
    )

    averager2 = hivemind.TrainingAverager(
        opt=torch.optim.Adam(make_params()),
        average_parameters=True,
        average_gradients=True,
        average_opt_statistics=("exp_avg",),
        compression=averaging_compression_adaptive,
        state_compression=state_compression_adaptive,
        prefix="test_avgr",
        target_group_size=2,
        part_size_bytes=5_000,
        start=True,
        dht=hivemind.DHT(initial_peers=averager1.dht.get_visible_maddrs(), start=True),
    )

    futures = [averager1.step(wait=False), averager2.step(wait=False)]

    for future in futures:
        future.result()

    assert UINT8.mp_counter.value == 4  # half gradients: 3 tensors, 1 is split
    assert UINT8.mp_part_size.value == 5_000  # single byte tensors
    assert FLOAT16.mp_counter.value == 13  # parameters and half gradients
    assert FLOAT16.mp_part_size.value == 2_500  # two-byte tensors
    assert FLOAT32.mp_counter.value == 16  # statistics
    assert FLOAT32.mp_part_size.value == 1250  # four-byte tensors

    averager1.load_state_from_peers()
    state_metadata, state_tensors, infos = averager1.get_current_state()
    assert STATE_FP16.mp_counter.value == len([tensor for tensor in state_tensors if tensor.numel() >= 500])
    assert STATE_FP32.mp_counter.value == len([tensor for tensor in state_tensors if tensor.numel() < 500])
    assert STATE_FP16.mp_part_size.value == STATE_FP32.mp_part_size.value == 0  # not partitioned
