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
from hivemind.proto.runtime_pb2 import CompressionType

from test_utils.dht_swarms import launch_dht_instances


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

    zeros = torch.zeros(5, 5)
    for compression_type in CompressionType.values():
        assert deserialize_torch_tensor(serialize_torch_tensor(zeros, compression_type)).isfinite().all()


def test_partitioning_compression():
    raise NotImplementedError()


def test_serialize_tensor():
    tensor = torch.randn(512, 12288)

    serialized_tensor = serialize_torch_tensor(tensor, CompressionType.NONE)
    for chunk_size in [1024, 64 * 1024, 64 * 1024 + 1, 10 ** 9]:
        chunks = list(hivemind.split_for_streaming(serialized_tensor, chunk_size))
        assert len(chunks) == (len(serialized_tensor.buffer) - 1) // chunk_size + 1
        restored = hivemind.combine_from_streaming(chunks)
        assert torch.allclose(deserialize_torch_tensor(restored), tensor)

    chunk_size = 30 * 1024
    serialized_tensor = serialize_torch_tensor(tensor, CompressionType.FLOAT16)
    chunks = list(hivemind.split_for_streaming(serialized_tensor, chunk_size))
    assert len(chunks) == (len(serialized_tensor.buffer) - 1) // chunk_size + 1
    restored = hivemind.combine_from_streaming(chunks)
    assert torch.allclose(deserialize_torch_tensor(restored), tensor, rtol=0, atol=1e-2)

    tensor = torch.randint(0, 100, (512, 1, 1))
    serialized_tensor = serialize_torch_tensor(tensor, CompressionType.NONE)
    chunks = list(hivemind.split_for_streaming(serialized_tensor, chunk_size))
    assert len(chunks) == (len(serialized_tensor.buffer) - 1) // chunk_size + 1
    restored = hivemind.combine_from_streaming(chunks)
    assert torch.allclose(deserialize_torch_tensor(restored), tensor)

    scalar = torch.tensor(1.0)
    serialized_scalar = serialize_torch_tensor(scalar, CompressionType.NONE)
    assert torch.allclose(deserialize_torch_tensor(serialized_scalar), scalar)

    serialized_scalar = serialize_torch_tensor(scalar, CompressionType.FLOAT16)
    assert torch.allclose(deserialize_torch_tensor(serialized_scalar), scalar)


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


class TestCompression(CompressionBase):
    def __init__(self, compression: CompressionBase):
        self.compression = compression
        self.mp_counter, self.mp_part_size = mp.Value(c_int32, 0), mp.Value(c_int32, 0)
        super().__init__()

    def estimate_compression_ratio(self, info: CompressionInfo):
        return self.compression.estimate_compression_ratio(info)

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


def test_adaptive_compression():
    UINT8 = TestCompression(Uniform8BitQuantization())
    FLOAT16 = TestCompression(Float16Compression())
    FLOAT32 = TestCompression(NoCompression())
    STATE_FP16 = TestCompression(Float16Compression())
    STATE_FP32 = TestCompression(NoCompression())

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

    import hivemind

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
    assert STATE_FP16.mp_counter.value == STATE_FP32.mp_counter.value == 9
    assert STATE_FP16.mp_part_size.value == STATE_FP32.mp_part_size.value == 0  # not partitioned
