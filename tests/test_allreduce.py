import asyncio
import random
import time
from typing import Sequence

import pytest
import torch

from hivemind.client.averaging.partition import TensorPartContainer, TensorPartReducer
from hivemind.utils import serialize_torch_tensor, deserialize_torch_tensor
from hivemind.proto.runtime_pb2 import CompressionType


@pytest.mark.forked
@pytest.mark.asyncio
async def test_partitioning():
    all_tensors = [
        torch.randn(30_000, 128), torch.rand(128), torch.ones(1, 1, 1, 1, 1, 1, 8),
        torch.ones(1, 0), torch.zeros(0), torch.zeros([]), torch.randn(65536),
        torch.rand(512, 2048), torch.randn(1024, 1024).add(-9), torch.zeros(1020), torch.randn(4096)
    ]

    # note: this test does _not_ use parameterization to reuse sampled tensors
    for num_tensors in 1, 3, 5:
        for part_size_bytes in 31337, 2 ** 20, 10 ** 10:
            for weights in [(1, 1), (0.333, 0.1667, 0.5003), (1.0, 0.0), [0.0, 0.4, 0.6, 0.0]]:
                tensors = random.choices(all_tensors, k=num_tensors)
                partition = TensorPartContainer(tensors, weights, part_size_bytes=part_size_bytes)

                async def write_tensors():
                    for i in range(partition.group_size):
                        async for part in partition.iterate_input_parts_for(i):
                            output_tensor = torch.sin(deserialize_torch_tensor(part))
                            part = serialize_torch_tensor(output_tensor, part.compression)
                            partition.append_averaged_part(i, part)
                asyncio.create_task(write_tensors())
                tensor_index = 0
                async for output_tensor in partition.iterate_output_tensors():
                    assert torch.allclose(output_tensor, torch.sin(tensors[tensor_index]))
                    tensor_index += 1


@pytest.mark.parametrize("tensors", [[torch.zeros(0)], [torch.zeros(0), torch.zeros(0), torch.zeros(1)],
                                     [torch.zeros(0), torch.zeros(999), torch.zeros(0), torch.zeros(0)]])
@pytest.mark.parametrize("peer_fractions", [(0.33, 0.44, 0.23), (0.5, 0.5), (0.1, 0.0, 0.9), (1.0,), (0.1,) * 9])
@pytest.mark.forked
@pytest.mark.asyncio
async def test_partitioning_edge_cases(tensors: Sequence[torch.Tensor], peer_fractions: Sequence[float]):
    partition = TensorPartContainer(tensors, peer_fractions, part_size_bytes=16)
    for i in range(len(peer_fractions)):
        async for part in partition.iterate_input_parts_for(i):
            partition.append_averaged_part(i, part)

    tensor_index = 0
    async for output_tensor in partition.iterate_output_tensors():
        assert torch.allclose(output_tensor, tensors[tensor_index])
        tensor_index += 1


@pytest.mark.forked
@pytest.mark.asyncio
async def test_partitioning_asynchronous():
    """ ensure that tensor partitioning does not interfere with asynchronous code """
    tensors = [torch.randn(2048, 2048), torch.randn(1024, 4096),
               torch.randn(4096, 1024), torch.randn(30_000, 1024)]
    peer_fractions = [0.4, 0.3, 0.2, 0.1]

    partition = TensorPartContainer(tensors, peer_fractions, compression_type=CompressionType.QUANTILE_8BIT)
    read_started, read_finished = asyncio.Event(), asyncio.Event()

    async def write_tensors():
        for i in range(partition.group_size):
            async for part in partition.iterate_input_parts_for(i):
                partition.append_averaged_part(i, serialize_torch_tensor(part))
        assert read_started.is_set(), "partitioner should have started reading before it finished writing"

    async def read_tensors():
        async for _ in partition.iterate_output_tensors():
            read_started.set()
        read_finished.set()

    async def wait_synchronously():
        time_in_waiting = 0.0
        while not read_finished.is_set():
            await asyncio.sleep(0.01)
            time_in_waiting += 0.01
        return time_in_waiting

    start_time = time.perf_counter()
    _, _, time_in_waiting = await asyncio.gather(write_tensors(), read_tensors(), wait_synchronously())
    wall_time = time.perf_counter() - start_time
    assert time_in_waiting > wall_time / 3, f"Event loop could only run {time_in_waiting / wall_time :.5f} of the time"


@pytest.mark.parametrize("num_senders", [1, 2, 4, 10])
@pytest.mark.parametrize("num_parts", [0, 1, 100])
@pytest.mark.parametrize("synchronize_prob", [1.0, 0.1, 0.0])
@pytest.mark.forked
@pytest.mark.asyncio
async def test_reducer(num_senders: int, num_parts: int, synchronize_prob: float):
    tensor_part_shapes = [torch.Size([i]) for i in range(num_parts)]
    reducer = TensorPartReducer(tensor_part_shapes, num_senders)

    local_tensors_by_sender = [[torch.randn(i) for i in range(num_parts)]
                               for j in range(num_senders)]

    async def send_tensors(sender_index: int):
        local_tensors = local_tensors_by_sender[sender_index]
        averaged_parts = []
        pending_tasks = []

        for part_index in range(num_parts):
            pending_tasks.append(asyncio.create_task(
                reducer.accumulate_part(sender_index, part_index, local_tensors[part_index])))

            if random.random() < synchronize_prob or part_index == num_parts - 1:
                averaged_parts.extend(await asyncio.gather(*pending_tasks))
                pending_tasks = []
        return averaged_parts

    averaged_tensors_by_peer = await asyncio.gather(*map(send_tensors, range(num_senders)))

    reference = [sum(local_tensors_by_sender[sender_index][part_index]
                     for sender_index in range(num_senders)) / num_senders
                 for part_index in range(num_parts)]

    for averaged_tensors in averaged_tensors_by_peer:
        assert len(averaged_tensors) == len(reference)
        for avg, ref in zip(averaged_tensors, reference):
            assert torch.allclose(avg, ref, rtol=1e-3, atol=1e-5)
