import asyncio
import random
import time
from typing import Sequence

import pytest
import torch

from hivemind import Quantile8BitQuantization, aenumerate
from hivemind.averaging.allreduce import AllReduceRunner, AveragingMode
from hivemind.averaging.partition import TensorPartContainer, TensorPartReducer
from hivemind.compression import deserialize_torch_tensor
from hivemind.p2p import P2P


@pytest.mark.forked
@pytest.mark.asyncio
async def test_partitioning():
    all_tensors = [
        torch.randn(30_000, 128),
        torch.rand(128),
        torch.ones(1, 1, 1, 1, 1, 1, 8),
        torch.ones(1, 0),
        torch.zeros(0),
        torch.zeros([]),
        torch.randn(65536),
        torch.rand(512, 2048),
        torch.randn(1024, 1024).add(-9),
        torch.zeros(1020),
        torch.randn(4096),
    ]

    # note: this test does _not_ use parameterization to reuse sampled tensors
    for num_tensors in 1, 3, 5:
        for part_size_bytes in 31337, 2**20, 10**10:
            for weights in [(1, 1), (0.333, 0.1667, 0.5003), (1.0, 0.0), [0.0, 0.4, 0.6, 0.0]]:
                tensors = random.choices(all_tensors, k=num_tensors)
                partition = TensorPartContainer(tensors, weights, part_size_bytes=part_size_bytes)

                async def write_tensors():
                    for peer_index in range(partition.group_size):
                        async for part_index, part in aenumerate(partition.iterate_input_parts_for(peer_index)):
                            output_tensor = torch.sin(deserialize_torch_tensor(part))
                            partition.register_processed_part(peer_index, part_index, output_tensor)

                task = asyncio.create_task(write_tensors())
                tensor_index = 0
                async for output_tensor in partition.iterate_output_tensors():
                    assert torch.allclose(output_tensor, torch.sin(tensors[tensor_index]))
                    tensor_index += 1
                assert tensor_index == len(tensors)
                await task


@pytest.mark.parametrize(
    "tensors",
    [
        [torch.zeros(0)],
        [torch.zeros(0), torch.zeros(0), torch.zeros(1)],
        [torch.zeros(0), torch.zeros(999), torch.zeros(0), torch.zeros(0)],
    ],
)
@pytest.mark.parametrize("peer_fractions", [(0.33, 0.44, 0.23), (0.5, 0.5), (0.1, 0.0, 0.9), (1.0,), (0.1,) * 9])
@pytest.mark.forked
@pytest.mark.asyncio
async def test_partitioning_edge_cases(tensors: Sequence[torch.Tensor], peer_fractions: Sequence[float]):
    partition = TensorPartContainer(tensors, peer_fractions, part_size_bytes=16)
    for peer_index in range(len(peer_fractions)):
        async for part_index, part in aenumerate(partition.iterate_input_parts_for(peer_index)):
            partition.register_processed_part(peer_index, part_index, deserialize_torch_tensor(part))

    tensor_index = 0
    async for output_tensor in partition.iterate_output_tensors():
        assert torch.allclose(output_tensor, tensors[tensor_index])
        tensor_index += 1


@pytest.mark.forked
@pytest.mark.asyncio
async def test_partitioning_asynchronous():
    """ensure that tensor partitioning does not interfere with asynchronous code"""
    tensors = [torch.randn(2048, 2048), torch.randn(1024, 4096), torch.randn(4096, 1024), torch.randn(30_000, 1024)]
    peer_fractions = [0.4, 0.3, 0.2, 0.1]

    partition = TensorPartContainer(tensors, peer_fractions, compression=Quantile8BitQuantization())
    read_started, read_finished = asyncio.Event(), asyncio.Event()

    async def write_tensors():
        for peer_index in range(partition.group_size):
            async for part_index, part in aenumerate(partition.iterate_input_parts_for(peer_index)):
                partition.register_processed_part(peer_index, part_index, deserialize_torch_tensor(part))
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
    *_, time_in_waiting = await asyncio.gather(write_tensors(), read_tensors(), wait_synchronously())
    wall_time = time.perf_counter() - start_time
    # check that event loop had enough time to respond to incoming requests; this is over 50% most of the time
    # we set 33% threshold to ensure that the test will pass reliably. If we break prefetch, this drops to <10%
    assert time_in_waiting > wall_time / 3, f"Event loop could only run {time_in_waiting / wall_time :.5f} of the time"


@pytest.mark.parametrize("num_senders", [1, 2, 4, 10])
@pytest.mark.parametrize("num_parts", [0, 1, 100])
@pytest.mark.parametrize("synchronize_prob", [1.0, 0.1, 0.0])
@pytest.mark.forked
@pytest.mark.asyncio
async def test_reducer(num_senders: int, num_parts: int, synchronize_prob: float):
    tensor_part_shapes = [torch.Size([i]) for i in range(num_parts)]
    reducer = TensorPartReducer(tensor_part_shapes, num_senders)

    local_tensors_by_sender = [[torch.randn(i) for i in range(num_parts)] for j in range(num_senders)]

    async def send_tensors(sender_index: int):
        local_tensors = local_tensors_by_sender[sender_index]
        averaged_parts = []
        pending_tasks = []

        for part_index in range(num_parts):
            pending_tasks.append(
                asyncio.create_task(reducer.accumulate_part(sender_index, part_index, local_tensors[part_index]))
            )

            if random.random() < synchronize_prob or part_index == num_parts - 1:
                averaged_parts.extend(await asyncio.gather(*pending_tasks))
                pending_tasks = []
        return averaged_parts

    averaged_tensors_by_peer = await asyncio.gather(*map(send_tensors, range(num_senders)))

    reference = [
        sum(local_tensors_by_sender[sender_index][part_index] for sender_index in range(num_senders)) / num_senders
        for part_index in range(num_parts)
    ]

    for averaged_tensors in averaged_tensors_by_peer:
        assert len(averaged_tensors) == len(reference)
        for averaging_result, reference_tensor in zip(averaged_tensors, reference):
            assert torch.allclose(averaging_result, reference_tensor, rtol=1e-3, atol=1e-5)


NODE, CLIENT, AUX = AveragingMode.NODE, AveragingMode.CLIENT, AveragingMode.AUX


@pytest.mark.parametrize(
    "peer_modes, averaging_weights, peer_fractions, part_size_bytes",
    [
        ((NODE, NODE, NODE, NODE), (1, 1, 1, 1), (1, 1, 1, 1), 2**20),
        ((NODE, NODE, NODE, NODE), (0.1, 0.2, 0.3, 0.4), (1, 1, 1, 1), 2**20),
        ((NODE, NODE, NODE, NODE), (1, 1, 1, 1), (1, 2, 3, 0), 2**20),
        ((NODE, NODE, NODE, CLIENT), (1, 1, 1, 1), (1, 2, 3, 0), 2**20),
        ((NODE, NODE, NODE, AUX), (1, 1, 1, 0), (1, 2, 3, 4), 2**20),
        ((NODE, NODE, NODE, NODE), (0.15, 0.0, 0.35, 0.45), (1, 1, 1, 1), 2**20),
        ((NODE, AUX, NODE, CLIENT), (0.15, 0.0, 0.35, 0.45), (150, 200, 67, 0), 2**20),
        ((NODE, AUX, NODE, CLIENT), (0.15, 0.0, 0.35, 0.45), (150, 200, 67, 0), 256),
        ((NODE, AUX, NODE, CLIENT), (0.15, 0.0, 0.35, 0.45), (150, 200, 67, 0), 19),
        ((AUX, AUX, AUX, AUX), (0.0, 0.0, 0.0, 0.0), (1, 2, 3, 4), 2**20),
    ],
)
@pytest.mark.forked
@pytest.mark.asyncio
async def test_allreduce_protocol(peer_modes, averaging_weights, peer_fractions, part_size_bytes):
    """Run group allreduce protocol manually without grpc, see if the internal logic is working as intended"""

    p2ps = [await P2P.create()]
    visible_maddrs = await p2ps[0].get_visible_maddrs()
    p2ps += await asyncio.gather(*[P2P.create(initial_peers=visible_maddrs) for _ in range(3)])

    peers = [instance.peer_id for instance in p2ps]
    tensors_by_peer = {
        peer: [torch.randn(3, 128), torch.rand(32), torch.tensor(i, dtype=torch.float32)]
        for i, peer in enumerate(peers)
    }

    group_id = random.getrandbits(160).to_bytes(length=20, byteorder="big")

    allreduce_protocols = []
    for i, p2p in enumerate(p2ps):
        allreduce_protocol = AllReduceRunner(
            p2p=p2p,
            servicer_type=AllReduceRunner,
            prefix=None,
            group_id=group_id,
            tensors=[x.clone() for x in tensors_by_peer[p2p.peer_id]],
            ordered_peer_ids=peers,
            peer_fractions=peer_fractions,
            modes=peer_modes,
            weight=averaging_weights[i],
            part_size_bytes=part_size_bytes,
        )
        await allreduce_protocol.add_p2p_handlers(p2p)
        allreduce_protocols.append(allreduce_protocol)

    async def _run_allreduce_inplace(allreduce: AllReduceRunner):
        async for tensor_index, tensor_delta in aenumerate(allreduce):
            allreduce.tensor_part_container.local_tensors[tensor_index].add_(tensor_delta)

    await asyncio.gather(*map(_run_allreduce_inplace, allreduce_protocols))

    reference_tensors = [
        sum(tensors_by_peer[peer][i] * averaging_weights[peer_index] for peer_index, peer in enumerate(peers))
        / sum(averaging_weights)
        for i in range(len(tensors_by_peer[peers[0]]))
    ]

    for peer_index, protocol in enumerate(allreduce_protocols):
        assert protocol._future.done()
        if protocol.modes[peer_index] != AveragingMode.AUX:
            targets_for_peer = reference_tensors
        else:
            targets_for_peer = tensors_by_peer[peers[peer_index]]
        output_tensors = protocol.tensor_part_container.local_tensors
        assert len(output_tensors) == len(targets_for_peer)
        assert all(torch.allclose(our, ref, atol=1e-6, rtol=0) for our, ref in zip(output_tensors, targets_for_peer))

    for instance in p2ps:
        await instance.shutdown()
