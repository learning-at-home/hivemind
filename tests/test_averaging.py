import asyncio
import random

import numpy as np
import torch
import pytest
import hivemind
from hivemind.client.averaging.allreduce import AllReduceProtocol, split_into_parts, restore_from_parts
from hivemind.client.averaging.load_balancing import load_balance_peers
from hivemind.client.averaging.key_manager import GroupKeyManager
from hivemind.utils import Endpoint


@pytest.mark.forked
@pytest.mark.asyncio
async def test_key_manager():
    key_manager = GroupKeyManager(hivemind.DHT(start=True), endpoint='localhvost',
                                  prefix='test_averaging', initial_group_bits='10110',
                                  target_group_size=2)

    t = hivemind.get_dht_time()
    key = key_manager.current_key
    await key_manager.declare_averager(key, 'localhvost', expiration_time=t + 60)
    await key_manager.declare_averager(key, 'localhvost2', expiration_time=t + 61)

    q1 = await key_manager.get_averagers(key, only_active=True)

    await key_manager.declare_averager(key, 'localhvost', expiration_time=t + 66)
    q2 = await key_manager.get_averagers(key, only_active=True)

    await key_manager.declare_averager(key, 'localhvost2', expiration_time=t + 61, looking_for_group=False)
    q3 = await key_manager.get_averagers(key, only_active=True)
    q4 = await key_manager.get_averagers(key, only_active=False)

    q5 = await key_manager.get_averagers('nonexistent_key.0b0101', only_active=False)

    assert len(q1) == 2 and ('localhvost', t + 60) in q1 and ('localhvost2', t + 61) in q1
    assert len(q2) == 2 and ('localhvost', t + 66) in q2 and ('localhvost2', t + 61) in q2
    assert len(q3) == 1 and ('localhvost', t + 66) in q3
    assert len(q4) == 2 and ('localhvost', t + 66) in q4 and ('localhvost2', t + 61) in q2
    assert len(q5) == 0


@pytest.mark.forked
def test_allreduce_once():
    dht = hivemind.DHT(start=True, endpoint=f'{hivemind.LOCALHOST}:*')

    tensors1 = [torch.randn(123), torch.zeros(3)]
    tensors2 = [torch.rand(123), torch.ones(3)]
    tensors3 = [-torch.rand(123), torch.arange(3).to(torch.float32)]
    tensors4 = [torch.randn(123) ** 3, torch.arange(3).to(torch.float32) / 2]

    reference = [(tensors1[i] + tensors2[i] + tensors3[i] + tensors4[i]) / 4 for i in range(len(tensors1))]

    averagers = [hivemind.DecentralizedAverager(tensors, dht=dht, target_group_size=4, averaging_expiration=15,
                                                prefix='mygroup', listen_on='127.0.0.1:*',
                                                start=True)
                 for tensors in [tensors1, tensors2, tensors3, tensors4]]

    futures = []
    for averager in averagers:
        futures.append(averager.step(wait=False))
    for future in futures:
        result = future.result()
        for averager in averagers:
            assert averager.endpoint in result

    for averager in averagers:
        with averager.get_tensors() as averaged_tensors:
            for ref, our in zip(reference, averaged_tensors):
                assert torch.allclose(ref, our, atol=1e-6)


def compute_mean_std(averagers, unbiased=True):
    results = []
    for averager in averagers:
        with averager.get_tensors() as tensors:
            results.append([tensor.clone() for tensor in tensors])

    results_stacked_per_tensor = list(map(torch.stack, zip(*results)))
    means = [stack.mean(dim=0) for stack in results_stacked_per_tensor]
    stds = [stack.std(dim=0, unbiased=unbiased) for stack in results_stacked_per_tensor]
    return means, stds


@pytest.mark.forked
def test_allreduce_grid():
    dht = hivemind.DHT(start=True, endpoint='127.0.0.1:*')
    averagers = [hivemind.DecentralizedAverager(
        averaged_tensors=[torch.randn(3)], dht=dht, target_group_size=2,
        prefix='mygroup', initial_group_bits=bin(i // 2)[2:].rjust(2, '0'), start=True)
        for i in range(8)]

    [means0], [stds0] = compute_mean_std(averagers)
    assert not torch.allclose(stds0, torch.zeros_like(stds0))

    prev_means, prev_stds = means0, stds0

    for i in range(5):
        step_futures = [averager.step(wait=False) for averager in averagers]
        groups = [future.result() for future in step_futures]
        [means], [stds] = compute_mean_std(averagers)
        assert torch.allclose(means, prev_means, atol=1e-6, rtol=0)
        assert all(len(group) == 2 for group in groups)

        if i <= 2:
            assert torch.all(torch.le(stds, prev_stds))
        else:
            assert torch.allclose(stds, torch.zeros_like(stds), atol=1e-6, rtol=0)


@pytest.mark.forked
def test_allgather():
    dht = hivemind.DHT(start=True, endpoint=f'{hivemind.LOCALHOST}:*')
    averagers = [hivemind.DecentralizedAverager(torch.ones(1), dht=dht, target_group_size=4, averaging_expiration=15,
                                                prefix='mygroup', initial_group_bits='000', listen_on='127.0.0.1:*',
                                                start=True)
                 for _ in range(8)]

    futures = []
    for i, averager in enumerate(averagers):
        futures.append(averager.step(wait=False, gather=dict(batch_size=123 + i, foo='bar')))

    assert len(set(repr(sorted(future.result())) for future in futures)) == 2

    reference_metadata = {averager.endpoint: dict(batch_size=123 + i, foo='bar')
                          for i, averager in enumerate(averagers)}
    for future in futures:
        gathered = future.result()

        assert len(gathered) == 4

        for endpoint in gathered:
            assert gathered[endpoint] == reference_metadata[endpoint]


@pytest.mark.forked
@pytest.mark.asyncio
async def test_allreduce_protocol():
    """ Run group allreduce protocol manually without grpc, see if the internal logic is working as intended """
    peers = "alice", "bob", "carol"

    tensors_by_peer = {peer: [torch.randn(3, 128), torch.rand(32), torch.tensor(i, dtype=torch.float32)]
                       for i, peer in enumerate(peers)}

    group_id = random.getrandbits(160).to_bytes(length=20, byteorder='big')
    allreduce_protocols = [AllReduceProtocol(
        group_id=group_id, endpoint=peer, tensors=tensors_by_peer[peer],
        ordered_group_endpoints=peers, part_sizes=(150, 200, 67))
        for peer in peers]

    async def _accumulate(sender: Endpoint, recipient: Endpoint):
        sender_allreduce = allreduce_protocols[peers.index(sender)]
        recipient_allreduce = allreduce_protocols[peers.index(recipient)]
        averaged_part = await recipient_allreduce.accumulate_part(
            source=sender, remote_part=sender_allreduce.local_tensor_parts[recipient])
        sender_allreduce.register_averaged_part(source=recipient, averaged_part=averaged_part)

    await asyncio.wait({_accumulate(sender, recipient) for sender in peers for recipient in peers
                        if sender != recipient})

    reference_tensors = [
        sum(tensors_by_peer[peer][i] for peer in peers) / len(peers)
        for i in range(len(tensors_by_peer[peers[0]]))
    ]

    for peer, allreduce in zip(peers, allreduce_protocols):
        assert allreduce.future.done()
        averaged_tensors = await allreduce
        assert len(averaged_tensors) == len(reference_tensors)
        assert all(torch.allclose(our, ref, atol=1e-6, rtol=0)
                   for our, ref in zip(averaged_tensors, reference_tensors))


@pytest.mark.forked
def test_partitioning():
    for _ in range(100):
        tensors = []
        for _ in range(random.randint(1, 5)):
            ndim = random.randint(0, 4)
            shape = torch.Size([random.randint(0, 16) for _ in range(ndim)])
            make_tensor = random.choice([torch.rand, torch.randn, torch.zeros, torch.ones])
            tensors.append(make_tensor(shape))

        total_size = sum(map(torch.Tensor.numel, tensors))
        if total_size == 0:
            continue
        num_chunks = random.randint(1, min(1000, sum(x.numel() for x in tensors)))
        part_sizes = load_balance_peers(total_size, [None] * num_chunks)
        chunks = split_into_parts(tensors, part_sizes)
        assert len(chunks) == num_chunks
        shapes = [tensor.shape for tensor in tensors]
        restored = restore_from_parts(chunks, shapes)
        assert len(restored) == len(tensors)
        assert all(new.shape == old.shape for new, old in zip(restored, tensors))
        assert all(torch.allclose(new, old) for new, old in zip(restored, tensors))


def get_cost(vector_size, partitions, throughputs):
    return max((vector_size - partitions[i] + (len(partitions) - 1) * partitions[i]) / max(throughputs[i], 1e-9)
               for i in range(len(partitions)))


def check_optimality(vector_size, throughputs, ref_partitions):
    partitions = list(load_balance_peers(vector_size, throughputs))
    assert get_cost(vector_size, partitions, throughputs) <= get_cost(vector_size, ref_partitions, throughputs)


@pytest.mark.forked
def test_load_balancing():
    check_optimality(60, np.array([0.25, 0.25, 0.25, 0.25]), [15, 15, 15, 15])
    check_optimality(1024, np.array([0.3, 0.5, 0.9]), [0, 255, 769])
    check_optimality(60, np.array([0.44, 0.33, 0.22]), [42, 18, 0])
    check_optimality(60, np.array([0.55, 0.44, 0.40]), [35, 16, 9])
    check_optimality(1024 * 1024, np.array([0.3, 0.5, 0.9, 0.6]), [0, 169327, 602629, 276620])
    check_optimality(1024 * 1024, np.array([0.0, 0.5, 0.0, 0.6]), [0, 428963, 0, 619613])
    assert load_balance_peers(60, np.array([0.55, 0.44, 0.40]), min_size=10) == (41, 19, 0)
    assert load_balance_peers(60, np.array([0.32, 0.55, 0.44]), min_size=10) == (0, 40, 20)
    assert load_balance_peers(2, np.array([0.55, 0.20, 0.44]), min_size=10) == (1, 0, 1)
    assert load_balance_peers(1, np.array([0.55, 0.20, 0.44]), min_size=10) == (1, 0, 0)

    assert load_balance_peers(100, (None, None)) == (50, 50)
    assert load_balance_peers(100, (None, None, None, None, None)) == (20, 20, 20, 20, 20)
    assert load_balance_peers(100, (0, 0, 0, None, None)) == (0, 0, 0, 50, 50)

    with pytest.raises(AssertionError):
        load_balance_peers(100, (0, 0, 0))

    for i in range(10):
        vector_size = np.random.randint(1, 1024 ** 3)
        num_peers = np.random.randint(1, 256)
        scale = 1e-9 + np.random.rand() * 1e5
        throughputs = np.random.rand(num_peers) * scale + 1e-6
        min_size = np.random.choice([0, np.random.randint(0, vector_size // 10)])
        assignment = load_balance_peers(vector_size, throughputs, min_size)
        assert np.sum(assignment) == vector_size
        assert np.min(assignment) >= 0


@pytest.mark.forked
def test_too_few_peers():
    dht = hivemind.DHT(start=True, endpoint='127.0.0.1:*')
    averagers = [hivemind.DecentralizedAverager(
        averaged_tensors=[torch.randn(3)], dht=dht, target_group_size=2,
        averaging_expiration=1, request_timeout=0.5,
        prefix='mygroup', initial_group_bits=bin(i)[2:].rjust(3, '0'), start=True)
        for i in range(4)]
    step_futures = [averager.step(wait=False) for averager in averagers]
    for future in step_futures:
        assert len(future.result()) == 2


@pytest.mark.forked
def test_overcrowded():
    dht = hivemind.DHT(start=True, endpoint='127.0.0.1:*')
    averagers = [hivemind.DecentralizedAverager(
        averaged_tensors=[torch.randn(3)], dht=dht, target_group_size=2,
        averaging_expiration=1, request_timeout=0.5,
        prefix='mygroup', initial_group_bits='', start=True)
        for _ in range(32)]
    for t in range(5):
        step_futures = [averager.step(wait=False, timeout=5) for averager in averagers]
        assert sum(len(future.result() or []) == 2 for future in step_futures) >= len(averagers) - 1
