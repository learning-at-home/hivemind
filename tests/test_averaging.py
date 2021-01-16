import asyncio
import random
import time

import torch
import pytest
import hivemind
from hivemind.client.averaging.allreduce import AllReduceProtocol, split_into_parts, restore_from_parts
from hivemind.utils import Endpoint


@pytest.mark.forked
def test_getset_averagers():
    dht = hivemind.DHT(start=True)

    t = hivemind.get_dht_time()
    dht.declare_averager(group_key='bucket.0b10110', endpoint='localhvost', expiration_time=t + 60)
    dht.declare_averager(group_key='bucket.0b10110', endpoint='localhvost2', expiration_time=t + 61)

    q1 = dht.get_averagers('bucket.0b10110', only_active=True)

    dht.declare_averager(group_key='bucket.0b10110', endpoint='localhvost', expiration_time=t + 66)
    q2 = dht.get_averagers('bucket.0b10110', only_active=True)

    dht.declare_averager(group_key='bucket.0b10110', endpoint='localhvost2', looking_for_group=False,
                         expiration_time=t + 61)
    q3 = dht.get_averagers('bucket.0b10110', only_active=True)
    q4 = dht.get_averagers('bucket.0b10110', only_active=False)

    assert len(q1) == 2 and ('localhvost', t + 60) in q1 and ('localhvost2', t + 61) in q1
    assert len(q2) == 2 and ('localhvost', t + 66) in q2 and ('localhvost2', t + 61) in q2
    assert len(q3) == 1 and ('localhvost', t + 66) in q3
    assert len(q4) == 2 and ('localhvost', t + 66) in q4 and ('localhvost2', t + 61) in q2


@pytest.mark.forked
def test_allreduce_once():
    dht = hivemind.DHT(start=True)

    tensors1 = [torch.randn(123), torch.zeros(3)]
    tensors2 = [torch.rand(123), torch.ones(3)]
    tensors3 = [-torch.rand(123), torch.arange(3).to(torch.float32)]
    tensors4 = [torch.randn(123) ** 3, torch.arange(3).to(torch.float32) / 2]

    reference = [(tensors1[i] + tensors2[i] + tensors3[i] + tensors4[i]) / 4 for i in range(len(tensors1))]

    averagers = [hivemind.DecentralizedAverager(tensors, dht=dht, target_group_size=4, averaging_expiration=15,
                                                prefix='mygroup', initial_group_bits='0110', listen_on='127.0.0.1:*',
                                                start=True)
                 for tensors in [tensors1, tensors2, tensors3, tensors4]]

    futures = []
    for averager in averagers:
        futures.append(averager.step(wait=False))
    for future in futures:
        assert future.result() is True

    for averager in averagers:
        with averager.get_tensors() as averaged_tensors:
            for ref, our in zip(reference, averaged_tensors):
                assert torch.allclose(ref, our, atol=1e-6)


@pytest.mark.forked
@pytest.mark.asyncio
async def test_allreduce_protocol():
    """ Run group allreduce protocol manually without grpc, see if the internal logic is working as intended """
    peers = "alice", "bob", "carol"

    tensors_by_peer = {peer: [torch.randn(3, 128), torch.rand(32), torch.tensor(i, dtype=torch.float32)]
                       for i, peer in enumerate(peers)}

    group_id = random.getrandbits(160).to_bytes(length=20, byteorder='big')
    allreduce_protocols = [AllReduceProtocol(
        group_id=group_id, endpoint=peer, tensors=tensors_by_peer[peer], ordered_group_endpoints=peers)
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
        chunks = split_into_parts(tensors, group_size=num_chunks)
        assert len(chunks) == num_chunks
        shapes = [tensor.shape for tensor in tensors]
        restored = restore_from_parts(chunks, shapes)
        assert len(restored) == len(tensors)
        assert all(new.shape == old.shape for new, old in zip(restored, tensors))
        assert all(torch.allclose(new, old) for new, old in zip(restored, tensors))

@pytest.mark.forked
def test_load_balancing():
    assert load_balance_peers(60, np.array([0.25, 0.25, 0.25, 0.25])).tolist() == [15, 15, 15, 15]
    assert load_balance_peers(1024, np.array([0.3, 0.5, 0.9])).tolist() == [0, 255, 769]
    assert load_balance_peers(60, np.array([0.44, 0.33, 0.22])).tolist() == [42, 18, 0]
    assert load_balance_peers(60, np.array([0.55, 0.44, 0.40])).tolist() == [35, 16, 9]
    assert load_balance_peers(60, np.array([0.55, 0.44, 0.40]), min_size=10).tolist() == [41, 19, 0]
    assert load_balance_peers(60, np.array([0.55, 0.20, 0.44]), min_size=10).tolist() == [36, 0, 24]
    assert load_balance_peers(2, np.array([0.55, 0.20, 0.44]), min_size=10).tolist() == [1, 0, 1]
    assert load_balance_peers(1, np.array([0.55, 0.20, 0.44]), min_size=10).tolist() == [1, 0, 0]

    for i in range(10):
        vector_size = np.random.randint(1, 1024 ** 3)
        num_peers = np.random.randint(1, 256)
        scale = 1e-9 + np.random.rand() * 1000
        throughputs = np.random.rand(num_peers) * scale + 1e-6
        min_size = np.random.choice([0, np.random.randint(0, vector_size // 10)])
        assignment = load_balance_peers(vector_size, throughputs, min_size)
        assert np.sum(assignment) == vector_size
        assert np.min(assignment) >= 0
