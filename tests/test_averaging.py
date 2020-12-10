import asyncio
import random
import time

import torch
import pytest
import hivemind
from hivemind.client.averager import GroupAllReduce, split_into_parts, restore_from_parts
from hivemind.utils import LOCALHOST, Endpoint


@pytest.mark.forked
@pytest.mark.asyncio
async def test_allreduce_direct():
    # WARNING! this test uses an early interface that will change by the time DecentralizedAverager is finished

    dht = hivemind.DHT(start=True)

    tensors1 = [torch.randn(123), torch.zeros(3)]
    tensors2 = [torch.rand(123), torch.ones(3)]
    tensors3 = [-torch.rand(123), torch.arange(3).to(torch.float32)]

    reference = [(tensors1[i] + tensors2[i] + tensors3[i]) / 3 for i in range(len(tensors1))]

    averager1 = hivemind.DecentralizedAverager(tensors1, dht=dht, start=True, target_group_size=3, timeout=5)
    averager2 = hivemind.DecentralizedAverager(tensors2, dht=dht, start=True, target_group_size=3, timeout=5)
    averager3 = hivemind.DecentralizedAverager(tensors3, dht=dht, start=True, target_group_size=3, timeout=5)

    future1 = averager1.group_allreduce(my_endpoint=f"{LOCALHOST}:{averager1.port}",
                                        leader_endpoint=None, return_future=True)
    time.sleep(0.1)

    future2 = averager2.group_allreduce(my_endpoint=f"{LOCALHOST}:{averager2.port}",
                                        leader_endpoint=f"{LOCALHOST}:{averager1.port}",
                                        return_future=True)

    future3 = averager3.group_allreduce(my_endpoint=f"{LOCALHOST}:{averager3.port}",
                                        leader_endpoint=f"{LOCALHOST}:{averager1.port}",
                                        return_future=True)

    for future in future1, future2, future3:
        for ref, our in zip(reference, await future):
            assert torch.allclose(ref, our)


@pytest.mark.forked
@pytest.mark.asyncio
async def test_allreduce_protocol():
    """ Run group allreduce protocol manually without grpc, see if the internal logic is working as intended """
    peers = "alice", "bob", "carol"

    tensors_by_peer = {peer: [torch.randn(3, 128), torch.rand(32), torch.tensor(i, dtype=torch.float32)]
                       for i, peer in enumerate(peers)}

    group_id = bytes(hivemind.DHTID.generate())
    allreduce_protocols = [GroupAllReduce(
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
        assert allreduce.averaged_tensors.done()
        averaged_tensors = await allreduce
        assert len(averaged_tensors) == len(reference_tensors)
        assert all(torch.allclose(our, ref, atol=1e-6, rtol=0)
                   for our, ref in zip(averaged_tensors, reference_tensors))


@pytest.mark.forked
def test_chunks():
    for i in range(100):
        tensors = []
        for i in range(random.randint(1, 5)):
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
def test_getset_averagers():
    import hivemind
    dht = hivemind.DHT(start=True)

    t = hivemind.get_dht_time()
    dht.declare_averager(allreduce_group='bucket.0b10110', endpoint='localhvost', expiration_time=t + 60)
    dht.declare_averager(allreduce_group='bucket.0b10110', endpoint='localhvost2', expiration_time=t + 61)

    q1 = dht.get_averagers('bucket.0b10110', only_active=True)

    dht.declare_averager(allreduce_group='bucket.0b10110', endpoint='localhvost', expiration_time=t + 66)
    q2 = dht.get_averagers('bucket.0b10110', only_active=True)

    dht.declare_averager(allreduce_group='bucket.0b10110', endpoint='localhvost2', looking_for_group=False,
                         expiration_time=t + 61)
    q3 = dht.get_averagers('bucket.0b10110', only_active=True)
    q4 = dht.get_averagers('bucket.0b10110', only_active=False)

    assert len(q1) == 2 and ('localhvost', t + 60) in q1 and ('localhvost2', t + 61) in q1
    assert len(q2) == 2 and ('localhvost', t + 66) in q2 and ('localhvost2', t + 61) in q2
    assert len(q3) == 1 and ('localhvost', t + 66) in q3
    assert len(q4) == 2 and ('localhvost', t + 66) in q4 and ('localhvost2', t + 61) in q2

