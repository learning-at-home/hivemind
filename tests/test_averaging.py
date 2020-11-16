import asyncio
import random
import time
from itertools import product

import torch
import pytest
import hivemind
from hivemind.client.allreduce import GroupAllReduce, split_into_parts, restore_from_parts
from hivemind.utils import LOCALHOST


@pytest.mark.forked
@pytest.mark.asyncio
async def test_allreduce_direct():
    # WARNING! this test uses an early interface that will change by the time DecentralizedAverager is finished

    dht = hivemind.DHT(start=True)

    tensors1 = [torch.randn(123), torch.zeros(3)]
    tensors2 = [torch.rand(123), torch.ones(3)]
    tensors3 = [-torch.rand(123), torch.arange(3).to(torch.float32)]

    reference = [(tensors1[i] + tensors2[i] + tensors3[i]) / 3 for i in range(len(tensors1))]

    averager1 = hivemind.DecentralizedAverager(tensors1, dht=dht, start=True, max_size=3, timeout=5)
    averager2 = hivemind.DecentralizedAverager(tensors2, dht=dht, start=True, max_size=3, timeout=5)
    averager3 = hivemind.DecentralizedAverager(tensors3, dht=dht, start=True, max_size=3, timeout=5)

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


@pytest.mark.asyncio
async def test_allreduce_protocol():
    """ Run group allreduce protocol manually without grpc, see if the internal logic is working as intended """
    peers = "alice", "bob", "carol"
    expiration_offsets = 4, 0, 1

    tensors_by_peer = {peer: [torch.randn(3, 128), torch.rand(32), torch.tensor(i, dtype=torch.float32)]
                       for i, peer in enumerate(peers)}

    alice, bob, carol = allreduce_protocols = [
        GroupAllReduce(my_endpoint=peer, expiration=hivemind.get_dht_time() + offset, my_tensors=tensors_by_peer[peer])
        for peer, offset in zip(peers, expiration_offsets)]

    bob.start_new_group()
    bob.add_peer_to_group(alice.info.endpoint)
    alice.join_group(bob, bob.group_id)
    bob.add_peer_to_group(carol.info.endpoint)
    carol.join_group(carol, bob.group_id)

    bob.leader_begin_allreduce()
    ordered_group_endpoints = await bob.group_assembled
    assert len(ordered_group_endpoints) == len(peers)

    carol.follower_begin_allreduce(ordered_group_endpoints)
    alice.follower_begin_allreduce(ordered_group_endpoints)

    chunks_by_peer = {protocol.info.endpoint: {
        peer: part for peer, part in zip(peers, split_into_parts(protocol.local_tensors, len(ordered_group_endpoints)))
    } for protocol in allreduce_protocols}

    all_pairs = list(product(allreduce_protocols, peers))
    random.shuffle(all_pairs)
    await asyncio.gather(*(
        peer_allreduce.accumulate(source_peer, chunks_by_peer[source_peer][peer_allreduce.info.endpoint])
        for peer_allreduce, source_peer in all_pairs))

    averaged_parts = await asyncio.gather(*(protocol.averaged_part for protocol in allreduce_protocols))
    tensor_shapes = [tensor.shape for tensor in alice.local_tensors]
    averaged_tensors = restore_from_parts(averaged_parts, tensor_shapes)

    reference_tensors = [
        sum(tensors_by_peer[peer][i] for peer in peers) / len(peers)
        for i in range(len(tensors_by_peer[peers[0]]))
    ]

    assert len(averaged_tensors) == len(reference_tensors)
    assert all(map(torch.allclose, averaged_tensors, reference_tensors))


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
