import asyncio
import random

import pytest
import hivemind
from hivemind.client.averaging import RunningAllReduce, split_into_chunks, restore_from_chunks
import torch


def test_chunks():
    for i in range(100_000):
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
        chunks = split_into_chunks(tensors, group_size=num_chunks)
        assert len(chunks) == num_chunks
        shapes = [tensor.shape for tensor in tensors]
        restored = restore_from_chunks(chunks, shapes)
        assert len(restored) == len(tensors)
        assert all(new.shape == old.shape for new, old in zip(restored, tensors))
        assert all(torch.allclose(new, old) for new, old in zip(restored, tensors))


@pytest.mark.asyncio
async def test_allreduce_state():
    allreduce_state = RunningAllReduce(my_endpoint='alice', group_endpoints={'alice', 'bob', 'carol'}, part_index=3)

    x, y, z = torch.randn(3, 128)

    results = await asyncio.gather(
        allreduce_state.accumulate('alice', x),
        allreduce_state.accumulate('bob', y),
        allreduce_state.accumulate('carol', z),
    )

    with pytest.raises(AssertionError):
        await allreduce_state.accumulate('carol', z),

    with pytest.raises(AssertionError):
        await allreduce_state.accumulate('mallory', z),

    ref = (x + y + z) / 3
    for tensor in results:
        assert torch.allclose(tensor, ref)

    allreduce_state = RunningAllReduce(my_endpoint='alice', group_endpoints={'alice'}, part_index=1337)
    assert torch.allclose(x, await allreduce_state.accumulate('alice', x))

    with pytest.raises(AssertionError):
        allreduce_state = RunningAllReduce(my_endpoint='alice', group_endpoints={'bob', 'carol'}, part_index=1337)
        assert torch.allclose(x, await allreduce_state.accumulate('alice', x))

    with pytest.raises(AssertionError):
        allreduce_state = RunningAllReduce(my_endpoint='alice', group_endpoints=set(), part_index=1337)
        assert torch.allclose(x, await allreduce_state.accumulate('alice', x))
