import asyncio
import random

import torch
import pytest
import hivemind
from hivemind.client.allreduce import GroupAllReduce, split_into_chunks, restore_from_chunks


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
        chunks = split_into_chunks(tensors, group_size=num_chunks)
        assert len(chunks) == num_chunks
        shapes = [tensor.shape for tensor in tensors]
        restored = restore_from_chunks(chunks, shapes)
        assert len(restored) == len(tensors)
        assert all(new.shape == old.shape for new, old in zip(restored, tensors))
        assert all(torch.allclose(new, old) for new, old in zip(restored, tensors))
