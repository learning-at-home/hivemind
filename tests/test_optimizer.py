import time
from functools import partial

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import hivemind
from hivemind.averaging.control import AveragingStage
from hivemind.optim.experimental.grad_averager import GradientAverager
from hivemind.optim.experimental.state_averager import TrainingStateAverager


@pytest.mark.forked
def test_grad_averager():
    dht1 = hivemind.DHT(start=True)
    model1 = nn.ParameterDict({"w": nn.Parameter(torch.zeros(3))})
    averager1 = GradientAverager(
        model1.parameters(), dht=dht1, prefix="test", target_group_size=2, reuse_grad_buffers=False, start=True
    )

    dht2 = hivemind.DHT(start=True, initial_peers=dht1.get_visible_maddrs())
    model2 = nn.ParameterDict({"w": nn.Parameter(torch.zeros(3))})
    averager2 = GradientAverager(
        model2.parameters(), dht=dht2, prefix="test", target_group_size=2, reuse_grad_buffers=True, start=True
    )

    control1 = averager1.schedule_step(hivemind.get_dht_time() + 5)
    control2 = averager2.schedule_step(hivemind.get_dht_time() + 5)

    for i in range(10):
        time.sleep(0.1)
        if i % 3 == 0:
            loss1 = F.mse_loss(model1.w, torch.ones(3))
            loss1.backward()
            averager1.accumulate_grads_(batch_size=2)  # total: 4 times * 2 samples = 8
            model1.zero_grad()
        else:
            loss2 = F.mse_loss(model2.w, -torch.ones(3))
            loss2.backward()
            averager2.accumulate_grads_(batch_size=3)  # total: 6 times * 3 samples = 18
            # note: we do not call zero grad here because reuse_grad_buffers=True

    assert control1.stage == control2.stage == AveragingStage.AWAITING_TRIGGER
    peer1_samples, peer1_times, peer2_samples, peer2_times = 8, 4, 18, 6
    assert averager1.local_samples_accumulated == peer1_samples and averager1.local_times_accumulated == peer1_times
    ref_grads1 = torch.full((3,), -2 * 1 / 3 * averager1.local_times_accumulated)
    assert torch.allclose(next(averager1._grad_accumulators()), ref_grads1)

    assert averager2.local_samples_accumulated == peer2_samples and averager2.local_times_accumulated == peer2_times
    ref_grads2 = torch.full((3,), 2 * 1 / 3 * averager2.local_times_accumulated)
    assert torch.allclose(next(averager2._grad_accumulators()), ref_grads2)

    averager1.step(control=control1, wait=False)
    averager2.step(control=control2, wait=False)
    for step in (control1, control2):
        step.result()  # wait for all-reduce to finish

    peer1_weight = peer1_samples / (peer1_samples + peer2_samples)
    peer2_weight = peer2_samples / (peer1_samples + peer2_samples)
    ref_average = peer1_weight * (ref_grads1 / peer1_times) + peer2_weight * (ref_grads2 / peer2_times)
    with averager1.use_averaged_gradients():
        assert torch.allclose(model1.w.grad, ref_average)
    with averager2.use_averaged_gradients():
        assert torch.allclose(model2.w.grad, ref_average)

    # after no longer use_averaged_gradients
    assert not torch.allclose(model1.w.grad, ref_average)
    assert not torch.allclose(model2.w.grad, ref_average)


@pytest.mark.forked
def test_load_state_from_peers():
    dht1 = hivemind.DHT(start=True)
    dht2 = hivemind.DHT(initial_peers=dht1.get_visible_maddrs(), start=True)

    model1 = nn.Linear(2, 3)
    model2 = nn.Linear(2, 3)

    avgr1 = TrainingStateAverager(
        dht=dht1,
        param_groups=model1.parameters(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
        scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)),
        allow_state_sharing=False,
        target_group_size=2,
        prefix="my_exp",
        start=True,
    )

    avgr2 = TrainingStateAverager(
        dht=dht2,
        param_groups=model2.parameters(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
        scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)),
        target_group_size=2,
        prefix="my_exp",
        start=True,
    )

    avgr2.local_epoch = 1337
    model2.weight.data[...] = 42
    time.sleep(0.1)

    avgr1.load_state_from_peers()
    assert avgr1.local_epoch == 1337
    assert torch.all(model1.weight == 42).item()
    assert np.allclose(avgr1.optimizer.param_groups[0]["lr"], 0.1 / 1337)
