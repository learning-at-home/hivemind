import multiprocessing as mp
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import hivemind
from hivemind.averaging.control import AveragingStage
from hivemind.optim.experimental.grad_averager import GradientAverager


@pytest.mark.forked
def test_grad_averager():
    dht1 = hivemind.DHT(start=True)
    model1 = nn.ParameterDict(dict(w=nn.Parameter(torch.zeros(3))))
    averager1 = GradientAverager(
        model1.parameters(), dht=dht1, prefix="test", target_group_size=2, reuse_grad_buffers=False, start=True
    )

    dht2 = hivemind.DHT(start=True, initial_peers=dht1.get_visible_maddrs())
    model2 = nn.ParameterDict(dict(w=nn.Parameter(torch.zeros(3))))
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
    assert averager1.local_samples_accumulated == 8 and averager1.local_times_accumulated == 4
    ref_grads1 = torch.full((3,), -2 * 1 / 3 * averager1.local_times_accumulated)
    assert torch.allclose(next(averager1._grad_acumulators()), ref_grads1)

    assert averager2.local_samples_accumulated == 18 and averager2.local_times_accumulated == 6
    ref_grads2 = torch.full((3,), 2 * 1 / 3 * averager2.local_times_accumulated)
    assert torch.allclose(next(averager2._grad_acumulators()), ref_grads2)

    averager1.step(control=control1, wait=False)
    averager2.step(control=control2, wait=False)
    for step in (control1, control2):
        step.result()  # wait for all-reduce to finish

    ref_average = 8 / 26 * (ref_grads1 / 4) + 18 / 26 * (ref_grads2 / 6)
    with averager1.use_averaged_gradients():
        assert torch.allclose(model1.w.grad, ref_average)
    with averager2.use_averaged_gradients():
        assert torch.allclose(model2.w.grad, ref_average)

    # after no longer use_averaged_gradients
    assert not torch.allclose(model1.w.grad, ref_average)
    assert not torch.allclose(model2.w.grad, ref_average)
