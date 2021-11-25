import ctypes
import multiprocessing as mp
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
from hivemind.optim.experimental.progress_tracker import ProgressTracker
from hivemind.optim.experimental.state_averager import TrainingStateAverager
from hivemind.utils.crypto import RSAPrivateKey


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
@pytest.mark.parametrize(
    "offload_optimizer, reuse_tensors, sync_epoch_when_averaging",
    [(False, False, False), (True, False, False), (False, True, True), (True, False, True)],
)
def test_state_averager(offload_optimizer: bool, reuse_tensors: bool, sync_epoch_when_averaging: bool):
    dht1 = hivemind.DHT(start=True)
    dht2 = hivemind.DHT(initial_peers=dht1.get_visible_maddrs(), start=True)

    torch.manual_seed(1337)
    torch.use_deterministic_algorithms(True)
    # note: use_deterministic_algorithms does not affect further tests because this test is forked

    model1 = nn.Linear(2, 3)
    model2 = nn.Linear(2, 3)

    extras1 = (torch.randn(2, 2), -torch.rand(1))
    extras2 = (-torch.randn(2, 2), torch.rand(1))

    common_kwargs = dict(
        optimizer=partial(torch.optim.Adam, lr=0.1, betas=(0.9, 0.9)),
        scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)),
        sync_epoch_when_averaging=sync_epoch_when_averaging,
        average_opt_statistics=("exp_avg_sq",),
        offload_optimizer=offload_optimizer,
        reuse_tensors=reuse_tensors,
        target_group_size=2,
        prefix="my_exp",
    )

    avgr1 = TrainingStateAverager(
        dht=dht1, params=model1.parameters(), extra_tensors=extras1, start=True, **common_kwargs
    )
    avgr2 = TrainingStateAverager(
        dht=dht2, params=model2.parameters(), extra_tensors=extras2, start=True, **common_kwargs
    )

    x = torch.ones(2)

    for step in range(20):
        F.mse_loss(model1(x), torch.ones(3)).mul(2).backward()
        avgr1.step(optimizer_step=True, zero_grad=True, averaging_round=(step == 10), delay_averaging=True)

        F.mse_loss(model2(x), -torch.ones(3)).backward()
        avgr2.step(optimizer_step=True, zero_grad=True, averaging_round=(step == 10), delay_averaging=False)

    assert torch.all(model1.weight.grad == 0) and torch.all(model2.weight.grad == 0), "zero grad did not trigger"
    assert model1(x).mean() > 0.5 and model2(x).mean() < -0.5, "models did not train properly"
    assert torch.allclose(extras1[0], extras2[0]), "first extra tensors were not averaged"
    assert torch.allclose(extras1[1], extras2[1]), "second extra tensors were not averaged"

    stats1 = avgr1.optimizer.state_dict()["state"][0]["exp_avg_sq"].clone()
    stats2 = avgr2.optimizer.state_dict()["state"][0]["exp_avg_sq"].clone()
    assert not torch.allclose(stats1, stats2)

    avgr1.step(increment_epoch=True)

    avgr1.step(increment_epoch=True, averaging_round=True, delay_averaging=True)
    avgr2.step(increment_epoch=True, averaging_round=True, delay_averaging=True)

    avgr1.step(wait_for_delayed_update=True)
    avgr2.step(wait_for_delayed_update=True)

    assert torch.allclose(model1(x), model2(x)), "model parameters were not averaged correctly"
    assert torch.allclose(avgr1.optimizer.state_dict()["state"][0]["exp_avg_sq"], (stats1 + stats2) / 2)
    assert torch.allclose(avgr2.optimizer.state_dict()["state"][0]["exp_avg_sq"], (stats1 + stats2) / 2)
    assert avgr1.local_epoch == 2
    assert avgr2.local_epoch == (2 if sync_epoch_when_averaging else 1)


@pytest.mark.forked
def test_load_state_from_peers():
    dht1 = hivemind.DHT(start=True)
    dht2 = hivemind.DHT(initial_peers=dht1.get_visible_maddrs(), start=True)

    model1 = nn.Linear(2, 3)
    model2 = nn.Linear(2, 3)

    common_kwargs = dict(
        optimizer=partial(torch.optim.SGD, lr=0.1),
        scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)),
        target_group_size=2,
        prefix="my_exp",
    )

    avgr1 = TrainingStateAverager(
        dht=dht1, params=model1.parameters(), allow_state_sharing=False, start=True, **common_kwargs
    )

    avgr2 = TrainingStateAverager(dht=dht2, params=model2.parameters(), start=True, **common_kwargs)

    avgr2.local_epoch = 1337
    model2.weight.data[...] = 42
    time.sleep(0.1)

    avgr1.load_state_from_peers()
    assert avgr1.local_epoch == 1337
    assert torch.all(model1.weight == 42).item()
    assert np.allclose(avgr1.optimizer.param_groups[0]["lr"], 0.1 / 1337)


@pytest.mark.forked
def test_progress_tracker():
    # note to a curious reader: no, you cannot reduce the timings without compromising realism or stability
    prefix = "my_exp"
    target_batch_size = 256
    dht_root = hivemind.DHT(start=True)
    barrier = mp.Barrier(parties=5)
    delayed_start_evt = mp.Event()
    finished_evt = mp.Event()
    emas = mp.Array(ctypes.c_double, 5)

    def run_worker(index: int, batch_size: int, period: float, **kwargs):
        dht = hivemind.DHT(initial_peers=dht_root.get_visible_maddrs(), start=True)
        tracker = ProgressTracker(
            dht,
            prefix,
            target_batch_size,
            start=True,
            min_refresh_period=0.1,
            default_refresh_period=0.2,
            max_refresh_period=0.5,
            private_key=RSAPrivateKey(),
            **kwargs,
        )

        barrier.wait()
        if index == 4:
            delayed_start_evt.wait()

        local_epoch = 2 if index == 4 else 0
        samples_accumulated = 0

        while True:
            time.sleep(period)
            if finished_evt.is_set():
                break

            samples_accumulated += batch_size
            tracker.report_local_progress(local_epoch, samples_accumulated)

            if tracker.ready_to_update_epoch:
                if index == 4 and local_epoch >= 4:
                    time.sleep(0.5)
                    break

                with tracker.pause_updates():
                    local_epoch = tracker.update_epoch(local_epoch + 1)
                    samples_accumulated = 0


        emas[index] = tracker.performance_ema.samples_per_second
        tracker.shutdown()
        dht.shutdown()

    workers = [
        mp.Process(target=run_worker, kwargs=dict(index=1, batch_size=12, period=0.6)),
        mp.Process(target=run_worker, kwargs=dict(index=2, batch_size=16, period=0.5)),
        mp.Process(target=run_worker, kwargs=dict(index=3, batch_size=24, period=0.4)),
        mp.Process(target=run_worker, kwargs=dict(index=4, batch_size=64, period=0.4)),
    ]
    for worker in workers:
        worker.start()

    tracker = ProgressTracker(
        dht_root,
        prefix,
        target_batch_size,
        start=True,
        min_refresh_period=0.1,
        default_refresh_period=0.2,
        max_refresh_period=0.5,
    )
    barrier.wait()

    local_epoch = 0
    last_timestamp = hivemind.get_dht_time()
    step_time_deltas = []

    while local_epoch < 6:
        time.sleep(0.1)

        if tracker.ready_to_update_epoch:
            with tracker.pause_updates():
                local_epoch = tracker.update_epoch(local_epoch + 1)

            time_delta = hivemind.get_dht_time() - last_timestamp
            if local_epoch == 2:
                delayed_start_evt.set()

            last_timestamp = hivemind.get_dht_time()
            step_time_deltas.append(time_delta)

    finished_evt.set()
    for worker in workers:
        worker.join()

    tracker.shutdown()
    dht_root.shutdown()
    assert not tracker.is_alive()

    mean_step_time = sum(step_time_deltas) / len(step_time_deltas)
    print(step_time_deltas, mean_step_time)
    for i in (0, 1, 5):  # Without the 4th worker (the fastest one)
        assert 1.05 * mean_step_time < step_time_deltas[i] < 2.0 * mean_step_time
    for i in (2, 3, 4):  # With the 4th worker
        assert 0.5 * mean_step_time < step_time_deltas[i] < 0.95 * mean_step_time
    assert emas[1] < emas[2] < emas[3] < emas[4]
    assert tracker.performance_ema.samples_per_second < 1e-9
