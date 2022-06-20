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
from hivemind.optim.grad_averager import GradientAverager, GradientAveragerFactory
from hivemind.optim.optimizer import Optimizer
from hivemind.optim.power_sgd_averager import PowerSGDGradientAverager
from hivemind.optim.progress_tracker import ProgressTracker
from hivemind.optim.state_averager import TrainingStateAverager
from hivemind.utils.crypto import RSAPrivateKey


@pytest.mark.forked
@pytest.mark.parametrize(
    "grad_averager_factory",
    [GradientAverager, partial(PowerSGDGradientAverager, averager_rank=1)],
)
def test_grad_averager(grad_averager_factory: GradientAveragerFactory):
    parameter_shape = (5, 5)

    dht1 = hivemind.DHT(start=True)
    model1 = nn.ParameterDict({"w": nn.Parameter(torch.zeros(parameter_shape))})
    averager1 = grad_averager_factory(
        model1.parameters(), dht=dht1, prefix="test", target_group_size=2, reuse_grad_buffers=False, start=True
    )

    dht2 = hivemind.DHT(start=True, initial_peers=dht1.get_visible_maddrs())
    model2 = nn.ParameterDict({"w": nn.Parameter(torch.zeros(parameter_shape))})
    averager2 = grad_averager_factory(
        model2.parameters(), dht=dht2, prefix="test", target_group_size=2, reuse_grad_buffers=True, start=True
    )

    control1 = averager1.schedule_step(hivemind.get_dht_time() + 5)
    control2 = averager2.schedule_step(hivemind.get_dht_time() + 5)

    for i in range(10):
        time.sleep(0.1)
        if i % 3 == 0:
            loss1 = F.mse_loss(model1.w, torch.ones(parameter_shape))
            loss1.backward()
            averager1.accumulate_grads_(batch_size=2)  # total: 4 times * 2 samples = 8
            model1.zero_grad()
        else:
            loss2 = F.mse_loss(model2.w, -torch.ones(parameter_shape))
            loss2.backward()
            averager2.accumulate_grads_(batch_size=3)  # total: 6 times * 3 samples = 18
            # note: we do not call zero grad here because reuse_grad_buffers=True

    assert control1.stage == control2.stage == AveragingStage.AWAITING_TRIGGER
    peer1_samples, peer1_times, peer2_samples, peer2_times = 8, 4, 18, 6
    assert averager1.local_samples_accumulated == peer1_samples and averager1.local_times_accumulated == peer1_times
    ref_grads1 = torch.full(parameter_shape, -2 / np.prod(parameter_shape) * averager1.local_times_accumulated)
    assert torch.allclose(next(averager1._grad_accumulators()), ref_grads1)

    assert averager2.local_samples_accumulated == peer2_samples and averager2.local_times_accumulated == peer2_times
    ref_grads2 = torch.full(parameter_shape, 2 / np.prod(parameter_shape) * averager2.local_times_accumulated)
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
    "grad_averager_factory",
    [GradientAverager, partial(PowerSGDGradientAverager, averager_rank=1)],
)
def test_grad_averager_wrong_shape(grad_averager_factory: GradientAveragerFactory):
    parameter_shape = (5, 5)
    model = nn.ParameterDict({"w": nn.Parameter(torch.zeros(parameter_shape))})
    dht = hivemind.DHT(start=True)

    with pytest.raises(ValueError):
        grad_averager_factory(
            model.parameters(),
            dht=dht,
            prefix="test_fail",
            target_group_size=2,
            reuse_grad_buffers=False,
            start=True,
            averaged_grads=[torch.zeros(parameter_shape + (1,))],
        )


@pytest.mark.forked
@pytest.mark.parametrize(
    "offload_optimizer, reuse_tensors, sync_epoch_when_averaging",
    [(False, False, False), (True, True, False), (True, False, False), (False, True, True), (True, False, True)],
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

    avgr1.step(wait_for_delayed_updates=True)
    avgr2.step(wait_for_delayed_updates=True)

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
        dht=dht1,
        params=model1.parameters(),
        allow_state_sharing=False,
        start=True,
        **common_kwargs,
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
    for i in (0, 1, 5):  # Without the 4th worker (the fastest one)
        assert 1.05 * mean_step_time < step_time_deltas[i] < 2.0 * mean_step_time
    for i in (2, 3, 4):  # With the 4th worker
        assert 0.5 * mean_step_time < step_time_deltas[i] < 0.95 * mean_step_time
    assert emas[1] < emas[2] < emas[3] < emas[4]
    assert tracker.performance_ema.samples_per_second < 1e-9


@pytest.mark.forked
@pytest.mark.parametrize(
    "use_local_updates, delay_state_averaging, delay_optimizer_step, delay_grad_averaging, reuse_grad_buffers",
    # fmt: off
    [
        (False, False, False, False, False),
        (False, True, False, False, False),
        (False, True, True, True, False),
        (False, False, False, False, True),
        (False, True, True, True, True),
        (False, True, True, False, True),
        (True, False, False, False, False),
        (True, True, False, False, False,),
    ],
    # fmt: on
)
def test_optimizer(
    use_local_updates: bool,
    delay_state_averaging: bool,
    delay_optimizer_step: bool,
    delay_grad_averaging: bool,
    reuse_grad_buffers: bool,
):
    _test_optimizer(
        use_local_updates=use_local_updates,
        delay_state_averaging=delay_state_averaging,
        delay_grad_averaging=delay_grad_averaging,
        delay_optimizer_step=delay_optimizer_step,
        reuse_grad_buffers=reuse_grad_buffers,
    )


def _test_optimizer(
    num_peers: int = 1,
    num_clients: int = 0,
    target_batch_size: int = 32,
    total_epochs: int = 3,
    use_local_updates: bool = False,
    reuse_grad_buffers: bool = True,
    delay_state_averaging: bool = True,
    delay_grad_averaging: bool = True,
    delay_optimizer_step: bool = True,
    average_state_every: int = 1,
):
    dht = hivemind.DHT(start=True)

    features = torch.randn(100, 5)
    targets = features @ torch.randn(5, 1)
    optimizer = None
    total_samples_accumulated = mp.Value(ctypes.c_int32, 0)

    def run_trainer(batch_size: int, batch_time: float, client_mode: bool):
        nonlocal optimizer
        model = nn.Linear(5, 1)

        assert isinstance(model, torch.nn.Module), "model_arch must evaluate to a pytorch module"

        optimizer = Optimizer(
            run_id="test_run",
            target_batch_size=target_batch_size,
            batch_size_per_step=batch_size,
            params=model.parameters(),
            optimizer=partial(torch.optim.SGD, lr=0.1),
            scheduler=partial(torch.optim.lr_scheduler.StepLR, gamma=0.5, step_size=1),
            dht=hivemind.DHT(initial_peers=dht.get_visible_maddrs(), client_mode=client_mode, start=True),
            tracker_opts=dict(private_key=RSAPrivateKey(), max_refresh_period=1.0),
            averager_opts=dict(request_timeout=0.5),
            use_local_updates=use_local_updates,
            matchmaking_time=1.0,
            averaging_timeout=5.0,
            reuse_grad_buffers=reuse_grad_buffers,
            delay_state_averaging=delay_state_averaging,
            delay_grad_averaging=delay_grad_averaging,
            delay_optimizer_step=delay_optimizer_step,
            average_state_every=average_state_every,
            client_mode=client_mode,
            verbose=False,
        )
        optimizer.load_state_from_peers()

        prev_time = time.perf_counter()

        while optimizer.local_epoch < total_epochs:
            time.sleep(max(0.0, prev_time + batch_time - time.perf_counter()))
            batch = torch.randint(0, len(features), (batch_size,))

            loss = F.mse_loss(model(features[batch]), targets[batch])
            loss.backward()

            optimizer.step()

            total_samples_accumulated.value += batch_size

            if not reuse_grad_buffers:
                optimizer.zero_grad()

            prev_time = time.perf_counter()

        time.sleep(1.0)
        optimizer.shutdown()
        return optimizer

    peers = []

    for index in range(num_peers):
        peers.append(
            mp.Process(
                target=run_trainer,
                name=f"trainer-{index}",
                kwargs=dict(
                    batch_size=4 + index,
                    batch_time=0.3 + 0.2 * index,
                    client_mode=(index >= num_peers - num_clients),
                ),
            )
        )

    for peer in peers[1:]:
        peer.start()
    peers[0].run()
    for peer in peers[1:]:
        peer.join()

    assert isinstance(optimizer, Optimizer)
    assert optimizer.local_epoch == optimizer.tracker.global_epoch == total_epochs
    expected_samples_accumulated = target_batch_size * total_epochs
    assert expected_samples_accumulated <= total_samples_accumulated.value <= expected_samples_accumulated * 1.2
    assert 4 / 0.3 * 0.8 <= optimizer.tracker.performance_ema.samples_per_second <= 4 / 0.3 * 1.2

    assert not optimizer.state_averager.is_alive()
    assert not optimizer.tracker.is_alive()
    if not use_local_updates:
        assert not optimizer.grad_averager.is_alive()
    else:
        assert optimizer.grad_averager is None

    assert optimizer.scheduled_grads is None or optimizer.scheduled_grads.done()
