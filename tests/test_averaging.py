import random
import time

import numpy as np
import pytest
import torch

import hivemind
from hivemind.averaging import DecentralizedAverager
from hivemind.averaging.allreduce import AveragingMode
from hivemind.averaging.control import AveragingStage
from hivemind.averaging.key_manager import GroupKeyManager
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.partition import AllreduceException
from hivemind.p2p import PeerID

from test_utils.dht_swarms import launch_dht_instances


@pytest.mark.forked
@pytest.mark.asyncio
async def test_key_manager():
    dht = hivemind.DHT(start=True)
    key_manager = GroupKeyManager(
        dht,
        prefix="test_averaging",
        initial_group_bits="10110",
        target_group_size=2,
    )
    alice = dht.peer_id
    bob = PeerID(b"bob")

    t = hivemind.get_dht_time()
    key = key_manager.current_key
    await key_manager.declare_averager(key, alice, expiration_time=t + 60)
    await key_manager.declare_averager(key, bob, expiration_time=t + 61)

    q1 = await key_manager.get_averagers(key, only_active=True)

    await key_manager.declare_averager(key, alice, expiration_time=t + 66)
    q2 = await key_manager.get_averagers(key, only_active=True)

    await key_manager.declare_averager(key, bob, expiration_time=t + 61, looking_for_group=False)
    q3 = await key_manager.get_averagers(key, only_active=True)
    q4 = await key_manager.get_averagers(key, only_active=False)

    q5 = await key_manager.get_averagers("nonexistent_key.0b0101", only_active=False)

    assert len(q1) == 2 and (alice, t + 60) in q1 and (bob, t + 61) in q1
    assert len(q2) == 2 and (alice, t + 66) in q2 and (bob, t + 61) in q2
    assert len(q3) == 1 and (alice, t + 66) in q3
    assert len(q4) == 2 and (alice, t + 66) in q4 and (bob, t + 61) in q2
    assert len(q5) == 0

    dht.shutdown()


def _test_allreduce_once(n_clients, n_aux):
    n_peers = 4
    modes = (
        [AveragingMode.CLIENT] * n_clients
        + [AveragingMode.AUX] * n_aux
        + [AveragingMode.NODE] * (n_peers - n_clients - n_aux)
    )
    random.shuffle(modes)

    tensors1 = [torch.randn(123), torch.zeros(3)]
    tensors2 = [torch.rand(123), torch.ones(3)]
    tensors3 = [-torch.rand(123), torch.arange(3).to(torch.float32)]
    tensors4 = [torch.randn(123) ** 3, torch.arange(3).to(torch.float32) / 2]
    peer_tensors = [tensors1, tensors2, tensors3, tensors4]

    reference = [
        sum(tensors[i] for tensors, mode in zip(peer_tensors, modes) if mode != AveragingMode.AUX)
        / max(1, n_peers - n_aux)
        for i in range(len(tensors1))
    ]

    dht_instances = launch_dht_instances(len(peer_tensors))
    averagers = [
        DecentralizedAverager(
            tensors,
            dht=dht,
            target_group_size=4,
            min_matchmaking_time=15,
            prefix="mygroup",
            client_mode=mode == AveragingMode.CLIENT,
            auxiliary=mode == AveragingMode.AUX,
            start=True,
        )
        for tensors, dht, mode in zip(peer_tensors, dht_instances, modes)
    ]

    futures = []
    for averager in averagers:
        futures.append(averager.step(wait=False))
    for future in futures:
        result = future.result()
        for averager in averagers:
            assert averager.peer_id in result

    for averager in averagers:
        if averager.mode != AveragingMode.AUX:
            with averager.get_tensors() as averaged_tensors:
                for ref, our in zip(reference, averaged_tensors):
                    assert torch.allclose(ref, our, atol=1e-6)

    for process in averagers + dht_instances:
        process.shutdown()


@pytest.mark.forked
@pytest.mark.parametrize("n_clients", [0, 1, 2])
@pytest.mark.parametrize("n_aux", [0, 1, 2])
def test_allreduce_once(n_clients, n_aux):
    _test_allreduce_once(n_clients, n_aux)


@pytest.mark.forked
@pytest.mark.parametrize("n_clients, n_aux", [(0, 4), (1, 3), (0, 3)])
def test_allreduce_once_edge_cases(n_clients, n_aux):
    _test_allreduce_once(n_clients, n_aux)


@pytest.mark.forked
def test_allreduce_weighted(n_client_mode_peers: int = 2):
    n_peers = 4
    client_modes = [True] * n_client_mode_peers + [False] * (n_peers - n_client_mode_peers)
    random.shuffle(client_modes)

    tensors1 = [torch.randn(123), torch.zeros(3)]
    tensors2 = [torch.rand(123), torch.ones(3)]
    tensors3 = [-torch.rand(123), torch.arange(3).to(torch.float32)]
    tensors4 = [torch.randn(123) ** 3, torch.arange(3).to(torch.float32) / 2]

    dht_instances = launch_dht_instances(4)
    averagers = [
        DecentralizedAverager(
            tensors,
            dht=dht,
            target_group_size=4,
            min_matchmaking_time=15,
            prefix="mygroup",
            client_mode=client_mode,
            start=True,
        )
        for tensors, dht, client_mode in zip([tensors1, tensors2, tensors3, tensors4], dht_instances, client_modes)
    ]

    weights = list(map(float, np.random.rand(len(averagers)) * 10 + 0.01))
    reference = [
        (tensors1[i] * weights[0] + tensors2[i] * weights[1] + tensors3[i] * weights[2] + tensors4[i] * weights[3])
        / sum(weights)
        for i in range(len(tensors1))
    ]

    futures = []
    for averager, weight in zip(averagers, weights):
        futures.append(averager.step(weight=weight, wait=False))
    for future in futures:
        future.result()

    for future, averager in zip(futures, averagers):
        with averager.get_tensors() as averaged_tensors:
            for ref, our in zip(reference, averaged_tensors):
                assert torch.allclose(ref, our, atol=1e-6)

    for process in averagers + dht_instances:
        process.shutdown()


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
    dht_instances = launch_dht_instances(8)
    averagers = [
        DecentralizedAverager(
            averaged_tensors=[torch.randn(3)],
            dht=dht,
            target_group_size=2,
            prefix="mygroup",
            initial_group_bits=bin(i // 2)[2:].rjust(2, "0"),
            start=True,
        )
        for i, dht in enumerate(dht_instances)
    ]

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

    for process in averagers + dht_instances:
        process.shutdown()


@pytest.mark.forked
def test_allgather(n_averagers=8, target_group_size=4):
    dht_instances = launch_dht_instances(n_averagers)
    averagers = [
        DecentralizedAverager(
            [torch.ones(1)],
            dht=dht,
            target_group_size=target_group_size,
            min_matchmaking_time=15,
            prefix="mygroup",
            initial_group_bits="000",
            start=True,
        )
        for dht in dht_instances
    ]

    futures = []
    for i, averager in enumerate(averagers):
        futures.append(averager.step(wait=False, gather=dict(batch_size=123 + i, foo="bar")))

    reference_metadata = {
        averager.peer_id: dict(batch_size=123 + i, foo="bar") for i, averager in enumerate(averagers)
    }
    for future in futures:
        gathered = future.result()
        assert len(gathered) == target_group_size
        for peer_id in gathered:
            assert gathered[peer_id] == reference_metadata[peer_id]

    for process in averagers + dht_instances:
        process.shutdown()


def get_cost(vector_size, partitions, bandwidths):
    return max(
        (vector_size - partitions[i] + (len(partitions) - 1) * partitions[i]) / max(bandwidths[i], 1e-9)
        for i in range(len(partitions))
    )


def check_optimality(vector_size, bandwidths, ref_partitions):
    partitions = list(load_balance_peers(vector_size, bandwidths))
    assert get_cost(vector_size, partitions, bandwidths) <= get_cost(vector_size, ref_partitions, bandwidths)


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
        vector_size = np.random.randint(1, 1024**3)
        num_peers = np.random.randint(1, 256)
        scale = 1e-9 + np.random.rand() * 1e5
        bandwidths = np.random.rand(num_peers) * scale + 1e-6
        min_size = np.random.choice([0, np.random.randint(0, vector_size // 10)])
        assignment = load_balance_peers(vector_size, bandwidths, min_size)
        assert np.sum(assignment) == vector_size
        assert np.min(assignment) >= 0


@pytest.mark.forked
def test_too_few_peers():
    dht_instances = launch_dht_instances(4)
    averagers = [
        DecentralizedAverager(
            averaged_tensors=[torch.randn(3)],
            dht=dht,
            target_group_size=2,
            min_matchmaking_time=1,
            request_timeout=0.5,
            prefix="mygroup",
            initial_group_bits=bin(i)[2:].rjust(3, "0"),
            start=True,
        )
        for i, dht in enumerate(dht_instances)
    ]
    step_futures = [averager.step(wait=False, timeout=2) for averager in averagers]

    for future in step_futures:
        with pytest.raises(AllreduceException):
            future.result()

    for process in averagers + dht_instances:
        process.shutdown()


@pytest.mark.skip(
    reason="The current implementation of elasticity (multi-stage averaging when num_peers > ~3 * target_group_size) "
    "is incorrect (TODO @justheuristic)"
)
@pytest.mark.forked
def test_overcrowded(num_peers=16):
    dht_instances = launch_dht_instances(num_peers)
    averagers = [
        DecentralizedAverager(
            averaged_tensors=[torch.randn(3)],
            dht=dht,
            target_group_size=2,
            min_matchmaking_time=1,
            request_timeout=0.5,
            prefix="mygroup",
            initial_group_bits="",
            start=True,
        )
        for dht in dht_instances
    ]
    for _ in range(5):
        step_futures = [averager.step(wait=False, timeout=5) for averager in averagers]
        assert sum(len(future.result() or []) == 2 for future in step_futures) >= len(averagers) - 1

    for process in averagers + dht_instances:
        process.shutdown()


@pytest.mark.forked
def test_load_state_from_peers():
    num_calls = 0
    super_metadata = dict(x=123)
    super_tensors = (torch.randn(3), torch.randint(0, 5, (3,)))

    class TestAverager(DecentralizedAverager):
        def get_current_state(self):
            """
            Get current state and send it to a peer. executed in the host process. Meant to be overridden.
            :returns: a tuple of (serializable_small_metadata, sequence of torch tensors)
            """
            nonlocal num_calls, super_metadata, super_tensors
            num_calls += 1
            return super_metadata, super_tensors

    dht_instances = launch_dht_instances(2)
    averager1 = TestAverager(
        [torch.randn(3), torch.rand(5)],
        dht=dht_instances[0],
        start=True,
        prefix="demo-run",
        target_group_size=2,
    )

    averager2 = TestAverager(
        [torch.randn(3), torch.rand(5)],
        dht=dht_instances[1],
        start=True,
        prefix="demo-run",
        target_group_size=2,
    )

    time.sleep(0.5)

    assert num_calls == 0
    got_metadata, got_tensors = averager2.load_state_from_peers()
    assert num_calls == 1
    assert got_metadata == super_metadata
    assert all(map(torch.allclose, got_tensors, super_tensors))

    super_metadata["y"] = 123
    super_tensors[1][2] = 9
    assert num_calls == 1
    assert got_metadata != super_metadata
    assert not all(map(torch.allclose, got_tensors, super_tensors))
    got_metadata, got_tensors = averager2.load_state_from_peers()
    assert num_calls == 2
    assert got_metadata == super_metadata
    assert all(map(torch.allclose, got_tensors, super_tensors))

    averager1.allow_state_sharing = False
    assert averager2.load_state_from_peers() is None

    averager1.allow_state_sharing = True
    time.sleep(0.5)
    got_metadata, got_tensors = averager2.load_state_from_peers()
    assert num_calls == 3
    assert got_metadata == super_metadata

    for instance in [averager1, averager2] + dht_instances:
        instance.shutdown()


@pytest.mark.forked
def test_load_state_priority():
    dht_instances = launch_dht_instances(4)

    averagers = []
    for i in range(4):
        averager = hivemind.DecentralizedAverager(
            [torch.randn(3), torch.rand(5), torch.tensor([i], dtype=torch.float32)],
            dht=dht_instances[i],
            start=True,
            prefix="demo-run",
            target_group_size=2,
            allow_state_sharing=i != 1,
        )
        averager.state_sharing_priority = 5 - abs(2 - i)
        averagers.append(averager)

    time.sleep(0.5)
    metadata, tensors = averagers[0].load_state_from_peers(timeout=1)
    assert tensors[-1].item() == 2

    metadata, tensors = averagers[2].load_state_from_peers(timeout=1)
    assert tensors[-1].item() == 3

    averagers[0].state_sharing_priority = 10
    time.sleep(0.2)

    metadata, tensors = averagers[2].load_state_from_peers(timeout=1)
    assert tensors[-1].item() == 0

    averagers[1].allow_state_sharing = False
    averagers[2].allow_state_sharing = False
    metadata, tensors = averagers[0].load_state_from_peers(timeout=1)
    assert tensors[-1].item() == 3

    for averager in averagers:
        averager.shutdown()
    for dht in dht_instances:
        dht.shutdown()


@pytest.mark.forked
def test_getset_bits():
    dht = hivemind.DHT(start=True)
    averager = DecentralizedAverager(
        [torch.randn(3)],
        dht=dht,
        start=True,
        prefix="test_prefix",
        target_group_size=2,
    )
    averager.set_group_bits("00101011101010")
    assert averager.get_group_bits() == "00101011101010"


@pytest.mark.forked
def test_averaging_trigger():
    averagers = tuple(
        DecentralizedAverager(
            averaged_tensors=[torch.randn(3)],
            dht=dht,
            min_matchmaking_time=0.5,
            request_timeout=0.3,
            prefix="mygroup",
            initial_group_bits="",
            start=True,
        )
        for dht in launch_dht_instances(4)
    )

    controls = []
    for i, averager in enumerate(averagers):
        controls.append(
            averager.step(
                wait=False,
                scheduled_time=hivemind.get_dht_time() + 0.5,
                weight=1.0,
                require_trigger=i in (1, 2),
            )
        )

    time.sleep(0.6)

    c0, c1, c2, c3 = controls
    assert not any(c.done() for c in controls)
    assert c0.stage == AveragingStage.RUNNING_ALLREDUCE
    assert c1.stage == AveragingStage.AWAITING_TRIGGER
    assert c2.stage == AveragingStage.AWAITING_TRIGGER
    assert c3.stage == AveragingStage.RUNNING_ALLREDUCE

    c1.allow_allreduce()
    c2.allow_allreduce()
    time.sleep(0.5)
    assert all(c.stage == AveragingStage.FINISHED for c in controls)
    assert all(c.done() for c in controls)

    # check that setting trigger twice does not raise error
    c0.allow_allreduce()


@pytest.mark.forked
def test_averaging_cancel():
    averagers = tuple(
        DecentralizedAverager(
            averaged_tensors=[torch.randn(3)],
            dht=dht,
            min_matchmaking_time=0.5,
            request_timeout=0.3,
            client_mode=(i % 2 == 0),
            prefix="mygroup",
            start=True,
        )
        for i, dht in enumerate(launch_dht_instances(4))
    )

    step_controls = [averager.step(wait=False, scheduled_time=hivemind.get_dht_time() + 1) for averager in averagers]

    time.sleep(0.1)
    step_controls[0].cancel()
    step_controls[1].cancel()

    for i, control in enumerate(step_controls):
        if i in (0, 1):
            assert control.cancelled()
        else:
            assert control.result() is not None and len(control.result()) == 2

    for averager in averagers:
        averager.shutdown()
