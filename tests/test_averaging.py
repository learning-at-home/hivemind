import asyncio
import random
import time

import numpy as np
import pytest
import torch

import hivemind
import hivemind.averaging.averager
from conftest import cleanup_children
from hivemind.averaging.allreduce import AveragingMode
from hivemind.averaging.key_manager import GroupKeyManager
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.p2p import PeerID
from hivemind.proto.runtime_pb2 import CompressionType
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
        hivemind.averaging.DecentralizedAverager(
            tensors,
            dht=dht,
            target_group_size=4,
            averaging_expiration=15,
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
            assert averager.endpoint in result

    for averager in averagers:
        if averager.mode != AveragingMode.AUX:
            with averager.get_tensors() as averaged_tensors:
                for ref, our in zip(reference, averaged_tensors):
                    assert torch.allclose(ref, our, atol=1e-6)

    for instance in averagers + dht_instances:
        instance.shutdown()


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
        hivemind.averaging.DecentralizedAverager(
            tensors,
            dht=dht,
            target_group_size=4,
            averaging_expiration=15,
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

    for instance in averagers + dht_instances:
        instance.shutdown()


@pytest.mark.forked
def test_allreduce_compression():
    """this test ensures that compression works correctly when multiple tensors have different compression types"""

    tensors1 = [torch.linspace(0, 500, 1000) ** 0.5, torch.randn(1000)]
    tensors2 = [torch.linspace(300, 800, 1000) ** 0.5, torch.randn(1000)]
    results = {}

    FLOAT16, UINT8 = CompressionType.FLOAT16, CompressionType.UNIFORM_8BIT

    for compression_type_pair in [(FLOAT16, FLOAT16), (FLOAT16, UINT8), (UINT8, FLOAT16), (UINT8, UINT8)]:
        dht_instances = launch_dht_instances(2)
        averager1 = hivemind.averaging.DecentralizedAverager(
            [x.clone() for x in tensors1],
            dht=dht_instances[0],
            compression_type=compression_type_pair,
            client_mode=True,
            target_group_size=2,
            prefix="mygroup",
            start=True,
        )
        averager2 = hivemind.averaging.DecentralizedAverager(
            [x.clone() for x in tensors2],
            dht=dht_instances[1],
            compression_type=compression_type_pair,
            target_group_size=2,
            prefix="mygroup",
            start=True,
        )

        for future in averager1.step(wait=False), averager2.step(wait=False):
            future.result()

        with averager1.get_tensors() as averaged_tensors:
            results[compression_type_pair] = averaged_tensors

        for instance in [averager1, averager2] + dht_instances:
            instance.shutdown()

    assert torch.allclose(results[UINT8, FLOAT16][0], results[UINT8, UINT8][0])
    assert torch.allclose(results[UINT8, FLOAT16][1], results[FLOAT16, FLOAT16][1])
    assert torch.allclose(results[UINT8, UINT8][1], results[FLOAT16, UINT8][1])
    assert torch.allclose(results[FLOAT16, UINT8][0], results[FLOAT16, FLOAT16][0])

    assert not torch.allclose(results[UINT8, FLOAT16][1], results[UINT8, UINT8][1])
    assert not torch.allclose(results[UINT8, FLOAT16][0], results[FLOAT16, FLOAT16][0])
    assert not torch.allclose(results[UINT8, UINT8][0], results[FLOAT16, UINT8][0])
    assert not torch.allclose(results[FLOAT16, UINT8][1], results[FLOAT16, FLOAT16][1])

    reference = [(tensors1[i] + tensors2[i]) / 2 for i in range(len(tensors1))]
    for i in range(2):
        assert 0 < torch.mean(torch.square(results[FLOAT16, FLOAT16][i] - reference[i])).item() <= 1e-5
        assert 1e-5 < torch.mean(torch.square(results[UINT8, UINT8][i] - reference[i])).item() <= 1e-2


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
        hivemind.averaging.DecentralizedAverager(
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

    for averager in averagers + dht_instances:
        averager.shutdown()


@pytest.mark.forked
def test_allgather():
    dht_instances = launch_dht_instances(8)
    averagers = [
        hivemind.averaging.DecentralizedAverager(
            [torch.ones(1)],
            dht=dht,
            target_group_size=4,
            averaging_expiration=15,
            prefix="mygroup",
            initial_group_bits="000",
            start=True,
        )
        for dht in dht_instances
    ]

    futures = []
    for i, averager in enumerate(averagers):
        futures.append(averager.step(wait=False, gather=dict(batch_size=123 + i, foo="bar")))

    gathered_data = [future.result() for future in futures]
    gathered_data_reprs = [
        repr(sorted({peer_id.to_base58(): data for peer_id, data in result.items()})) for result in gathered_data
    ]
    assert len(set(gathered_data_reprs)) == 2

    reference_metadata = {
        averager.endpoint: dict(batch_size=123 + i, foo="bar") for i, averager in enumerate(averagers)
    }
    for future in futures:
        gathered = future.result()

        assert len(gathered) == 4

        for endpoint in gathered:
            assert gathered[endpoint] == reference_metadata[endpoint]

    for averager in averagers + dht_instances:
        averager.shutdown()


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
        vector_size = np.random.randint(1, 1024 ** 3)
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
        hivemind.averaging.DecentralizedAverager(
            averaged_tensors=[torch.randn(3)],
            dht=dht,
            target_group_size=2,
            averaging_expiration=1,
            request_timeout=0.5,
            prefix="mygroup",
            initial_group_bits=bin(i)[2:].rjust(3, "0"),
            start=True,
        )
        for i, dht in enumerate(dht_instances)
    ]
    step_futures = [averager.step(wait=False) for averager in averagers]
    for future in step_futures:
        assert len(future.result()) == 2

    for averager in averagers + dht_instances:
        averager.shutdown()


@pytest.mark.skip(
    reason="The current implementation of elasticity (multi-stage averaging for the case when "
    "num_peers > ~3 * target_group_size) is incorrect (TODO @justheuristic)"
)
@pytest.mark.forked
def test_overcrowded(num_peers=16):
    dht_instances = launch_dht_instances(num_peers)
    averagers = [
        hivemind.averaging.DecentralizedAverager(
            averaged_tensors=[torch.randn(3)],
            dht=dht,
            target_group_size=2,
            averaging_expiration=1,
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

    for averager in averagers + dht_instances:
        averager.shutdown()


@pytest.mark.forked
def test_load_state_from_peers():
    num_calls = 0
    super_metadata = dict(x=123)
    super_tensors = (torch.randn(3), torch.randint(0, 5, (3,)))

    class TestAverager(hivemind.averaging.DecentralizedAverager):
        def get_current_state(self):
            """
            Get current state and send it to a peer. executed in the host process. Meant to be overriden.
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

    dht_instances[1].get("demo-run.all_averagers")
    averager2 = TestAverager(
        [torch.randn(3), torch.rand(5)],
        dht=dht_instances[1],
        start=True,
        prefix="demo-run",
        target_group_size=2,
    )

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
    got_metadata, got_tensors = averager2.load_state_from_peers()
    assert num_calls == 3
    assert got_metadata == super_metadata

    for instance in [averager1, averager2] + dht_instances:
        instance.shutdown()


@pytest.mark.forked
def test_getset_bits():
    dht = hivemind.DHT(start=True)
    averager = hivemind.averaging.DecentralizedAverager(
        [torch.randn(3)],
        dht=dht,
        start=True,
        prefix="test_prefix",
        target_group_size=2,
    )
    averager.set_group_bits("00101011101010")
    assert averager.get_group_bits() == "00101011101010"


@pytest.mark.forked
def test_training_averager(n_steps: int = 10, n_dims: int = 16):
    torch.manual_seed(42)

    dht_instances = launch_dht_instances(2)
    common_kwargs = {
        "start": True,
        "prefix": "demo-run",
        "target_group_size": 2,
    }

    x1 = torch.randn(n_dims, requires_grad=True)
    opt1 = torch.optim.Adam([x1], lr=0.05)
    averager1 = hivemind.averaging.TrainingAverager(
        opt1,
        average_gradients=True,
        average_parameters=True,
        average_opt_statistics=["exp_avg_sq"],
        dht=dht_instances[0],
        **common_kwargs
    )

    x2 = torch.randn(n_dims, requires_grad=True)
    opt2 = torch.optim.Adam([x2], lr=0.05)
    averager2 = hivemind.averaging.TrainingAverager(
        opt2,
        average_gradients=True,
        average_parameters=True,
        average_opt_statistics=["exp_avg_sq"],
        dht=dht_instances[1],
        **common_kwargs
    )
    a = torch.ones(n_dims)

    for i in range(n_steps):
        opt1.zero_grad()
        opt2.zero_grad()
        (x1 - a).pow(2).sum().backward()
        (x2 - a).pow(2).sum().backward()
        opt1.step()
        opt2.step()

        with torch.no_grad():
            x_avg = 0.5 * (x1 + x2)
            grad_avg = 0.5 * (x1.grad + x2.grad)
            stats_avg = 0.5 * (opt1.state[x1]["exp_avg_sq"] + opt2.state[x2]["exp_avg_sq"])

        # we set wait=False in order to prevent deadlock, when averager1 locks and waits for averager2
        f1 = averager1.step(wait=False)
        f2 = averager2.step(wait=False)
        f1.result()
        f2.result()

        assert torch.allclose(x1, x_avg)
        assert torch.allclose(x2, x_avg)
        assert torch.allclose(x1.grad, grad_avg)
        assert torch.allclose(x2.grad, grad_avg)
        assert torch.allclose(opt1.state[x1]["exp_avg_sq"], stats_avg)
        assert torch.allclose(opt2.state[x2]["exp_avg_sq"], stats_avg)

    for instance in [averager1, averager2] + dht_instances:
        instance.shutdown()

if __name__ == '__main__':
    with cleanup_children():
        loop = asyncio.new_event_loop()
        loop.run_until_complete(test_key_manager())
        print(f"test_key_manager()")
        del loop
    for n_clients in [0, 1, 2]:
        for n_aux in [0, 1, 2]:
            with cleanup_children():
                _test_allreduce_once(n_clients, n_aux)
                print(f"_test_allreduce_once({n_clients}, {n_aux})")
    for n_clients, n_aux in [(0, 4), (1, 3), (0, 3)]:
        with cleanup_children():
            _test_allreduce_once(n_clients, n_aux)
            print(f"_test_allreduce_once({n_clients}, {n_aux})")
    print("DONE!")
