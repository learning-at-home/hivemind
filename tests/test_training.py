import time
from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

from hivemind import DHT
from hivemind.moe.client import RemoteExpert, RemoteMixtureOfExperts, RemoteSwitchMixtureOfExperts
from hivemind.moe.server import background_server
from hivemind.optim import DecentralizedSGD, DecentralizedAdam


@pytest.mark.forked
def test_training(max_steps: int = 100, threshold: float = 0.9):
    dataset = load_digits(n_class=2)
    X_train, y_train = torch.tensor(dataset['data'], dtype=torch.float), torch.tensor(dataset['target'])
    SGD = partial(torch.optim.SGD, lr=0.05)

    with background_server(num_experts=2, device='cpu', optim_cls=SGD, hidden_dim=64, num_handlers=1,
                           no_dht=True) as (server_endpoint, dht_endpoint):
        expert1 = RemoteExpert('expert.0', server_endpoint)
        expert2 = RemoteExpert('expert.1', server_endpoint)
        model = nn.Sequential(expert2, nn.ReLU(), expert1, nn.Linear(64, 2))

        opt = SGD(model.parameters(), lr=0.05)

        for step in range(max_steps):
            outputs = model(X_train)
            loss = F.cross_entropy(outputs, y_train)
            loss.backward()
            opt.step()
            opt.zero_grad()

            accuracy = (outputs.argmax(dim=1) == y_train).float().mean().item()
            if accuracy >= threshold:
                break

        assert accuracy >= threshold, f"too small accuracy: {accuracy}"


@pytest.mark.forked
def test_moe_training(max_steps: int = 100, threshold: float = 0.9, num_experts=2):
    dataset = load_digits(n_class=2)
    X_train, y_train = torch.tensor(dataset['data'], dtype=torch.float), torch.tensor(dataset['target'])
    SGD = partial(torch.optim.SGD, lr=0.05)

    all_expert_uids = [f'expert.{i}' for i in range(num_experts)]
    with background_server(expert_uids=all_expert_uids, device='cpu', optim_cls=SGD, hidden_dim=64, num_handlers=1) \
            as (server_endpoint, dht_endpoint):
        dht = DHT(start=True, initial_peers=[dht_endpoint])

        moe = RemoteMixtureOfExperts(in_features=64, grid_size=(num_experts,), dht=dht, uid_prefix='expert.', k_best=2)
        model = nn.Sequential(moe, nn.Linear(64, 2))

        opt = SGD(model.parameters(), lr=0.05)

        for step in range(max_steps):
            outputs = model(X_train)
            loss = F.cross_entropy(outputs, y_train)
            loss.backward()
            opt.step()
            opt.zero_grad()

            accuracy = (outputs.argmax(dim=1) == y_train).float().mean().item()
            if accuracy >= threshold:
                break

        assert accuracy >= threshold, f"too small accuracy: {accuracy}"


class SwitchNetwork(nn.Module):
    def __init__(self, dht, in_features, num_classes, num_experts):
        super().__init__()
        self.moe = RemoteSwitchMixtureOfExperts(in_features=in_features, grid_size=(num_experts,), dht=dht,
                                                jitter_eps=0, uid_prefix='expert.', k_best=1,
                                                k_min=1)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        moe_output, balancing_loss = self.moe(x)
        return self.linear(moe_output), balancing_loss


@pytest.mark.forked
def test_switch_training(max_steps: int = 10, threshold: float = 0.9, num_experts=5):
    dataset = load_digits(n_class=2)
    X_train, y_train = torch.tensor(dataset['data'], dtype=torch.float), torch.tensor(dataset['target'])
    SGD = partial(torch.optim.SGD, lr=0.05)

    all_expert_uids = [f'expert.{i}' for i in range(num_experts)]
    with background_server(expert_uids=all_expert_uids, device='cpu', optim_cls=SGD, hidden_dim=64,
                           num_handlers=1) as (server_endpoint, dht_endpoint):
        dht = DHT(start=True, initial_peers=[dht_endpoint])

        model = SwitchNetwork(dht, 64, 2, num_experts)
        opt = SGD(model.parameters(), lr=0.05)

        for step in range(max_steps):
            outputs, balancing_loss = model(X_train)
            loss = F.cross_entropy(outputs, y_train) + 0.01 * balancing_loss
            loss.backward()
            opt.step()
            opt.zero_grad()

            accuracy = (outputs.argmax(dim=1) == y_train).float().mean().item()
            if accuracy >= threshold:
                break

        assert model.moe.grid_utilization.min().item() > (1 / num_experts) / 2
        assert accuracy >= threshold, f"too small accuracy: {accuracy}"


@pytest.mark.forked
def test_decentralized_optimizer_step():
    dht_root = DHT(start=True)
    initial_peers = [f"127.0.0.1:{dht_root.port}"]

    param1 = torch.nn.Parameter(torch.zeros(32, 32), requires_grad=True)
    opt1 = DecentralizedSGD([param1], lr=0.1, dht=DHT(initial_peers=initial_peers, start=True),
                            prefix='foo', target_group_size=2, verbose=True)

    param2 = torch.nn.Parameter(torch.ones(32, 32), requires_grad=True)
    opt2 = DecentralizedSGD([param2], lr=0.05, dht=DHT(initial_peers=initial_peers, start=True),
                            prefix='foo', target_group_size=2, verbose=True)

    assert not torch.allclose(param1, param2)

    (param1.sum() + 300 * param2.sum()).backward()

    opt1.step()
    opt2.step()

    time.sleep(0.5)
    assert torch.allclose(param1, param2)
    reference = 0.5 * (0.0 - 0.1 * 1.0) + 0.5 * (1.0 - 0.05 * 300)
    assert torch.allclose(param1, torch.full_like(param1, reference))


@pytest.mark.forked
def test_decentralized_optimizer_averaging():
    dht_root = DHT(start=True)
    initial_peers = [f"127.0.0.1:{dht_root.port}"]

    param1 = torch.nn.Parameter(torch.zeros(32, 32), requires_grad=True)
    opt1 = DecentralizedAdam([param1], lr=0.1, averaging_steps_period=1, dht=DHT(initial_peers=initial_peers, start=True),
                            prefix='foo', target_group_size=2, verbose=True)

    param2 = torch.nn.Parameter(torch.ones(32, 32), requires_grad=True)
    opt2 = DecentralizedAdam([param2], lr=0.05, averaging_steps_period=1, dht=DHT(initial_peers=initial_peers, start=True),
                            prefix='foo', target_group_size=2, verbose=True)

    assert not torch.allclose(param1, param2)

    (param1.sum() + param2.sum()).backward()

    opt1.step()
    opt2.step()

    time.sleep(0.5)
    assert torch.allclose(param1, param2)
    assert torch.allclose(opt1.state[param1]["exp_avg_sq"], opt2.state[param2]["exp_avg_sq"])
