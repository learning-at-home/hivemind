from functools import partial

import time
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

from hivemind import RemoteExpert, background_server, DHT, DecentralizedSGD


@pytest.mark.forked
def test_training(max_steps: int = 100, threshold: float = 0.9):
    dataset = load_digits()
    X_train, y_train = torch.tensor(dataset['data'], dtype=torch.float), torch.tensor(dataset['target'])
    SGD = partial(torch.optim.SGD, lr=0.05)

    with background_server(num_experts=2, device='cpu', optim_cls=SGD, hidden_dim=64, num_handlers=1,
                           no_dht=True) as (server_endpoint, dht_endpoint):
        expert1 = RemoteExpert('expert.0', server_endpoint)
        expert2 = RemoteExpert('expert.1', server_endpoint)
        model = nn.Sequential(expert2, nn.Tanh(), expert1, nn.Linear(64, 10))

        opt = torch.optim.SGD(model.parameters(), lr=0.05)

        for step in range(max_steps):
            opt.zero_grad()

            outputs = model(X_train)
            loss = F.cross_entropy(outputs, y_train)
            loss.backward()
            opt.step()

            accuracy = (outputs.argmax(dim=1) == y_train).float().mean().item()
            if accuracy >= threshold:
                break

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
    torch.allclose(param1, torch.full_like(param1, reference))
