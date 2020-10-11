from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

from hivemind import RemoteExpert, background_server


def test_training(port: Optional[int] = None, max_steps: int = 100, threshold: float = 0.9):
    dataset = load_digits()
    X_train, y_train = torch.tensor(dataset['data'], dtype=torch.float), torch.tensor(dataset['target'])
    SGD = partial(torch.optim.SGD, lr=0.05)

    with background_server(num_experts=2, device='cpu', optim_cls=SGD, hidden_dim=64) as (server_endpoint, _):
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
