import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind import RemoteExpert, find_open_port
from test_utils.run_server import background_server

from sklearn.datasets import load_digits


def test_training(port: Optional[int] = None, max_steps: int = 100, threshold: float = 0.9):
    if port is None:
        port = find_open_port()
    dataset = load_digits()
    X_train, y_train = torch.tensor(dataset['data'], dtype=torch.float), torch.tensor(dataset['target'])

    with background_server(num_experts=2, device='cpu', port=port, hidden_dim=64):
        expert1 = RemoteExpert('expert.0', host='127.0.0.1', port=port)
        expert2 = RemoteExpert('expert.1', host='127.0.0.1', port=port)
        model = nn.Sequential(expert2, nn.Tanh(), expert1, nn.Linear(64, 10))

        opt = torch.optim.SGD(model.parameters(), lr=0.05)

        for step in range(max_steps):
            opt.zero_grad()

            outputs = model(X_train)
            loss = F.cross_entropy(outputs, y_train)
            loss.backward()
            opt.step()

            accuracy = (outputs.argmax(dim=1) == y_train).numpy().mean()
            if accuracy >= threshold:
                break

        assert accuracy >= threshold, f"too small accuracy: {accuracy}"
