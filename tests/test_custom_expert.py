import os
import pytest
import torch
import hivemind

from typing import Optional
from hivemind import RemoteExpert, background_server

@pytest.mark.forked
def test_custom_expert(port: Optional[int] = None, hid_dim=16):
    with background_server(
        expert_cls='perceptron', num_experts=2, device='cpu',
        hidden_dim=hid_dim, num_handlers=2, no_dht=True,
        custom_module_path=os.path.join(os.path.dirname(__file__), 'custom_networks.py')) as (server_endpoint, _):

        expert0 = RemoteExpert('expert.0', server_endpoint)
        expert1 = RemoteExpert('expert.1', server_endpoint)

        for step in range(1, 4):
            batch = torch.randn(step, hid_dim)

            output0 = expert0(batch)
            output1 = expert1(batch)

            loss = output0.sum()
            loss.backward()
            loss = output1.sum()
            loss.backward()

@pytest.mark.forked
def test_multihead_expert(port: Optional[int] = None, hid_dim=16):
    with background_server(
        expert_cls='multihead', num_experts=2, device='cpu',
        hidden_dim=hid_dim, num_handlers=2, no_dht=True,
        custom_module_path=os.path.join(os.path.dirname(__file__), 'custom_networks.py')) as (server_endpoint, _):

        expert0 = RemoteExpert('expert.0', server_endpoint)
        expert1 = RemoteExpert('expert.1', server_endpoint)

        for step in range(1, 4):
            batch = (torch.randn(step, hid_dim), torch.randn(step, 2 * hid_dim), torch.randn(step, 3 * hid_dim))

            output0 = expert0(*batch)
            output1 = expert1(*batch)

            loss = output0.sum()
            loss.backward()
            loss = output1.sum()
            loss.backward()

if __name__ == "__main__":
    test_custom_expert()
    with open('debug_logs', 'a') as f:
        print("Done first : ", name, os.getpid(), file=f)
    test_multihead_expert()
