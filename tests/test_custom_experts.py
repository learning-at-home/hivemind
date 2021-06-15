import os

import pytest
import torch

from hivemind import RemoteExpert
from hivemind.moe.server import background_server

CUSTOM_EXPERTS_PATH = os.path.join(os.path.dirname(__file__), "test_utils", "custom_networks.py")


@pytest.mark.forked
def test_custom_expert(hid_dim=16):
    with background_server(
        expert_cls="perceptron",
        num_experts=2,
        device="cpu",
        hidden_dim=hid_dim,
        num_handlers=2,
        no_dht=True,
        custom_module_path=CUSTOM_EXPERTS_PATH,
    ) as (server_endpoint, _):
        expert0 = RemoteExpert("expert.0", server_endpoint)
        expert1 = RemoteExpert("expert.1", server_endpoint)

        for batch_size in (1, 4):
            batch = torch.randn(batch_size, hid_dim)

            output0 = expert0(batch)
            output1 = expert1(batch)

            loss = output0.sum()
            loss.backward()
            loss = output1.sum()
            loss.backward()


@pytest.mark.forked
def test_multihead_expert(hid_dim=16):
    with background_server(
        expert_cls="multihead",
        num_experts=2,
        device="cpu",
        hidden_dim=hid_dim,
        num_handlers=2,
        no_dht=True,
        custom_module_path=CUSTOM_EXPERTS_PATH,
    ) as (server_endpoint, _):
        expert0 = RemoteExpert("expert.0", server_endpoint)
        expert1 = RemoteExpert("expert.1", server_endpoint)

        for batch_size in (1, 4):
            batch = (
                torch.randn(batch_size, hid_dim),
                torch.randn(batch_size, 2 * hid_dim),
                torch.randn(batch_size, 3 * hid_dim),
            )

            output0 = expert0(*batch)
            output1 = expert1(*batch)

            loss = output0.sum()
            loss.backward()
            loss = output1.sum()
            loss.backward()
