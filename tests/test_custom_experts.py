import os

import pytest
import torch

from hivemind.dht import DHT
from hivemind.moe.client.expert import create_remote_experts
from hivemind.moe.expert_uid import ExpertInfo
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
        custom_module_path=CUSTOM_EXPERTS_PATH,
    ) as server_peer_info:
        dht = DHT(initial_peers=server_peer_info.addrs, start=True)
        expert0, expert1 = create_remote_experts(
            [
                ExpertInfo(uid="expert.0", peer_id=server_peer_info.peer_id),
                ExpertInfo(uid="expert.1", peer_id=server_peer_info.peer_id),
            ],
            dht=dht,
        )

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
        custom_module_path=CUSTOM_EXPERTS_PATH,
    ) as server_peer_info:
        dht = DHT(initial_peers=server_peer_info.addrs, start=True)
        expert0, expert1 = create_remote_experts(
            [
                ExpertInfo(uid="expert.0", peer_id=server_peer_info.peer_id),
                ExpertInfo(uid="expert.1", peer_id=server_peer_info.peer_id),
            ],
            dht=dht,
        )

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
