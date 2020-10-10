import asyncio

import grpc
import numpy as np
import pytest
import torch
import hivemind
from hivemind.client.expert import DUMMY
from hivemind import background_server


def test_moe():
    all_expert_uids = [f'ffn.{np.random.randint(0, 3)}.{np.random.randint(0, 3)}.{np.random.randint(0, 3)}'
                       for _ in range(20)]
    with background_server(expert_uids=all_expert_uids, device='cpu', expert_cls='ffn',
                           num_handlers=1, hidden_dim=16) as (server_endpoint, dht_endpoint):
        dht = hivemind.DHT(start=True, expiration=999, initial_peers=[dht_endpoint])

        dmoe = hivemind.RemoteMixtureOfExperts(
            in_features=16, grid_size=(32, 32, 32), dht=dht, k_best=3, uid_prefix='ffn')

        for i in range(10):
            out = dmoe(torch.randn(10, 16))
            out.sum().backward()


def test_call_many():
    k_min = 1
    timeout_after_k_min = None
    backward_k_min = 1
    forward_timeout = None
    backward_timeout = None
    rtol = 1e-3
    atol = 1e-6

    with background_server(num_experts=5, device='cpu', expert_cls='ffn', num_handlers=8, hidden_dim=64,
                           Optimizer=None, no_dht=True) as (server_endpoint, dht_endpoint):
        inputs = torch.randn(4, 64, requires_grad=True)
        inputs_clone = inputs.clone().detach().requires_grad_(True)
        e0, e1, e2, e3, e4 = [hivemind.RemoteExpert(f'expert.{i}', server_endpoint) for i in range(5)]
        e5 = hivemind.RemoteExpert(f'thisshouldnotexist', '127.0.0.1:80')

        mask, expert_outputs = hivemind.client.moe._RemoteCallMany.apply(
            DUMMY, [[e0, e1, e2], [e2, e4], [e1, e5, e3], []],
            k_min, backward_k_min, timeout_after_k_min, forward_timeout, backward_timeout,
            asyncio.new_event_loop(), e1.info, inputs
        )
        assert mask.shape == (4, 3)
        assert expert_outputs.shape == (4, 3, 64)

        assert np.all(mask.data.numpy() == np.array([[True, True, True],
                                                     [True, True, False],
                                                     [True, False, True],
                                                     [False, False, False]])), f"Incorrect mask, {mask}"

        reference_outputs = torch.zeros_like(expert_outputs)
        reference_outputs[0, 0] = e0(inputs_clone[0:1])
        reference_outputs[0, 1] = e1(inputs_clone[0:1])
        reference_outputs[0, 2] = e2(inputs_clone[0:1])
        reference_outputs[1, 0] = e2(inputs_clone[1:2])
        reference_outputs[1, 1] = e4(inputs_clone[1:2])
        reference_outputs[2, 0] = e1(inputs_clone[2:3])
        reference_outputs[2, 2] = e3(inputs_clone[2:3])

        assert torch.allclose(expert_outputs, reference_outputs, rtol, atol)
        proj = torch.randn(4, 64)
        loss = (expert_outputs[(0, 1, 1, 2), (0, 2, 1, 0)] * proj).sum()
        loss.backward()
        our_grad = inputs.grad.data.cpu().clone()

        reference_loss = (reference_outputs[(0, 1, 1, 2), (0, 2, 1, 0)] * proj).sum()
        reference_loss.backward()
        reference_grad = inputs_clone.grad.data.cpu().clone()
        assert torch.allclose(our_grad, reference_grad, rtol, atol)


def test_remote_module_call():
    with background_server(num_experts=1, device='cpu', expert_cls='ffn', num_handlers=1, hidden_dim=1024,
                           Optimizer=None, no_dht=True) as (server_endpoint, dht_endpoint):
        real_expert = hivemind.RemoteExpert('expert.0', server_endpoint)
        fake_expert = hivemind.RemoteExpert('oiasfjiasjf', server_endpoint)

        out1 = real_expert(torch.randn(1, 1024))
        assert out1.shape == (1, 1024)
        dummy_x = torch.randn(3, 1024, requires_grad=True)
        out3 = real_expert(dummy_x)
        assert out3.shape == (3, 1024)
        out3_again = real_expert(dummy_x[1:])
        assert torch.allclose(out3_again, out3[1:])
        out3_again.norm().backward()
        assert dummy_x.grad is not None and dummy_x.grad.norm() > 0

        with pytest.raises(grpc.RpcError):
            real_expert(torch.randn(3, 11))
        with pytest.raises(grpc.RpcError):
            fake_expert(dummy_x)


def test_moe_beam_search():
    all_expert_uids = [f'ffn.{5 + i}.{10 + j}.{15 + k}' for i in range(10) for j in range(10) for k in range(10)]
    dht = hivemind.DHT(start=True, expiration=999)
    assert all(dht.declare_experts(all_expert_uids, endpoint='fake-endpoint'))

    dmoe = hivemind.RemoteMixtureOfExperts(
        in_features=32, grid_size=(32, 32, 32), dht=dht, k_best=4, uid_prefix='ffn')

    for i in range(25):
        input = torch.randn(32)
        grid_scores = dmoe.proj(input).split_with_sizes(dmoe.grid_size, dim=-1)

        chosen_experts = dmoe.loop.run_until_complete(dmoe.beam_search(grid_scores, k_best=dmoe.k_best))

        chosen_scores = dmoe.compute_expert_scores([dim_scores[None] for dim_scores in grid_scores],
                                                   [chosen_experts])[0]

        all_scores = dmoe.compute_expert_scores([dim_scores[None] for dim_scores in grid_scores],
                                                [[hivemind.RemoteExpert(uid, '') for uid in all_expert_uids]])[0]
        true_best_scores = sorted(all_scores.cpu().detach().numpy(), reverse=True)[:len(chosen_experts)]
        our_best_scores = list(chosen_scores.cpu().detach().numpy())
        assert np.allclose(true_best_scores, our_best_scores)


def test_determinism():
    rtol = 0
    atol = 1e-6

    xx = torch.randn(32, 1024, requires_grad=True)
    mask = torch.randint(0, 1, (32, 1024))

    with background_server(num_experts=1, device='cpu', expert_cls='det_dropout', num_handlers=1,
                           Optimizer=None, no_dht=True) as (server_endpoint, dht_endpoint):
        expert = hivemind.RemoteExpert(uid=f'expert.0', endpoint=server_endpoint)

        out = expert(xx, mask)
        out_rerun = expert(xx, mask)

        grad, = torch.autograd.grad(out.sum(), xx, retain_graph=True)
        grad_rerun, = torch.autograd.grad(out_rerun.sum(), xx, retain_graph=True)

    assert torch.allclose(out, out_rerun, rtol, atol), "Dropout layer outputs are non-deterministic."
    assert torch.allclose(grad, grad_rerun, rtol, atol), "Gradients are non-deterministic."


def test_compute_expert_scores():
    try:
        dht = hivemind.DHT(start=True)
        moe = hivemind.client.moe.RemoteMixtureOfExperts(
            dht=dht, in_features=1024, grid_size=(40,), k_best=4, k_min=1, timeout_after_k_min=1,
            uid_prefix='expert')
        gx, gy = torch.randn(4, 5, requires_grad=True), torch.randn(4, 3, requires_grad=True)
        ii = [[4, 0, 2], [3, 1, 1, 1, 3], [0], [3, 2]]
        jj = [[2, 2, 1], [0, 1, 2, 0, 1], [0], [1, 2]]
        batch_experts = [
            [hivemind.RemoteExpert(uid=f'expert.{ii[batch_i][expert_i]}.{jj[batch_i][expert_i]}', endpoint="[::]:1337")
             for expert_i in range(len(ii[batch_i]))]
            for batch_i in range(len(ii))
        ]  # note: these experts do not exists on server, we use them only to test moe compute_expert_scores
        logits = moe.compute_expert_scores([gx, gy], batch_experts)
        torch.softmax(logits, dim=-1).norm(dim=-1).mean().backward()
        assert gx.grad.norm().item() > 0 and gy.grad.norm().item(), "compute_expert_scores didn't backprop"

        for batch_i in range(len(ii)):
            for expert_i in range(len(ii[batch_i])):
                assert torch.allclose(logits[batch_i, expert_i],
                                      gx[batch_i, ii[batch_i][expert_i]] + gy[batch_i, jj[batch_i][expert_i]]), \
                    "compute_expert_scores returned incorrect score"
    finally:
        dht.shutdown()
