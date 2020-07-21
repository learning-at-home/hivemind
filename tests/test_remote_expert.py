import asyncio

import grpc
import numpy as np
import pytest
import torch

import hivemind
from hivemind.client.expert import DUMMY
from test_utils.run_server import background_server


def test_remote_module_call():
    with background_server(num_experts=1, device='cpu', expert_cls='ffn', num_handlers=1, hidden_dim=1024,
                           no_optimizer=True, no_dht=True) as (server_endpoint, dht_endpoint):
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


def test_determinism():
    rtol = 0
    atol = 1e-6

    xx = torch.randn(32, 1024, requires_grad=True)
    mask = torch.randint(0, 1, (32, 1024))

    with background_server(num_experts=1, device='cpu', expert_cls='det_dropout', num_handlers=1,
                           no_optimizer=True, no_dht=True) as (server_endpoint, dht_endpoint):
        expert = hivemind.RemoteExpert(uid=f'expert.0', endpoint=server_endpoint)

        out = expert(xx, mask)
        out_rerun = expert(xx, mask)

        grad, = torch.autograd.grad(out.sum(), xx, retain_graph=True)
        grad_rerun, = torch.autograd.grad(out_rerun.sum(), xx, retain_graph=True)

    assert torch.allclose(out, out_rerun, rtol, atol), "Dropout layer outputs are non-deterministic."
    assert torch.allclose(grad, grad_rerun, rtol, atol), "Gradients are non-deterministic."


def test_call_many():
    k_min = 1
    timeout_after_k_min = None
    backward_k_min = 1
    forward_timeout = None
    backward_timeout = None
    rtol = 1e-3
    atol = 1e-6

    with background_server(num_experts=5, device='cpu', expert_cls='ffn', num_handlers=8, hidden_dim=64,
                           no_optimizer=True, no_dht=True) as (server_endpoint, dht_endpoint):

        inputs = torch.randn(4, 64, requires_grad=True)
        inputs_clone = inputs.clone().detach().requires_grad_(True)
        e0, e1, e2, e3, e4 = [hivemind.RemoteExpert(f'expert.{i}', server_endpoint) for i in range(5)]
        e5 = hivemind.RemoteExpert(f'thisshouldnotexist', '127.0.0.1:80')

        mask, expert_outputs = hivemind.client.moe._RemoteCallMany.apply(
            DUMMY, [[e0, e1, e2], [e2, e4], [e1, e5, e3], []],
            k_min, backward_k_min, timeout_after_k_min, forward_timeout, backward_timeout,
            asyncio.new_event_loop(), inputs
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
