import numpy as np
import pytest
import torch

from hivemind.dht import DHT
from hivemind.moe.client.expert import RemoteExpert, create_remote_experts
from hivemind.moe.client.moe import DUMMY, RemoteMixtureOfExperts, _RemoteCallMany
from hivemind.moe.client.switch_moe import RemoteSwitchMixtureOfExperts
from hivemind.moe.expert_uid import ExpertInfo
from hivemind.moe.server import ModuleBackend, Server, background_server, declare_experts
from hivemind.moe.server.layers import name_to_block
from hivemind.p2p.p2p_daemon_bindings.control import P2PHandlerError
from hivemind.utils import BatchTensorDescriptor, get_dht_time


@pytest.mark.forked
def test_moe():
    all_expert_uids = [
        f"ffn.{np.random.randint(0, 3)}.{np.random.randint(0, 3)}.{np.random.randint(0, 3)}" for _ in range(10)
    ]
    with background_server(
        expert_uids=all_expert_uids, device="cpu", expert_cls="ffn", num_handlers=1, hidden_dim=16
    ) as server_peer_info:
        dht = DHT(start=True, initial_peers=server_peer_info.addrs)

        dmoe = RemoteMixtureOfExperts(in_features=16, grid_size=(4, 4, 4), dht=dht, k_best=3, uid_prefix="ffn.")

        for i in range(3):
            out = dmoe(torch.randn(10, 16))
            out.sum().backward()


@pytest.mark.forked
def test_no_experts():
    all_expert_uids = [
        f"expert.{np.random.randint(0, 3)}.{np.random.randint(0, 3)}.{np.random.randint(0, 3)}" for _ in range(10)
    ]
    with background_server(
        expert_uids=all_expert_uids, device="cpu", expert_cls="nop_delay", num_handlers=1, hidden_dim=16
    ) as server_peer_info:
        dht = DHT(start=True, initial_peers=server_peer_info.addrs)
        dmoe = RemoteSwitchMixtureOfExperts(
            in_features=16,
            grid_size=(4, 4, 4),
            dht=dht,
            uid_prefix="expert.",
            forward_timeout=0.1,
            backward_timeout=0.1,
            allow_zero_outputs=True,
        )

        for i in range(3):
            out, balancing_loss = dmoe(torch.randn(10, 16))
            out.sum().backward()


@pytest.mark.forked
def test_call_many(hidden_dim=16):
    k_min = 1
    timeout_after_k_min = None
    backward_k_min = 1
    forward_timeout = None
    backward_timeout = None
    detect_anomalies = False
    allow_zero_outputs = False
    atol = 1e-5

    with background_server(
        num_experts=5,
        device="cpu",
        expert_cls="ffn",
        num_handlers=1,
        hidden_dim=hidden_dim,
        optim_cls=None,
    ) as server_peer_info:
        inputs = torch.randn(4, hidden_dim, requires_grad=True)
        inputs_clone = inputs.clone().detach().requires_grad_(True)

        dht = DHT(initial_peers=server_peer_info.addrs, start=True)
        e0, e1, e2, e3, e4 = create_remote_experts(
            [ExpertInfo(uid=f"expert.{i}", peer_id=server_peer_info.peer_id) for i in range(5)],
            dht,
        )
        e5 = RemoteExpert(ExpertInfo(f"thisshouldnotexist", server_peer_info), None)

        mask, expert_outputs = _RemoteCallMany.apply(
            DUMMY,
            [[e0, e1, e2], [e2, e4], [e1, e5, e3], []],
            k_min,
            backward_k_min,
            timeout_after_k_min,
            forward_timeout,
            backward_timeout,
            detect_anomalies,
            allow_zero_outputs,
            e1.info,
            inputs,
        )
        assert mask.shape == (4, 3)
        assert expert_outputs.shape == (4, 3, hidden_dim)

        assert np.all(
            mask.data.numpy()
            == np.array([[True, True, True], [True, True, False], [True, False, True], [False, False, False]])
        ), f"Incorrect mask, {mask}"

        reference_outputs = torch.zeros_like(expert_outputs)
        reference_outputs[0, 0] = e0(inputs_clone[0:1])
        reference_outputs[0, 1] = e1(inputs_clone[0:1])
        reference_outputs[0, 2] = e2(inputs_clone[0:1])
        reference_outputs[1, 0] = e2(inputs_clone[1:2])
        reference_outputs[1, 1] = e4(inputs_clone[1:2])
        reference_outputs[2, 0] = e1(inputs_clone[2:3])
        reference_outputs[2, 2] = e3(inputs_clone[2:3])

        assert torch.allclose(expert_outputs, reference_outputs, atol=atol, rtol=0)
        proj = torch.randn(4, hidden_dim)
        loss = (expert_outputs[(0, 1, 1, 2), (0, 2, 1, 0)] * proj).sum()
        loss.backward()
        our_grad = inputs.grad.data.cpu().clone()

        reference_loss = (reference_outputs[(0, 1, 1, 2), (0, 2, 1, 0)] * proj).sum()
        reference_loss.backward()
        reference_grad = inputs_clone.grad.data.cpu().clone()
        assert torch.allclose(our_grad, reference_grad, atol=atol, rtol=0)


@pytest.mark.forked
def test_remote_module_call(hidden_dim=16):
    with background_server(
        num_experts=1,
        device="cpu",
        expert_cls="ffn",
        num_handlers=1,
        hidden_dim=hidden_dim,
        optim_cls=None,
    ) as server_peer_info:
        dht = DHT(initial_peers=server_peer_info.addrs, start=True)
        real_expert, fake_expert = create_remote_experts(
            [
                ExpertInfo(uid="expert.0", peer_id=server_peer_info.peer_id),
                ExpertInfo(uid="oiasfjiasjf", peer_id=server_peer_info.peer_id),
            ],
            dht=dht,
        )
        out1 = real_expert(torch.randn(1, hidden_dim))
        assert out1.shape == (1, hidden_dim)
        dummy_x = torch.randn(3, hidden_dim, requires_grad=True)
        out3 = real_expert(dummy_x)
        assert out3.shape == (3, hidden_dim)
        out3_again = real_expert(dummy_x[1:])
        assert torch.allclose(out3_again, out3[1:], atol=1e-5, rtol=0)
        out3_again.norm().backward()
        assert dummy_x.grad is not None and dummy_x.grad.norm() > 0

        try:
            real_expert(torch.randn(3, 11))
        except P2PHandlerError as e:
            assert str(11) in repr(e), "Exception must relay the remote server error (i.e. incorrect dimensions)"
        with pytest.raises(P2PHandlerError):
            fake_expert(dummy_x)

        # check that the server is still alive after processing a malformed request
        out3_yet_again = real_expert(dummy_x[1:])
        assert torch.allclose(out3_yet_again, out3[1:], atol=1e-5, rtol=0)


@pytest.mark.forked
def test_beam_search_correctness():
    all_expert_uids = [f"ffn.{5 + i}.{10 + j}.{15 + k}" for i in range(10) for j in range(10) for k in range(10)]
    dht = DHT(start=True)
    assert all(declare_experts(dht, all_expert_uids, expiration_time=get_dht_time() + 30))

    dmoe = RemoteMixtureOfExperts(in_features=32, grid_size=(32, 32, 32), dht=dht, k_best=4, uid_prefix="ffn.")

    for _ in range(25):
        input = torch.randn(32)
        grid_scores = dmoe.proj(input).split_with_sizes(dmoe.beam_search.grid_size, dim=-1)

        chosen_experts = dmoe.beam_search.find_best_experts(
            [tensor.detach().numpy() for tensor in grid_scores], beam_size=dmoe.k_best
        )
        chosen_scores = dmoe.compute_expert_scores([dim_scores[None] for dim_scores in grid_scores], [chosen_experts])[
            0
        ]
        our_best_scores = list(chosen_scores.cpu().detach().numpy())

        # reference: independently find :beam_size: best experts with exhaustive search
        all_scores = dmoe.compute_expert_scores(
            [dim_scores.unsqueeze(0) for dim_scores in grid_scores],
            [[RemoteExpert(ExpertInfo(uid, None), None) for uid in all_expert_uids]],
        )[0]
        true_best_scores = sorted(all_scores.cpu().detach().numpy(), reverse=True)[: len(chosen_experts)]

        assert np.allclose(true_best_scores, our_best_scores)


@pytest.mark.forked
def test_determinism(hidden_dim=16):
    atol = 1e-5

    xx = torch.randn(32, hidden_dim, requires_grad=True)
    mask = torch.randint(0, 1, (32, hidden_dim))

    with background_server(
        num_experts=1,
        device="cpu",
        expert_cls="det_dropout",
        num_handlers=1,
        hidden_dim=hidden_dim,
        optim_cls=None,
    ) as server_peer_info:
        dht = DHT(initial_peers=server_peer_info.addrs, start=True)
        expert = create_remote_experts(
            [ExpertInfo(uid="expert.0", peer_id=server_peer_info.peer_id)],
            dht=dht,
        )[0]

        out = expert(xx, mask)
        out_rerun = expert(xx, mask)

        (grad,) = torch.autograd.grad(out.sum(), xx, retain_graph=True)
        (grad_rerun,) = torch.autograd.grad(out_rerun.sum(), xx, retain_graph=True)

    assert torch.allclose(out, out_rerun, atol=atol, rtol=0), "Dropout layer outputs are non-deterministic."
    assert torch.allclose(grad, grad_rerun, atol=atol, rtol=0), "Gradients are non-deterministic."


@pytest.mark.forked
def test_compute_expert_scores():
    try:
        dht = DHT(start=True)
        moe = RemoteMixtureOfExperts(
            dht=dht, in_features=16, grid_size=(40,), k_best=4, k_min=1, timeout_after_k_min=1, uid_prefix="expert."
        )
        gx, gy = torch.randn(4, 5, requires_grad=True), torch.randn(4, 3, requires_grad=True)
        ii = [[4, 0, 2], [3, 1, 1, 1, 3], [0], [3, 2]]
        jj = [[2, 2, 1], [0, 1, 2, 0, 1], [0], [1, 2]]
        batch_experts = [
            [
                RemoteExpert(ExpertInfo(f"expert.{ii[batch_i][expert_i]}.{jj[batch_i][expert_i]}", None), None)
                for expert_i in range(len(ii[batch_i]))
            ]
            for batch_i in range(len(ii))
        ]  # note: these experts do not exist on server, we use them only to test compute_expert_scores
        logits = moe.compute_expert_scores([gx, gy], batch_experts)
        torch.softmax(logits, dim=-1).norm(dim=-1).mean().backward()
        assert gx.grad.norm().item() > 0 and gy.grad.norm().item(), "compute_expert_scores didn't backprop"

        for batch_i in range(len(ii)):
            for expert_i in range(len(ii[batch_i])):
                assert torch.allclose(
                    logits[batch_i, expert_i], gx[batch_i, ii[batch_i][expert_i]] + gy[batch_i, jj[batch_i][expert_i]]
                ), "compute_expert_scores returned incorrect score"
    finally:
        dht.shutdown()


@pytest.mark.forked
def test_client_anomaly_detection():
    HID_DIM = 16

    experts = {}
    for i in range(4):
        expert = name_to_block["ffn"](HID_DIM)
        experts[f"expert.{i}"] = ModuleBackend(
            name=f"expert.{i}",
            module=expert,
            optimizer=torch.optim.Adam(expert.parameters()),
            args_schema=(BatchTensorDescriptor(HID_DIM),),
            outputs_schema=BatchTensorDescriptor(HID_DIM),
            max_batch_size=16,
        )

    experts["expert.3"].module.ffn.weight.data[0, 0] = float("nan")

    dht = DHT(start=True)
    server = Server(dht, experts, num_connection_handlers=1)
    server.start()
    try:
        server.ready.wait()
        client_side_dht = DHT(initial_peers=dht.get_visible_maddrs(), start=True)

        dmoe = RemoteMixtureOfExperts(
            in_features=16, grid_size=(3,), dht=client_side_dht, k_best=3, uid_prefix="expert.", detect_anomalies=True
        )

        input = torch.randn(1, 16)
        input[0, 0] = float("nan")

        with pytest.raises(ValueError):
            dmoe(input)

        input[0, 0] = 0
        output = dmoe(input)

        inf_loss = float("inf") * output.sum()
        with pytest.raises(ValueError):
            inf_loss.backward()

        dmoe = RemoteMixtureOfExperts(
            in_features=16, grid_size=(4,), dht=client_side_dht, k_best=4, uid_prefix="expert.", detect_anomalies=True
        )
        output = dmoe(input)
        assert output.isfinite().all()

    finally:
        server.shutdown()
