import torch
import hivemind
from test_utils.run_server import background_server


def test_remote_module_call():
    """ Check that remote_module_call returns correct outputs and gradients if called directly """
    num_experts = 8
    k_min = 1
    timeout_after_k_min = None
    backward_k_min = 1
    timeout_total = None
    backward_timeout = None
    rtol = 1e-3
    atol = 1e-6

    xx = torch.randn(32, 1024, requires_grad=True)
    logits = torch.randn(3, requires_grad=True)
    random_proj = torch.randn_like(xx)

    with background_server(num_experts=num_experts, device='cpu',
                           no_optimizer=True, no_dht=True) as (localhost, server_port, dht_port):
        experts = [hivemind.RemoteExpert(
            uid=f'expert.{i}', port=server_port) for i in range(num_experts)]
        moe_output, = hivemind.client.moe._RemoteMoECall.apply(
            logits, experts[:len(
                logits)], k_min, timeout_after_k_min, backward_k_min, timeout_total, backward_timeout,
            [(None,), {}], xx)

        grad_xx_moe, = torch.autograd.grad(
            torch.sum(random_proj * moe_output), xx, retain_graph=True)
        grad_logits_moe, = torch.autograd.grad(
            torch.sum(random_proj * moe_output), logits, retain_graph=True)

        # reference outputs: call all experts manually and average their outputs with softmax probabilities
        probs = torch.softmax(logits, 0)
        outs = [expert(xx) for expert in experts[:3]]
        manual_output = sum(p * x for p, x in zip(probs, outs))
        grad_xx_manual, = torch.autograd.grad(
            torch.sum(random_proj * manual_output), xx, retain_graph=True)
        grad_xx_manual_rerun, = torch.autograd.grad(
            torch.sum(random_proj * manual_output), xx, retain_graph=True)
        grad_logits_manual, = torch.autograd.grad(
            torch.sum(random_proj * manual_output), logits, retain_graph=True)

    assert torch.allclose(grad_xx_manual, grad_xx_manual_rerun, rtol, atol), "Experts are non-deterministic. The test" \
                                                                             " is only valid for deterministic experts"
    assert torch.allclose(moe_output, manual_output, rtol,
                          atol), "_RemoteMoECall returned incorrect output"
    assert torch.allclose(grad_xx_moe, grad_xx_manual,
                          rtol, atol), "incorrect gradient w.r.t. input"
    assert torch.allclose(grad_logits_moe, grad_logits_manual,
                          rtol, atol), "incorrect gradient w.r.t. logits"


def test_determinism():
    rtol = 0
    atol = 1e-6

    xx = torch.randn(32, 1024, requires_grad=True)
    mask = torch.randint(0, 1, (32, 1024))

    with background_server(num_experts=1, device='cpu', expert_cls='det_dropout',
                           no_optimizer=True, no_dht=True) as (localhost, server_port, dht_port):
        expert = hivemind.RemoteExpert(uid=f'expert.0', port=server_port)

        out = expert(xx, mask)
        out_rerun = expert(xx, mask)

        grad, = torch.autograd.grad(out.sum(), xx, retain_graph=True)
        grad_rerun, = torch.autograd.grad(
            out_rerun.sum(), xx, retain_graph=True)

    assert torch.allclose(out, out_rerun, rtol,
                          atol), "Dropout layer outputs are non-deterministic."
    assert torch.allclose(grad, grad_rerun, rtol,
                          atol), "Gradients are non-deterministic."


def test_compute_expert_scores():
    try:
        dht = hivemind.DHT(port=hivemind.find_open_port(), start=True)
        moe = hivemind.client.moe.RemoteMixtureOfExperts(
            dht=dht, in_features=1024, grid_size=[40], k_best=4, k_min=1, timeout_after_k_min=1,
            uid_prefix='expert')
        gx, gy = torch.randn(4, 5, requires_grad=True), torch.torch.randn(
            4, 3, requires_grad=True)
        ii = [[4, 0, 2], [3, 1, 1, 1, 3], [0], [3, 2]]
        jj = [[2, 2, 1], [0, 1, 2, 0, 1], [0], [1, 2]]
        batch_experts = [
            [hivemind.RemoteExpert(
                uid=f'expert.{ii[b][e]}.{jj[b][e]}') for e in range(len(ii[b]))]
            for b in range(len(ii))
        ]  # note: these experts do not exists on server, we use them only to test moe compute_expert_scores
        logits = moe.compute_expert_scores([gx, gy], batch_experts)
        torch.softmax(logits, dim=-1).norm(dim=-1).mean().backward()
        assert gx.grad.norm().item() > 0 and gy.grad.norm(
        ).item(), "compute_expert_scores didn't backprop"

        for b in range(len(ii)):
            for e in range(len(ii[b])):
                assert torch.allclose(
                    logits[b, e], gx[b, ii[b][e]] + gy[b, jj[b][e]]), "compute_expert_scores returned incorrect score"
    finally:
        dht.shutdown()


if __name__ == '__main__':
    test_remote_module_call()
    test_compute_expert_scores()
    test_determinism()
