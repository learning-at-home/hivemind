import torch
import tesseract
from test_utils.run_server import background_server


def test_remote_module_call():
    """ Check that remote_module_call returns correct outputs and gradients if called directly """
    num_experts = 8
    k_min = 1
    timeout_after_k_min = None
    backward_k_min = 1
    timeout_total = None
    backward_timeout = None

    xx = torch.randn(32, 1024, requires_grad=True)
    logits = torch.randn(3, requires_grad=True)
    random_proj = torch.randn_like(xx)

    with background_server(num_experts=num_experts, no_optimizer=True, no_network=True) as localhost, server_port:
        experts = [tesseract.RemoteExpert(uid=f'expert.{i}', port=server_port) for i in range(num_experts)]
        moe_output, = tesseract.client.moe._RemoteMoECall.apply(
            logits, experts[:len(logits)], k_min, timeout_after_k_min, backward_k_min, timeout_total, backward_timeout,
            [(None,), {}], xx)

        grad_xx_moe, = torch.autograd.grad(torch.sum(random_proj * moe_output), xx, retain_graph=True)
        grad_logits_moe, = torch.autograd.grad(torch.sum(random_proj * moe_output), logits, retain_graph=True)

        # reference outputs: call all experts manually and average their outputs with softmax probabilities
        probs = torch.softmax(logits, 0)
        outs = [expert(xx) for expert in experts[:3]]
        manual_output = sum(p * x for p, x in zip(probs, outs))
        grad_xx_manual, = torch.autograd.grad(torch.sum(random_proj * manual_output), xx, retain_graph=True)
        grad_xx_manual_rerun, = torch.autograd.grad(torch.sum(random_proj * manual_output), xx, retain_graph=True)
        grad_logits_manual, = torch.autograd.grad(torch.sum(random_proj * manual_output), logits, retain_graph=True)

    assert torch.allclose(moe_output, manual_output), "_RemoteMoECall returned incorrect output"
    assert torch.allclose(grad_xx_manual, grad_xx_manual_rerun), "Experts are non-deterministic. This test is only " \
                                                                 "valid for deterministic experts"
    assert torch.allclose(grad_xx_moe, grad_xx_manual, rtol=1e-3, atol=1e-6), "incorrect gradient w.r.t. input"
    assert torch.allclose(grad_logits_moe, grad_logits_manual, rtol=1e-3, atol=1e-6), "incorrect gradient w.r.t. logits"


if __name__ == '__main__':
    test_remote_module_call()
