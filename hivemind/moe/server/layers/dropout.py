import torch.autograd
from torch import nn as nn

from hivemind.moe.server.layers.custom_experts import register_expert_class


class DeterministicDropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, keep_prob, mask):
        ctx.keep_prob = keep_prob
        ctx.save_for_backward(mask)
        return x * mask / keep_prob

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.saved_tensors[0] * grad_output / ctx.keep_prob, None, None


class DeterministicDropout(nn.Module):
    """
    Custom dropout layer which accepts dropout mask as an input (drop_prob is only used for scaling input activations).
    Can be used with RemoteExpert/ExpertBackend to ensure that dropout mask is the same at forward and backward steps
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.keep_prob = 1 - drop_prob

    def forward(self, x, mask):
        if self.training:
            return DeterministicDropoutFunction.apply(x, self.keep_prob, mask)
        else:
            return x


dropout_sample_input = lambda batch_size, hid_dim: (
    torch.empty((batch_size, hid_dim)),
    torch.randint(0, 1, (batch_size, hid_dim)),
)


@register_expert_class("det_dropout", dropout_sample_input)
class DeterministicDropoutNetwork(nn.Module):
    def __init__(self, hid_dim, dropout_prob=0.2):
        super().__init__()
        self.linear_in = nn.Linear(hid_dim, 2 * hid_dim)
        self.activation = nn.ReLU()
        self.dropout = DeterministicDropout(dropout_prob)
        self.linear_out = nn.Linear(2 * hid_dim, hid_dim)

    def forward(self, x, mask):
        x = self.linear_in(self.dropout(x, mask))
        return self.linear_out(self.activation(x))
