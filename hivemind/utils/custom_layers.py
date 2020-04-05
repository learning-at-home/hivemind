import torch.autograd
import torch.nn as nn


class DeterministicDropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, keep_prob, mask):
        ctx.keep_prob = keep_prob
        ctx.mask = mask
        return x * mask / keep_prob

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.mask * grad_output / ctx.keep_prob, None, None


class DeterministicDropout(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.keep_prob = 1 - drop_prob

    def forward(self, x, mask):
        if self.training:
            return DeterministicDropoutFunction.apply(x, self.keep_prob, mask)
        else:
            return x
