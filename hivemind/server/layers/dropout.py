import torch.autograd
import torch.nn as nn


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
