from typing import Tuple, Any

import torch
from torch.autograd.function import _ContextMethodMixin


class EmulatedAutogradContext(_ContextMethodMixin):
    """
    A special class that pretends to be pytorch autograd context. Used to circumvent limitatons of pytorch autograd,
    such as running several parallel backwards or transferring backward to a separate device.
    This class is not tested outside its use cases in RemoteMixtureOfExperts and we do not recommend using it elsewhere.
    """
    @property
    def saved_tensors(self):
        return tuple(self.to_save)


def run_isolated_forward(func: torch.autograd.Function, *args, **kwargs) -> Tuple[EmulatedAutogradContext, Any]:
    """
    run :func: in a detached pytorch graph, return *detached* function outputs and an EmulatedAutogradContext that
    can be used to run backward through the same graph (manually by the user).
    """
    ctx = EmulatedAutogradContext()
    # create detached copies of every input so that we can differentiate w.r.t. them without modifying actual variables
    args = tuple(x.detach().requires_grad_(x.requires_grad) for x in args)
    kwargs = {k: x.detach().requires_grad_(x.requires_grad) for k, x in kwargs.items()}
    with torch.no_grad():
        return ctx, func.forward(ctx, *args, **kwargs)


def run_isolated_backward(func: torch.autograd.Function, ctx: EmulatedAutogradContext, *grad_outputs):
    """
    run backward pass for :func: in an isolated graph that was previously created through run_isolated_forward
    """
    with torch.no_grad():
        return func.backward(ctx, *grad_outputs)
