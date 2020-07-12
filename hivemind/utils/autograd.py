"""
Temporary autograd extensions to enable inter-op parallelism during backward pass
Note: we should get rid of this module if https://github.com/pytorch/pytorch/pull/33157 reaches a pytorch release
"""
from itertools import chain
from typing import Tuple, Any
from concurrent.futures import Future

import numpy as np
import torch
import torch.autograd.function

from hivemind.utils.threading import run_in_background


class EmulatedAutogradContext(torch.autograd.function._ContextMethodMixin):
    """
    A special class that pretends to be pytorch autograd context. Used to circumvent limitatons of pytorch autograd,
    such as running several parallel backwards or transferring backward to a separate device.
    This class is not tested outside its use cases in RemoteMixtureOfExperts and we do not recommend using it elsewhere.
    """

    @property
    def saved_tensors(self):
        return tuple(self.to_save)


def run_isolated_forward(func: torch.autograd.Function, *args) -> Tuple[EmulatedAutogradContext, Any]:
    """
    run :func: in a detached pytorch graph, return *detached* function outputs and an EmulatedAutogradContext that
    can be used to run backward through the same graph (performed manually by the user).
    """
    ctx = EmulatedAutogradContext()
    # create detached copies of every input so that we can differentiate w.r.t. them without modifying actual variables
    args = tuple(x.detach().requires_grad_(x.requires_grad) if isinstance(x, torch.Tensor) else x for x in args)
    with torch.no_grad():
        return ctx, func.forward(ctx, *args)


def run_isolated_backward(func: torch.autograd.Function, ctx: EmulatedAutogradContext, *grad_outputs):
    """
    run backward pass for :func: in an isolated graph that was previously created through run_isolated_forward
    """
    with torch.no_grad():
        return func.backward(ctx, *grad_outputs)


def map_with_parallel_backward(
        func: torch.autograd.Function, *args_per_call: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...]]:
    """
    Apply an autograd function to several sets of inputs with two extra guarantees:
    (1) both forward and backward pass happens concurrently for each set of inputs
    (2) any operation dependent on any individual function will wait for all functions to finish
    :param func: torch autograd function to be called several times in parallel
    :param args_per_call: a sequence of tuples of arguments, each tuple corresponds to one function call
    :returns: a tuple of outputs from each func call

    Note: this function currently requires that all :func: calls succeed (i.e. do not raise an exception).
    """
    arg_counts = list(map(len, args_per_call))
    assert len(set(arg_counts)) == 1, "All input sets must have the same number of arguments"
    output_strides_ph = Future()
    flat_outputs: Tuple[torch.Tensor, ...] = _ParallelApplyFunction.apply(
        func, len(args_per_call), arg_counts[0], output_strides_ph, *chain(*args_per_call))
    output_strides = output_strides_ph.result()
    return tuple(flat_outputs[output_strides[i]: output_strides[i + 1]] for i in range(len(output_strides) - 1))


class _ParallelApplyFunction(torch.autograd.Function):
    """
    A special torch autograd function that runs another function several times in parallel.
    Please do not call this function directly. Use apply_with_parallel_backward instead.
    Unlike default pytorch behavior, the backward pass for each function will also happen in parallel.
    """

    @staticmethod
    def forward(ctx, func: torch.autograd.Function, num_calls: int, num_args_per_call: int,
                output_strides_ph: Future, *args_flat) -> Tuple[torch.Tensor, ...]:
        assert num_calls * num_args_per_call == len(args_flat)
        args_per_call = [args_flat[i * num_args_per_call: (i + 1) * num_args_per_call] for i in range(num_calls)]

        futures = [run_in_background(run_isolated_forward, func, *args) for args in args_per_call]

        contexts, outputs = zip(*[future.result() for future in futures])
        output_strides = np.cumsum([0] + list(map(len, outputs)))
        ctx._inner_func = func
        ctx._call_contexts = contexts
        ctx._output_strides = output_strides
        output_strides_ph.set_result(output_strides)
        return tuple(chain(*outputs))

    @staticmethod
    def backward(ctx, *grad_outputs_flat: torch.Tensor):
        func, contexts, output_strides = ctx._inner_func, ctx._call_contexts, ctx._output_strides
        grad_outputs_per_call = [grad_outputs_flat[output_strides[i]: output_strides[i + 1]]
                                 for i in range(len(contexts))]
        futures = [run_in_background(run_isolated_backward, func, context, *grads)
                   for context, grads in zip(contexts, grad_outputs_per_call)]
        flat_grads_wrt_input = tuple(grad for future in futures for grad in future.result())
        return (None, None, None, None, *flat_grads_wrt_input)
