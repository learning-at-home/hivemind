from typing import Dict, Sequence, Any, Tuple, Union

import torch
from torch import nn

from .task_pool import TaskPool
from ..utils import nested_flatten, nested_pack, nested_compare, BatchTensorProto, DUMMY_BATCH_SIZE, nested_map


class ExpertBackend(nn.Module):
    """
    ExpertBackend is a wrapper around torch module that allows it to run tasks asynchronously with Runtime
    By default, ExpertBackend handles three types of requests:

     - forward - receive inputs and compute outputs. Concurrent requests will be batched for better GPU utilization.
     - backward - receive gradients w.r.t. outputs, compute gradients w.r.t. inputs and **update expert**. Also batched.
     - get_info - return expert metadata. Not batched.

    :param expert: nn.Module to be wrapped into a backend. Arbitrary pytorch module with a few limitations:

        - Experts must always receive the same set of \*args and \*\*kwargs and produce output tensors of same type
        - All \*args, \*\*kwargs and outputs must be **tensors** where 0-th dimension represents to batch size
        - We recommend using experts that are ~invariant to the order in which they process batches
        - Using randomness (e.g. Dropout) leads to different samples at forward and backward. If you want to ensure consistency,
            you should explicitly register these random variables as model outputs, so that they are sent back to the client.
            See hivemind.utils.custom_layers.DeterministicDropout for an example

    :param opt: torch optimizer to be applied on every backward call
    :param args_schema: description of positional arguments to expert.forward, list of BatchTensorProto
    :param kwargs_schema: description of keyword arguments to expert.forward, dict of BatchTensorProto
    :param outputs_schema: description of outputs from expert.forward, nested structure of BatchTensorProto
    :param kwargs: extra parameters to be forwarded into TaskPool.__init__
    """

    def __init__(self, name: str, expert: nn.Module, opt: torch.optim.Optimizer, *,
                 args_schema: Tuple[BatchTensorProto, ...] = None,
                 kwargs_schema: Dict[str, BatchTensorProto] = None,
                 outputs_schema: Union[BatchTensorProto, Tuple[BatchTensorProto, ...]] = None,
                 **kwargs):
        super().__init__()
        self.expert, self.opt, self.name = expert, opt, name

        self.args_schema = args_schema = tuple(args_schema or ())
        self.kwargs_schema = kwargs_schema = dict(kwargs_schema or {})
        assert args_schema or kwargs_schema, "expert must receive at least one positional or keyword input." \
                                             " Did you forget to provide args_schema/kwargs_schema?"

        if outputs_schema is None:
            # run expert once to get outputs schema
            dummy_args = tuple(sample.make_empty(DUMMY_BATCH_SIZE) for sample in args_schema)
            dummy_kwargs = {key: sample.make_empty(DUMMY_BATCH_SIZE) for key, sample in kwargs_schema.items()}
            dummy_outputs = self.expert(*dummy_args, **dummy_kwargs)
            outputs_schema = nested_map(BatchTensorProto.from_tensor, dummy_outputs)

        self.outputs_schema = outputs_schema
        self.forward_schema = (self.args_schema, self.kwargs_schema)
        self.backward_schema = (self.forward_schema, self.outputs_schema)  # original inputs and grad w.r.t. outputs
        self.forward_pool = TaskPool(self.forward, uid=f'{self.name}_forward', **kwargs)
        self.backward_pool = TaskPool(self.backward, uid=f'{self.name}_backward', **kwargs)

    def forward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Apply forward pass to an aggregated batch of requests. Used by Runtime, do not call this manually;
        To submit a request for asynchronous processing, please use ``ExpertBackend.forward_pool.submit_task``.

        Subclassing:
           This method receives a sequence of torch tensors following ``nested_flatten(self.forward_schema)``;

           It should return gradients w.r.t. inputs that follow ``nested_flatten(self.outputs_schema)``;

           .. todo we handle layer states (e.g. batchnorm stats) incorrectly, updating them twice.
           .. For now, either register all buffers as outputs or avoid stateful experts

        """
        args, kwargs = nested_pack(inputs, structure=self.forward_schema)

        with torch.no_grad():
            outputs = self.expert(*args, **kwargs)

        # Note: TaskPool requires function to accept and return a flat tuple of values, we pack/unpack it on client side
        return tuple(nested_flatten(outputs))

    def backward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Apply backward pass to an aggregated batch of requests. Used by Runtime, do not call this manually
        To submit a request for asynchronous processing, please use ``ExpertBackend.backward_pool.submit_task``.

        Subclassing:
           This method receives a sequence of torch tensors following ``nested_flatten(self.backward_schema)``;

           It should return gradients w.r.t. inputs that follow ``nested_flatten(self.forward_schema)``;

           Runtime doesn't guarantee that backward will be performed in the same order and for the same data
           as forward, so we recommend stateless backward pass that re-runs expert forward pass inside backward.

           .. todo correct state handling (see forward)

           Please make sure to call ``ExpertBackend.apply_gradients`` **within** this method, otherwise the expert will not train
        """
        (args, kwargs), grad_outputs = nested_pack(inputs, structure=self.backward_schema)

        with torch.enable_grad():
            args = [tensor.detach().requires_grad_(True) if tensor.dtype in (torch.half, torch.float, torch.double)
                    else tensor.detach() for tensor in args]
            kwargs = {input_key: (tensor.detach().requires_grad_(True) if tensor.dtype in (torch.half, torch.float, torch.double)
                                  else tensor.detach()) for input_key, tensor in kwargs.items()}

            outputs = self.expert(*args, **kwargs)
            assert nested_compare(outputs, grad_outputs), "outputs and grad_outputs must have the same structure"

            outputs_flat = tuple(nested_flatten(outputs))

            grad_outputs_flat = tuple(map(
                lambda grad, out: grad.to(device=out.device, dtype=out.dtype, non_blocking=True),
                nested_flatten(grad_outputs), outputs_flat))
            torch.autograd.backward(outputs_flat, grad_tensors=grad_outputs_flat,
                                    create_graph=False, retain_graph=False)
            self.apply_gradients()

        return tuple(x.grad if isinstance(x.grad, torch.Tensor) else torch.zeros_like(x)
                     for x in nested_flatten((args, kwargs)))

    def apply_gradients(self) -> None:
        """
        Train the expert for a single step. This method is called by ``ExpertBackend.backward`` after computing gradients.
        """
        self.opt.step()
        self.opt.zero_grad()

    def get_info(self) -> Dict[str, Any]:
        """ Get expert parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration. """
        return dict(forward_schema=self.forward_schema, outputs_schema=self.outputs_schema,
                    keyword_names=tuple(self.kwargs_schema.keys()))

    def get_pools(self) -> Sequence[TaskPool]:
        """ return all pools that should be processed by ``Runtime`` """
        return self.forward_pool, self.backward_pool
