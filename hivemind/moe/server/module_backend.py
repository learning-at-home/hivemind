from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from hivemind.moe.server.task_pool import TaskPool
from hivemind.utils.logging import get_logger
from hivemind.utils.nested import nested_compare, nested_flatten, nested_map, nested_pack
from hivemind.utils.tensor_descr import DUMMY_BATCH_SIZE, BatchTensorDescriptor

LRSchedulerBase = getattr(torch.optim.lr_scheduler, "_LRScheduler", None)
logger = get_logger(__name__)


class ModuleBackend:
    """
    ModuleBackend is a wrapper around torch module that allows it to run tasks asynchronously with Runtime
    By default, ModuleBackend handles three types of requests:

     - forward - receive inputs and compute outputs. Concurrent requests will be batched for better GPU utilization.
     - backward - receive gradients w.r.t. outputs, compute gradients w.r.t. inputs and **update expert**. Also batched.
     - get_info - return expert metadata. Not batched.

    :param module: nn.Module to be wrapped into a backend. Arbitrary pytorch module with a few limitations:

     - Experts must always receive the same set of args and kwargs and produce output tensors of same type
     - All args, kwargs and outputs must be **tensors** where 0-th dimension represents to batch size
     - We recommend using experts that are ~invariant to the order in which they process batches
     - Using randomness (e.g. Dropout) leads to different samples at forward and backward. If you want consistency,
        you should explicitly register these random variables as model inputs or outputs.
        See hivemind.utils.custom_layers.DeterministicDropout for an example

    :param optimizer: torch optimizer to be applied on every backward call
    :param scheduler: a function to create the learning rate scheduler for the expert
    :param args_schema: description of positional arguments to expert.forward, list of BatchTensorProto
    :param kwargs_schema: description of keyword arguments to expert.forward, dict of BatchTensorProto
    :param outputs_schema: description of outputs from expert.forward, nested structure of BatchTensorProto
    :param kwargs: extra parameters to be forwarded into TaskPool.__init__
    """

    def __init__(
        self,
        name: str,
        module: nn.Module,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LRSchedulerBase] = None,
        args_schema: Tuple[BatchTensorDescriptor, ...] = None,
        kwargs_schema: Dict[str, BatchTensorDescriptor] = None,
        outputs_schema: Union[BatchTensorDescriptor, Tuple[BatchTensorDescriptor, ...]] = None,
        **kwargs,
    ):
        super().__init__()
        self.name, self.module, self.optimizer, self.scheduler = name, module, optimizer, scheduler

        self.args_schema = args_schema = tuple(args_schema or ())
        self.kwargs_schema = kwargs_schema = dict(kwargs_schema or {})
        assert args_schema or kwargs_schema, (
            f"Module must take at least one positional or keyword input."
            " Did you forget to provide args_schema/kwargs_schema?"
        )
        assert optimizer is not None or scheduler is None, "scheduler should only be used if optimizer is not None"

        if outputs_schema is None:
            # run expert once to get outputs schema
            dummy_args = tuple(sample.make_zeros(DUMMY_BATCH_SIZE) for sample in args_schema)
            dummy_kwargs = {key: sample.make_zeros(DUMMY_BATCH_SIZE) for key, sample in kwargs_schema.items()}
            dummy_outputs = self.module(*dummy_args, **dummy_kwargs)
            outputs_schema = nested_map(BatchTensorDescriptor.from_tensor, dummy_outputs)

        self.forward_schema = (self.args_schema, self.kwargs_schema)  # inputs for forward
        self.outputs_schema = outputs_schema  # outputs from forward

        self.backward_schema = (self.forward_schema, self.outputs_schema)  # inputs to backward
        self.grad_inputs_schema = self.forward_schema  # outputs from backward
        self.forward_pool = TaskPool(self.forward, name=f"{self.name}_forward", **kwargs)
        self.backward_pool = TaskPool(self.backward, name=f"{self.name}_backward", **kwargs)

    def forward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Apply forward pass to an aggregated batch of requests. Used by Runtime, do not call this manually;
        To submit a request for asynchronous processing, please use ``ModuleBackend.forward_pool.submit_task``.

        .. warning: if the underlying module performs non-gradient updates (e.g. batchnorm), it will be updated twice:
           once during forward pass, and again during backward. This behavior is similar to gradient checkpointing.

        Subclassing:
           This method receives a sequence of torch tensors following ``nested_flatten(self.forward_schema)``;
           It should return gradients w.r.t. inputs that follow ``nested_flatten(self.outputs_schema)``;
        """
        args, kwargs = nested_pack(inputs, structure=self.forward_schema)

        if args[0].shape[0] == 0:
            raise RuntimeError("Batch should contain more than 0 samples")

        with torch.no_grad():
            outputs = self.module(*args, **kwargs)

        # Note: TaskPool requires function to accept and return a flat tuple of values, we pack/unpack it on client side
        return tuple(nested_flatten(outputs))

    def backward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Apply backward pass to an aggregated batch of requests. Used by Runtime, do not call this manually
        To submit a request for asynchronous processing, please use ``ModuleBackend.backward_pool.submit_task``.

        Subclassing:
           This method receives a sequence of torch tensors following ``nested_flatten(self.backward_schema)``;

           It should return gradients w.r.t. inputs that follow ``nested_flatten(self.forward_schema)``;

           Runtime doesn't guarantee that backward will be performed in the same order and for the same data
           as forward, so we recommend stateless backward pass that re-runs expert forward pass inside backward.

           Please make sure to call ``ModuleBackend.on_backward`` after each call to backward
        """
        (args, kwargs), grad_outputs = nested_pack(inputs, structure=self.backward_schema)

        with torch.enable_grad():
            args = [
                tensor.detach().requires_grad_(True) if tensor.is_floating_point() else tensor.detach()
                for tensor in args
            ]
            kwargs = {
                input_key: (tensor.detach().requires_grad_(True) if tensor.is_floating_point() else tensor.detach())
                for input_key, tensor in kwargs.items()
            }

            batch_size = args[0].size(0)

            outputs = self.module(*args, **kwargs)
            assert nested_compare(outputs, grad_outputs), "outputs and grad_outputs must have the same structure"

            outputs_flat = tuple(nested_flatten(outputs))

            grad_outputs_flat = tuple(
                map(
                    lambda grad, out: grad.to(device=out.device, dtype=out.dtype, non_blocking=True),
                    nested_flatten(grad_outputs),
                    outputs_flat,
                )
            )
            torch.autograd.backward(
                outputs_flat, grad_tensors=grad_outputs_flat, create_graph=False, retain_graph=False
            )
            self.on_backward(batch_size)

        return tuple(
            x.grad if isinstance(x.grad, torch.Tensor) else torch.zeros_like(x) for x in nested_flatten((args, kwargs))
        )

    def on_backward(self, batch_size: int) -> None:
        """
        Train the expert for one step. This method is called by ``ModuleBackend.backward`` after computing gradients.
        """
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

    def state_dict(self) -> Dict:
        """Return the current state of the module, optimizer, and scheduler"""
        full_state = dict(module=self.module.state_dict())
        if self.optimizer is not None:
            full_state["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            full_state["scheduler"] = self.scheduler.state_dict()
        return full_state

    def load_state_dict(self, state_dict: Dict):
        self.module.load_state_dict(state_dict["module"])
        if self.optimizer is not None:
            if "optimizer" in state_dict:
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                logger.warning(f"Optimizer state missing for {self.name}")

        if self.scheduler is not None:
            if "scheduler" in state_dict:
                self.scheduler.load_state_dict(state_dict["scheduler"])
            else:
                logger.warning(f"Learning rate scheduler state missing for {self.name}")

    def get_info(self) -> Dict[str, Any]:
        """Get expert parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(
            forward_schema=self.forward_schema,
            outputs_schema=self.outputs_schema,
            keyword_names=tuple(self.kwargs_schema.keys()),
        )

    def get_pools(self) -> Sequence[TaskPool]:
        """return all pools that should be processed by ``Runtime``"""
        return self.forward_pool, self.backward_pool
