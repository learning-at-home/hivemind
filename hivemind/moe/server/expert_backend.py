from typing import Any, Callable, Dict, Sequence, Tuple, Union

import torch
from torch import nn

from hivemind.moe.server.task_pool import TaskPool
from hivemind.utils.logging import get_logger
from hivemind.utils.nested import nested_compare, nested_flatten, nested_map, nested_pack
from hivemind.utils.tensor_descr import DUMMY_BATCH_SIZE, BatchTensorDescriptor

logger = get_logger(__name__)


class ExpertBackend:
    """
    ExpertBackend is a wrapper around torch module that allows it to run tasks asynchronously with Runtime
    By default, ExpertBackend handles three types of requests:

     - forward - receive inputs and compute outputs. Concurrent requests will be batched for better GPU utilization.
     - backward - receive gradients w.r.t. outputs, compute gradients w.r.t. inputs and **update expert**. Also batched.
     - get_info - return expert metadata. Not batched.

    :param expert: nn.Module to be wrapped into a backend. Arbitrary pytorch module with a few limitations:

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
    :param num_warmup_steps: the number of warmup steps for LR schedule
    :param num_total_steps: the total number of steps for LR schedule
    :param clip_grad_norm: maximum gradient norm used for clipping
    :param kwargs: extra parameters to be forwarded into TaskPool.__init__
    """

    def __init__(
        self,
        name: str,
        expert: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: Callable = None,
        args_schema: Tuple[BatchTensorDescriptor, ...] = None,
        kwargs_schema: Dict[str, BatchTensorDescriptor] = None,
        outputs_schema: Union[BatchTensorDescriptor, Tuple[BatchTensorDescriptor, ...]] = None,
        num_warmup_steps: int = None,
        num_total_steps: int = None,
        clip_grad_norm: float = None,
        **kwargs,
    ):
        super().__init__()
        self.expert, self.optimizer, self.name = expert, optimizer, name

        if scheduler is None:
            self.scheduler = None
        else:
            assert optimizer is not None and num_warmup_steps is not None and num_total_steps is not None
            self.scheduler = scheduler(self.optimizer, num_warmup_steps, num_total_steps)
        self.clip_grad_norm = clip_grad_norm

        self.args_schema = args_schema = tuple(args_schema or ())
        self.kwargs_schema = kwargs_schema = dict(kwargs_schema or {})
        assert args_schema or kwargs_schema, (
            "expert must receive at least one positional or keyword input."
            " Did you forget to provide args_schema/kwargs_schema?"
        )

        if outputs_schema is None:
            # run expert once to get outputs schema
            dummy_args = tuple(sample.make_empty(DUMMY_BATCH_SIZE) for sample in args_schema)
            dummy_kwargs = {key: sample.make_empty(DUMMY_BATCH_SIZE) for key, sample in kwargs_schema.items()}
            dummy_outputs = self.expert(*dummy_args, **dummy_kwargs)
            outputs_schema = nested_map(BatchTensorDescriptor.from_tensor, dummy_outputs)

        self.forward_schema = (self.args_schema, self.kwargs_schema)  # inputs for forward
        self.outputs_schema = outputs_schema  # outputs from forward

        self.backward_schema = (self.forward_schema, self.outputs_schema)  # inputs to backward
        self.grad_inputs_schema = self.forward_schema  # outputs from backward
        self.forward_pool = TaskPool(self.forward, name=f"{self.name}_forward", **kwargs)
        self.backward_pool = TaskPool(self.backward, name=f"{self.name}_backward", **kwargs)

        self.update_count = 0
        self.examples_processed = 0

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

        if args[0].shape[0] == 0:
            raise RuntimeError("Batch should contain more than 0 samples")

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

           Please make sure to call ``ExpertBackend.apply_gradients`` here, otherwise the expert will not train
        """
        (args, kwargs), grad_outputs = nested_pack(inputs, structure=self.backward_schema)

        with torch.enable_grad():
            args = [
                tensor.detach().requires_grad_(True)
                if tensor.dtype in (torch.half, torch.float, torch.double)
                else tensor.detach()
                for tensor in args
            ]
            kwargs = {
                input_key: (tensor.detach().requires_grad_(True) if tensor.is_floating_point() else tensor.detach())
                for input_key, tensor in kwargs.items()
            }

            batch_size = args[0].size(0)

            outputs = self.expert(*args, **kwargs)
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
            self.apply_gradients(batch_size)

        return tuple(
            x.grad if isinstance(x.grad, torch.Tensor) else torch.zeros_like(x) for x in nested_flatten((args, kwargs))
        )

    def apply_gradients(self, batch_size) -> None:
        """
        Train the expert for one step. This method is called by ``ExpertBackend.backward`` after computing gradients.
        """
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.expert.parameters(), self.clip_grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        self.update_count += 1
        self.examples_processed += batch_size

    def get_stats(self) -> Dict:
        """
        Return current expert training statistics (number of updates, number of processed examples after
        last optimizer step)
        """
        return {"updates": self.update_count, "examples_processed": self.examples_processed}

    def get_full_state(self) -> Dict:
        """
        Return the current state of the expert (including batch processing statistics)
        """
        full_state = {
            "stats": self.get_stats(),
            "model": self.expert.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": {} if self.scheduler is None else self.scheduler.state_dict(),
        }
        return full_state

    def load_full_state(self, state_dict: Dict):
        if "stats" in state_dict:
            self.update_count = state_dict["stats"]["updates"]
            self.examples_processed = state_dict["stats"]["examples_processed"]
        else:
            logger.warning(f"Batch processing stats missing for expert {self.name}")

        self.expert.load_state_dict(state_dict["model"])

        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        else:
            logger.warning(f"Optimizer state missing for expert {self.name}")

        if self.scheduler is not None and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        else:
            logger.warning(f"Learning rate scheduler state missing for expert {self.name}")

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
