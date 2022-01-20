""" An extension of averager that supports common optimization use cases. """
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import chain
from threading import Event, Lock
from typing import Dict, Iterator, Optional, Sequence

import torch

from hivemind.averaging import DecentralizedAverager
from hivemind.compression import CompressionInfo, TensorRole
from hivemind.utils import get_logger, nested_flatten, nested_pack

logger = get_logger(__name__)


class TrainingAverager(DecentralizedAverager):
    """
    A high-level interface to DecentralizedAverager that averages trainable params or gradients for an optimizer.

    This averager implements a number of typical use cases that arise in collaborative optimization
    - averaging parameters or gradients or both (in future, this will support averaging learning rates as well)
    - this peer's weight (e.g. based on its batch size) can be specified via averager.step(weight=...)
    - when out of sync, the averager will load the entire optimizer state from an up-to-date peer

    :param opt: a pytorch optimizer to be averaged between peers (complete with model parameters)
    :param average_parameters: whether or not to average model parameters in self.step(...)
    :param average_gradients: whether or not to average model gradients in self.step(...)
    :param average_opt_statistics: if specified, average optimizer statistics with corresponding names in statedict
    :param initialize_optimizer: if True, this will run a speculative optimizer step with
      zero gradients to initialize all tensors. If False, please initialize the optimizer state manually.
    :param extra_tensors: if specified, these extra tensors will also be averaged and shared in load_state_from_peers.
    :note: you can use extra_tensors for averaging tensors that are updated outside of opt.step (e.g. batchnorm stats)
    :param kwargs: any additional parameters will be forwarded to DecentralizedAverager
    """

    def __init__(
        self,
        opt: torch.optim.Optimizer,
        *,
        average_parameters: bool,
        average_gradients: bool,
        average_opt_statistics: Sequence[str] = (),
        extra_tensors: Sequence[torch.Tensor] = (),
        parameter_names: Optional[Sequence[str]] = None,
        initialize_optimizer: bool = True,
        **kwargs
    ):
        if initialize_optimizer:
            initialize_optimizer_state(opt)  # note: this will run one optimizer step!
        if parameter_names is None:
            parameter_names = tuple(i for group in opt.param_groups for i in range(len(group["params"])))

        self.opt, self.extra_tensors, self.local_step = opt, tuple(extra_tensors), 0
        self.opt_statistics = tuple(average_opt_statistics)
        self.average_parameters, self.average_gradients = average_parameters, average_gradients
        self.parameter_names = parameter_names
        self.step_executor = ThreadPoolExecutor(max_workers=1)
        self.lock_averager_step = Lock()
        self.pending_updates_done = Event()
        self.pending_updates_done.set()

        with torch.no_grad():
            averaged_tensors = [tensor.detach().cpu().float().clone() for tensor in self.local_tensors()]

        super().__init__(averaged_tensors=averaged_tensors, tensor_infos=list(self.tensor_infos()), **kwargs)

    def step(self, data_lock: Optional[Lock] = None, wait: bool = True, **kwargs):
        """
        Average optimizer weights and gradients with peers.

        :param data_lock: averager locks it when model parameters are modified. Otherwise it's assumed that no model
        modifications occur during averaging step
        """
        if not wait:
            return self.step_executor.submit(self.step, data_lock, wait=True, **kwargs)

        # if data_lock is supplied, tensors might change during averaging, so we need to copy them
        use_old_local_tensors = data_lock is not None
        if data_lock is None:
            data_lock = nullcontext()

        local_tensors = list(self.local_tensors())
        with self.lock_averager_step, torch.no_grad():
            # fill averager's tensors with current local tensors
            self.pending_updates_done.clear()
            with data_lock, self.get_tensors() as averaged_tensors:
                if use_old_local_tensors:
                    old_local_tensors = tuple(x.cpu().float().clone() for x in local_tensors)
                assert len(local_tensors) == len(
                    averaged_tensors
                ), "The number of optimized parameters should not change."
                for averaged_tensor, local_tensor in zip(averaged_tensors, local_tensors):
                    averaged_tensor[...] = local_tensor.cpu().float()
            self.pending_updates_done.set()

            # find a group and hopefully average tensors with peers, use batch sizes as weights
            gathered = super().step(**kwargs)
            if gathered is not None:
                # load averaged tensors back into model
                self.pending_updates_done.clear()
                with data_lock, self.get_tensors() as averaged_tensors:
                    if len(averaged_tensors) != len(local_tensors):
                        raise RuntimeError("The number of optimized parameters should not change")

                    if use_old_local_tensors:
                        # since tensors might have changed, we subtract old_local_tensor and add averaged. This prevents
                        # losing local updates that might have occurred during averaging
                        for averaged_tensor, local_tensor, old_local_tensor in zip(
                            averaged_tensors, local_tensors, old_local_tensors
                        ):
                            averaged_tensor = averaged_tensor.to(
                                dtype=local_tensor.dtype, device=local_tensor.device, non_blocking=True
                            )
                            old_local_tensor = old_local_tensor.to(
                                dtype=local_tensor.dtype, device=local_tensor.device, non_blocking=True
                            )

                            local_tensor.add_(averaged_tensor - old_local_tensor)
                    else:
                        for averaged_tensor, local_tensor in zip(averaged_tensors, local_tensors):
                            local_tensor.copy_(averaged_tensor, non_blocking=True)
                self.pending_updates_done.set()

            self.local_step += 1
            return gathered

    def local_tensors(self) -> Iterator[torch.Tensor]:
        """Iterate local trainer's tensors that should be averaged with peers"""
        if self.average_parameters:
            for param_group in self.opt.param_groups:
                yield from param_group["params"]
        if self.average_gradients:
            for param_group in self.opt.param_groups:
                for param in param_group["params"]:
                    if param.grad is not None:
                        yield param.grad
                    else:
                        yield torch.zeros_like(param)
        for stats in self.opt_statistics:
            for param_group in self.opt.param_groups:
                for param in param_group["params"]:
                    yield self.opt.state[param][stats]
        yield from iter(self.extra_tensors)

    def tensor_infos(self):
        """Get CompressionInfo for each tensor, accounting for its role and specification"""
        params = tuple(param for param_group in self.opt.param_groups for param in param_group["params"])
        assert len(params) == len(self.parameter_names)
        if self.average_parameters:
            for param, key in zip(params, self.parameter_names):
                yield CompressionInfo.from_tensor(param, key=key, role=TensorRole.PARAMETER)
        if self.average_gradients:
            for param, key in zip(params, self.parameter_names):
                if param.grad is not None:
                    grad = param.grad if param.grad is not None else torch.zeros_like(param)
                    yield CompressionInfo.from_tensor(grad, key=key, role=TensorRole.GRADIENT)
        for stats in self.opt_statistics:
            for param, key in zip(params, self.parameter_names):
                yield CompressionInfo.from_tensor(
                    self.opt.state[param][stats], key=(key, stats), role=TensorRole.OPTIMIZER
                )
        for i, extra_tensor in enumerate(self.extra_tensors):
            yield CompressionInfo.from_tensor(extra_tensor, key=i, role=TensorRole.UNSPECIFIED)

    def get_current_state(self):
        """
        Get current model/optimizer state and when requested by a newbie peer. executed in the host process.
        :returns: a tuple of (serializable_small_metadata, sequence of torch tensors)
        """
        with torch.no_grad():
            optimized_parameters = tuple(
                param.detach().cpu() for param_group in self.opt.param_groups for param in param_group["params"]
            )
            parameter_infos = [
                CompressionInfo.from_tensor(param, key=key, role=TensorRole.PARAMETER)
                for param, key in zip(optimized_parameters, self.parameter_names)
            ]
            extra_tensors = tuple(tensor.detach().cpu() for tensor in self.extra_tensors)
            extra_infos = [
                CompressionInfo.from_tensor(extra_tensor, key=i, role=TensorRole.UNSPECIFIED)
                for i, extra_tensor in enumerate(extra_tensors)
            ]
            optimizer_metadata, optimizer_tensors = dump_optimizer_state(self.opt)
            optimizer_infos = [
                CompressionInfo.from_tensor(opt_tensor, key=i, role=TensorRole.OPTIMIZER)
                for i, opt_tensor in enumerate(optimizer_tensors)
            ]

        metadata = dict(step=self.local_step, group_bits=self.get_group_bits(), optimizer_metadata=optimizer_metadata)
        all_tensors = list(chain(optimized_parameters, extra_tensors, optimizer_tensors))
        all_tensor_infos = list(chain(parameter_infos, extra_infos, optimizer_infos))
        return metadata, all_tensors, all_tensor_infos

    def load_state_from_peers(self, **kwargs):
        """
        Attempt to download the latest optimizer state from peers and update trainer parameters/statistics.
        :returns: whether or the averager succeeded in loading parameters
        """
        parameters_and_extras = [param for param_group in self.opt.param_groups for param in param_group["params"]]
        parameters_and_extras.extend(self.extra_tensors)
        num_local_tensors = len(parameters_and_extras)

        loaded_state = super().load_state_from_peers(**kwargs)
        if loaded_state is None:
            return
        metadata, flat_tensors = loaded_state
        loaded_parameters_and_extras = flat_tensors[:num_local_tensors]
        loaded_opt_tensors = flat_tensors[num_local_tensors:]

        with torch.no_grad():
            for local_param, loaded_param in zip(parameters_and_extras, loaded_parameters_and_extras):
                local_param[...] = loaded_param
            load_optimizer_state(self.opt, metadata["optimizer_metadata"], loaded_opt_tensors)

        self.local_step = max(self.local_step, metadata["step"])


def initialize_optimizer_state(opt: torch.optim.Optimizer):
    for param_group in opt.param_groups:
        for param in param_group["params"]:
            if param.grad is None:
                (0 * param.sum()).backward()
    opt.step()


def dump_optimizer_state(opt: torch.optim.Optimizer):
    """Convert optimizer state into a format of DecentralizedAverager's get_current_state/load_state_from_peers"""
    with torch.no_grad():
        flat_metadata, flat_tensors = [], []
        for elem in nested_flatten(opt.state_dict()):
            if isinstance(elem, torch.Tensor):
                flat_metadata.append(dict(type="tensor", index=len(flat_tensors)))
                flat_tensors.append(elem.cpu())
            else:
                flat_metadata.append(dict(type="value", value=elem))
        return flat_metadata, flat_tensors


def load_optimizer_state(optimizer: torch.optim.Optimizer, flat_metadata: Dict, flat_tensors: Sequence[torch.Tensor]):
    flat_optimizer_state = []
    for elem in flat_metadata:
        if elem.get("type") == "tensor" and isinstance(elem.get("index"), int):
            flat_optimizer_state.append(flat_tensors[elem["index"]])
        elif elem.get("type") == "value" and "value" in elem:
            flat_optimizer_state.append(elem["value"])
    with torch.no_grad():
        try:
            return optimizer.load_state_dict(nested_pack(flat_optimizer_state, structure=optimizer.state_dict()))
        except StopIteration:
            return optimizer
