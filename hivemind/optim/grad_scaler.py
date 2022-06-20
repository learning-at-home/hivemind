import contextlib
import threading
from copy import deepcopy
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler as TorchGradScaler
from torch.cuda.amp.grad_scaler import OptState, _refresh_per_optimizer_state
from torch.optim import Optimizer as TorchOptimizer

import hivemind
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


class GradScaler(TorchGradScaler):
    """
    A wrapper over pytorch GradScaler made specifically for training hivemind.Optimizer with reuse_grad_buffers=True.

    :note: if not using reuse_grad_buffers=True, one can and *should* train normally without this class, e.g. using
      standard PyTorch AMP or Apex. This custom GradScaler is more memory-efficient, but requires custom training code.

    hivemind.GradScaler makes 3 modifications to the regular PyTorch AMP:

    - bypass .unscale_ and .update calls in order to accumulate gradients over several steps
    - limit increasing gradient scale to only immediately after global optimizer steps
    - allow training with some or master parameters in float16

    :note: The above modiffications will be enabled automatically. One can (and should) use hivemind.GradScaler exactly
      as regular ``torch.amp.GradScaler``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_running_global_step = False
        self._is_ready_to_update = False
        self._inner_optimizer_states = {}
        self._optimizer_states_to_reset = set()
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def running_global_step(self):
        with self._lock:
            was_running, self._is_running_global_step = self._is_running_global_step, True
            try:
                yield
            finally:
                self._is_running_global_step = was_running

    def unscale_(self, optimizer: TorchOptimizer) -> bool:
        with self._lock:
            assert isinstance(optimizer, hivemind.Optimizer)
            if self._is_running_global_step:
                super().unscale_(optimizer)
                self._inner_optimizer_states[id(optimizer.opt)] = deepcopy(self._per_optimizer_states[id(optimizer)])
                # note: we store unscaled optimizer state in a separate dict and not in _per_optimizer_states in order
                # to avoid an edge case where full DPU peer encounters overflow in local gradients while averaging
                # offloaded gradients (i.e. after global unscale but before global step). Due to overflow, next call to
                # .update on user side would reset *all* optimizer states and cause .step to unscale gradients twice.
                # Offloaded optimizer is not affected by overflow in on-device gradients and should not be reset.
                return True
            else:
                self._check_inf_per_device(optimizer)
                self._optimizer_states_to_reset.add(id(optimizer))
                return False

    def step(self, optimizer: TorchOptimizer, *args, **kwargs) -> bool:
        if self._is_running_global_step and not isinstance(optimizer, hivemind.Optimizer):
            # ^-- invoked privately within hivemind optimizer
            inner_optimizer = optimizer
            with self._lock:
                if self._is_ready_to_update:
                    logger.warning("Please call grad_scaler.update() after each step")

                inner_optimizer_state = self._inner_optimizer_states.pop(id(inner_optimizer), None)
                if inner_optimizer_state is not None:
                    self._per_optimizer_states[id(inner_optimizer)] = inner_optimizer_state
                assert (
                    self._per_optimizer_states[id(inner_optimizer)]["stage"] == OptState.UNSCALED
                ), "InternalError: Optimizer should have called .unscale internally before invoking grad_scaler.step"
                if self.are_grads_finite(inner_optimizer, use_cached=True):
                    super().step(inner_optimizer, *args, **kwargs)
                else:
                    logger.warning("Skipping global step due to gradient over/underflow")
                self._is_ready_to_update = True
                return True
        else:
            super().step(optimizer)
            self._optimizer_states_to_reset.add(id(optimizer))
            return False

    def update(self, new_scale: Optional[float] = None) -> bool:
        with self._lock:
            total_infs = 0
            for optimizer_state in self._per_optimizer_states.values():
                total_infs += sum(v.item() for v in optimizer_state["found_inf_per_device"].values())

            if self._is_ready_to_update or total_infs != 0:
                # note: we update either during actual optimizer step or if we need to reduce scale due to NaN
                super().update(new_scale)
                self._is_ready_to_update = False
                return True
            else:
                for opt_id in self._optimizer_states_to_reset:
                    self._per_optimizer_states[opt_id] = _refresh_per_optimizer_state()
                self._optimizer_states_to_reset.clear()
                return False

    def _unscale_grads_(
        self, optimizer: TorchOptimizer, inv_scale: torch.Tensor, found_inf: torch.Tensor, allow_fp16: bool
    ) -> Dict[torch.device, torch.Tensor]:
        # note: the code below sets allow_fp16=True to allow training with master weights (partially) in fp16
        # inspired by: https://github.com/facebookresearch/fairscale/blob/945b9666/fairscale/optim/grad_scaler.py
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, allow_fp16=True)

    def are_grads_finite(self, optimizer: TorchOptimizer, use_cached: bool = False) -> bool:
        opt_dict = self._found_inf_per_device(optimizer) if use_cached else self._check_inf_per_device(optimizer)
        return not sum(v.item() for v in opt_dict.values())


class HivemindGradScaler(GradScaler):
    def __init__(self, *args, **kwargs):
        logger.warning("HivemindGradScaler was renamed to hivemind.GradScaler, this reference will be removed in v1.1")
        super().__init__(*args, **kwargs)
