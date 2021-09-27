import contextlib
from typing import Dict

import torch
from torch.cuda.amp import GradScaler as TorchGradScaler
from torch.cuda.amp.grad_scaler import _refresh_per_optimizer_state
from torch.optim import Optimizer

from hivemind.optim.base import DecentralizedOptimizerBase
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


class HivemindGradScaler(TorchGradScaler):
    """A thin wrapper over GradScaler that supports hivemind-style training with CollaborativeOptimizer and others"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_running_global_step = False
        self._optimizer_states_to_reset = set()

    @contextlib.contextmanager
    def running_global_step(self):
        was_running, self._is_running_global_step = self._is_running_global_step, True
        try:
            yield
        finally:
            self._is_running_global_step = was_running

    def unscale_(self, optimizer: Optimizer) -> bool:
        assert isinstance(optimizer, DecentralizedOptimizerBase)
        if self._is_running_global_step:
            super().unscale_(optimizer.opt)
            return True
        else:
            self._check_inf_per_device(optimizer.opt)
            self._optimizer_states_to_reset.add(id(optimizer))
            return False

    def step(self, optimizer, *args, **kwargs):
        assert isinstance(optimizer, DecentralizedOptimizerBase)
        if self._is_running_global_step:
            if self.are_grads_finite(optimizer):
                super().step(optimizer.opt, *args, **kwargs)
            else:
                logger.warning("Skipping global step due to gradient over/underflow")
            return True
        else:
            super().step(optimizer)
            self._optimizer_states_to_reset.add(id(optimizer))
            return False

    def update(self, new_scale=None):
        total_infs = 0
        for optimizer_state in self._per_optimizer_states.values():
            total_infs += sum(v.item() for v in optimizer_state["found_inf_per_device"].values())

        if self._is_running_global_step or total_infs != 0:
            # note: we update either during actual optimizer step or if we need to reduce scale due to NaN
            super().update(new_scale)
            return True
        else:
            for opt_id in self._optimizer_states_to_reset:
                self._per_optimizer_states[opt_id] = _refresh_per_optimizer_state()
            self._optimizer_states_to_reset.clear()
            return False

    def _unscale_grads_(
        self, optimizer: Optimizer, inv_scale: torch.Tensor, found_inf: torch.Tensor, allow_fp16: bool
    ) -> Dict[torch.device, torch.Tensor]:
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, allow_fp16=True)

    def are_grads_finite(self, optimizer: DecentralizedOptimizerBase):
        assert isinstance(optimizer, DecentralizedOptimizerBase)
        return not sum(v.item() for v in self._check_inf_per_device(optimizer.opt).values())
