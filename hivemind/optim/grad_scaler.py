import contextlib
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler as TorchGradScaler
from torch.cuda.amp.grad_scaler import _refresh_per_optimizer_state, OptState
from torch.optim import Optimizer as TorchOptimizer

import hivemind
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


class GradScaler(TorchGradScaler):
    """
    A thin wrapper over pytorch GradScaler that supports hivemind-style training with CollaborativeOptimizer, namely:
    - bypass .unscale_ and .update calls in order to accumulate gradients over several steps
    - limit increasing gradient scale to only immediately after global optimizer steps
    - allow training with some or all master parameters in fp16
    """

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

    def unscale_(self, optimizer: TorchOptimizer) -> bool:
        assert isinstance(optimizer, (hivemind.Optimizer, hivemind.DecentralizedOptimizerBase))
        if self._is_running_global_step:
            super().unscale_(optimizer)
            self._per_optimizer_states[id(optimizer.opt)] = self._per_optimizer_states[id(optimizer)]
            return True
        else:
            self._check_inf_per_device(optimizer)
            self._optimizer_states_to_reset.add(id(optimizer))
            return False

    def step(self, optimizer: TorchOptimizer, *args, **kwargs) -> bool:
        if self._is_running_global_step:
            if self.are_grads_finite(optimizer):
                assert self._per_optimizer_states[id(optimizer.opt)]["stage"] == OptState.UNSCALED
                super().step(optimizer.opt, *args, **kwargs)
            else:
                logger.warning("Skipping global step due to gradient over/underflow")
            return True
        else:
            super().step(optimizer)
            self._optimizer_states_to_reset.add(id(optimizer))
            return False

    def update(self, new_scale: Optional[float] = None) -> bool:
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
        self, optimizer: TorchOptimizer, inv_scale: torch.Tensor, found_inf: torch.Tensor, allow_fp16: bool
    ) -> Dict[torch.device, torch.Tensor]:
        # note: the code below sets allow_fp16=True to allow training with master weights (partially) in fp16
        # inspired by: https://github.com/facebookresearch/fairscale/blob/945b9666/fairscale/optim/grad_scaler.py
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, allow_fp16=True)

    def are_grads_finite(self, optimizer: TorchOptimizer) -> bool:
        return not sum(v.item() for v in self._check_inf_per_device(optimizer).values())


class HivemindGradScaler(GradScaler):
    def __init__(self, *args, **kwargs):
        logger.warning("HivemindGradScaler was renamed to hivemind.GradScaler, this reference will be removed in v1.1")
        super().__init__(*args, **kwargs)
