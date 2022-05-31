import time
from typing import Callable, Optional, Union

import torch
from torch.distributed.distributed_c10d import _get_default_group, _get_default_store

from hivemind.dht import DHT
from hivemind.optim.grad_scaler import GradScaler
from hivemind.optim.optimizer import Optimizer
from hivemind.optim.state_averager import OptimizerFactory, Parameters, ParamGroups, TorchOptimizer, TrainingStateAverager
from hivemind.utils import get_logger

logger = get_logger(__name__)


class DDPOptimizer(Optimizer):
    _DDP_LEADER_RANK = 0
    _BROADCAST_BUFFER_SIZE = 250 * 1024 ** 2

    @staticmethod
    def is_ddp_enabled():
        return torch.distributed.is_initialized()

    @staticmethod
    def is_ddp_leader():
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == DDPOptimizer._DDP_LEADER_RANK

    def __init__(
        self,
        *,
        dht: Optional[DHT] = None,
        optimizer: Union[TorchOptimizer, OptimizerFactory],
        params: Optional[Union[Parameters, ParamGroups]] = None,
        reuse_grad_buffers: bool = False,
        use_local_updates: bool = False,
        **kwargs
    ):
        if self.is_ddp_leader() != (dht is not None):
            class_name = self.__class__.__name__
            raise ValueError(
                f"{class_name}(dht=...) is expected to be a hivemind.DHT instance "
                f"if {class_name}.is_ddp_leader(), None otherwise. "
                f"Please write code as follows:\n\n"
                f"if {class_name}.is_ddp_leader():\n"
                f"    dht = hivemind.DHT(...)\n"
                f"else:\n"
                f"    dht = None\n"
                f"optimizer = {class_name}(dht=dht, ...)"
            )

        if self.is_ddp_leader():
            super().__init__(
                dht,
                optimizer,
                params,
                reuse_grad_buffers,
                use_local_updates,
                **kwargs
            )
            self._main_parameters = self.state_averager.main_parameters
        else:
            self._param_groups, self._main_parameters, _ = TrainingStateAverager.check_params(optimizer, params)
            self.reuse_grad_buffers, self.use_local_updates = reuse_grad_buffers, use_local_updates

        self._checksum_counter = 0
        self._prev_version = self._prev_epoch = -1
        self._sync_among_ddp_ranks()

        # Collect fields of DDPOptimizer and its descendants
        self._ddp_aware_fields = set(self.__dict__.keys())
        for klass in self.__mro__:
            self._ddp_aware_fields.update(klass.__dict__.keys())
            if klass is DDPOptimizer:
                break

    def __getattribute__(self, name: str):
        """
        This works as usual on leaders, but denies access to non DDP-aware fields
        (i.e., fields defined in DDPOptimizer ancestors) on followers.
        """

        if (
            not name.startswith("_") and
            name not in self._ddp_aware_fields and
            not DDPOptimizer.is_ddp_leader()
        ):
            raise RuntimeError(
                f"{self.__class__.__name__}.{name} is only available on the DDP leader. "
                f"Please access it only if DDPOptimizer.is_ddp_leader() == True"
            )

        return super().__getattribute__(name)

    def is_alive(self) -> bool:
        # On followers, this always returns False since there's nothing to shut down in __del__()
        return self.is_ddp_leader() and super().is_alive()

    def _compute_state_version(self) -> int:
        """Return a non-decreasing integer that goes up whenever model params and/or buffers were updated"""

        assert self.is_ddp_leader()
        return sum(state["step"] for state in self.opt.state.values())

    def _has_updated_params_after_sync(self) -> bool:
        if not self.is_ddp_enabled():
            return False

        store = _get_default_store()
        if self.is_ddp_leader():
            current_version = self._compute_state_version()
            if current_version == self._prev_version and self.local_epoch > self._prev_epoch + 1:
                logger.warning("Model state version has not changed during a full epoch; "
                               "broadcasting parameters between torch.distributed synchronization may be broken")

            should_broadcast = (current_version != self._prev_version or self.local_epoch > self._prev_epoch + 1)

            store.set(f"_hivemind_should_broadcast_state", str(int(should_broadcast)))
            torch.distributed.barrier()
            return should_broadcast
        else:
            torch.distributed.barrier()
            raw_should_broadcast = store.get(f"_hivemind_should_broadcast_state")
            return bool(int(raw_should_broadcast))

    def _sync_among_ddp_ranks(self) -> None:
        """Synchronize model params and buffers from the DDP leader"""

        if not self.is_ddp_enabled():
            return

        t_start = time.perf_counter()
        with torch.no_grad():
            torch.distributed._broadcast_coalesced(
                _get_default_group(), self._main_parameters, self._BROADCAST_BUFFER_SIZE, self._DDP_LEADER_RANK
            )
        if self.is_ddp_leader():
            self._prev_version = self._compute_state_version()
            self._prev_epoch = self.local_epoch
            elapsed = time.perf_counter() - t_start
            logger.debug(f"Broadcasting leader params among DDP ranks took {elapsed:.2f} sec")

    def step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        grad_scaler: Optional[GradScaler] = None,
    ):
        if self.is_ddp_leader():
            loss = super().step(closure, batch_size, grad_scaler)

        if self._has_updated_params_after_sync():
            self._sync_among_ddp_ranks()
        else:
            logger.debug("No need to broadcast leader params among DDP ranks")

        if self.is_ddp_enabled():
            self._checksum_counter += 1
            if self._checksum_counter % 100 == 0:
                rank = torch.distributed.get_rank()
                checksum = sum(p.sum().item() for p in self._main_parameters)
                logger.debug(f"Parameter checksum (ddp_rank={rank}): {float(checksum)}")

        return loss if self.is_ddp_leader() else None

    def load_state_from_peers(self, **kwargs) -> None:
        if self.is_ddp_leader():
            super().load_state_from_peers(**kwargs)

        self._sync_among_ddp_ranks()

    def load_state_dict(self, state_dict: dict) -> None:
        if self.is_ddp_leader():
            super().load_state_dict(state_dict)

        self._sync_among_ddp_ranks()

    @property
    def param_groups(self) -> ParamGroups:
        if self.is_ddp_leader():
            return super().param_groups
        else:
            return self._param_groups

    def zero_grad(self, set_to_none: bool = False):
        # We explicitly define this method to mark that it should be available on the DDP followers
        super().zero_grad(set_to_none)

    def shutdown(self):
        if self.is_ddp_leader():
            super().shutdown()
