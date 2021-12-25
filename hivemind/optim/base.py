from warnings import warn

import torch

from hivemind.dht import DHT


class DecentralizedOptimizerBase(torch.optim.Optimizer):
    """A shared interface for all hivemind optimizers. Cooperates with DHT peers to train a shared model"""

    def __init__(self, opt: torch.optim.Optimizer, dht: DHT):
        self.opt, self.dht = opt, dht
        warn(
            "DecentralizedOptimizerBase and its subclasses have been deprecated and will be removed "
            "in hivemind 1.1.0. Use hivemind.Optimizer instead",
            FutureWarning,
            stacklevel=2,
        )

    @property
    def state(self):
        return self.opt.state

    @property
    def param_groups(self):
        return self.opt.param_groups

    def add_param_group(self, param_group: dict) -> None:
        raise ValueError(
            f"{self.__class__.__name__} does not support calling add_param_group after creation."
            f"Please provide all parameter groups at init."
        )

    def state_dict(self) -> dict:
        return self.opt.state_dict()

    def load_state_dict(self, state_dict: dict):
        return self.opt.load_state_dict(state_dict)

    def __repr__(self):
        return f"{self.__class__.__name__}(opt={repr(self.opt)}, dht={repr(self.dht)})"

    def shutdown(self):
        raise NotImplementedError()
