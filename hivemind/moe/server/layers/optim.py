import torch


class OptimizerWrapper(torch.optim.Optimizer):
    """A wrapper for pytorch.optim.Optimizer that forwards all methods to the wrapped optimizer"""

    def __init__(self, optim: torch.optim.Optimizer):
        super().__init__(optim.param_groups, optim.defaults)
        self.optim = optim

    @property
    def defaults(self):
        return self.optim.defaults

    @property
    def state(self):
        return self.optim.state

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.optim)})"

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        return self.optim.load_state_dict(state_dict)

    def step(self, *args, **kwargs):
        return self.optim.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        return self.optim.zero_grad(*args, **kwargs)

    @property
    def param_groups(self):
        return self.optim.param_groups

    def add_param_group(self, param_group: dict) -> None:
        return self.optim.add_param_group(param_group)


class ClippingWrapper(OptimizerWrapper):
    """A wrapper of torch.Optimizer that clips gradients by global norm before each step"""

    def __init__(self, optim: torch.optim.Optimizer, clip_grad_norm: float):
        super().__init__(optim)
        self.clip_grad_norm = clip_grad_norm

    def step(self, *args, **kwargs):
        parameters = tuple(param for group in self.param_groups for param in group["params"])
        torch.nn.utils.clip_grad_norm_(parameters, self.clip_grad_norm)
        return super().step(*args, **kwargs)
