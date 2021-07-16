import importlib
import os
from typing import Callable, Type

import torch
import torch.nn as nn

from hivemind.moe.server.layers import name_to_block, name_to_input


def add_custom_models_from_file(path: str):
    spec = importlib.util.spec_from_file_location("custom_module", os.path.abspath(path))
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)


def register_expert_class(name: str, sample_input: Callable[[int, int], torch.tensor]):
    """
    Adds a custom user expert to hivemind server.
    :param name: the name of the expert. It shouldn't coincide with existing modules\
        ('ffn', 'transformer', 'nop', 'det_dropout')
    :param sample_input: a function which gets batch_size and hid_dim and outputs a \
        sample of an input in the module
    :unchanged module
    """

    def _register_expert_class(custom_class: Type[nn.Module]):
        if name in name_to_block or name in name_to_input:
            raise RuntimeError("The class might already exist or be added twice")
        name_to_block[name] = custom_class
        name_to_input[name] = sample_input

        return custom_class

    return _register_expert_class
