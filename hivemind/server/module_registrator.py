import typing as tp
import functools
import torch
import torch.nn as nn

from hivemind.server.layers import name_to_block, name_to_input

def register_expert(name: str, sample_input: tp.Callable[[int, int], torch.tensor]):
    """
    Adds a custom user expert to hivemind server.
    :param name: the name of the expert. It shouldn't coincide with existing modules\
        ('ffn', 'transforme', 'nop', 'det_dropout')
    :param sample_input: a function which gets batch_size and hid_dim and outputs a \
        sample of an input in the module
    : unchanged module
    """
    def registrator(custom_class: tp.Type[nn.Module]):
        if name in name_to_block or name in name_to_input:
            raise RuntimeWarning("The class might already exist or be added twice")
        name_to_block[name] = custom_class
        name_to_input[name] = sample_input

        return custom_class
    return registrator
