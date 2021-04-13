import torch

name_to_block = {}
name_to_input = {}

from hivemind.server.layers.common import FeedforwardBlock, TransformerEncoderLayer, NopExpert
from hivemind.server.layers.dropout import DeterministicDropout, DeterministicDropoutNetwork
from hivemind.server.layers.lr_schedule import get_linear_schedule_with_warmup

from hivemind.server.layers.custom_experts import add_custom_models_from_file, register_expert_class

import hivemind.server.layers.common
import hivemind.server.layers.dropout

schedule_name_to_scheduler = {'linear': get_linear_schedule_with_warmup, 'none': None}
