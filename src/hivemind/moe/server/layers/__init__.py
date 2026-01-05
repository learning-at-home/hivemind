name_to_block = {}
name_to_input = {}

import hivemind.moe.server.layers.common
import hivemind.moe.server.layers.dropout
from hivemind.moe.server.layers.custom_experts import add_custom_models_from_file, register_expert_class
from hivemind.moe.server.layers.lr_schedule import get_linear_schedule_with_warmup

schedule_name_to_scheduler = {"linear": get_linear_schedule_with_warmup, "none": None}
