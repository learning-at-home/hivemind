from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import Linear

from hivemind import BatchTensorDescriptor, ExpertBackend
from hivemind.server.checkpoint_saver import store_experts, load_weights


def test_save_load_checkpoints():
    experts = {}
    expert = Linear(1, 1)
    opt = torch.optim.SGD(expert.parameters(), 0.0)
    expert_name = f'test_expert'
    args_schema = (BatchTensorDescriptor(1),)
    experts[expert_name] = ExpertBackend(name=expert_name, expert=expert, opt=opt,
                                         args_schema=args_schema,
                                         outputs_schema=BatchTensorDescriptor(1),
                                         max_batch_size=1,
                                         )
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        expert.weight.data[0] = 1
        store_experts(experts, tmp_path)
        expert.weight.data[0] = 2
        store_experts(experts, tmp_path)
        expert.weight.data[0] = 3
        store_experts(experts, tmp_path)

        checkpoints_dir = tmp_path / expert_name

        assert checkpoints_dir.exists()
        assert len(list(checkpoints_dir.iterdir())) == 3

        expert.weight.data[0] = 4

        load_weights(experts, tmp_path)
        assert expert.weight.data[0] == 3
