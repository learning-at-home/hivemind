from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import Linear

from hivemind import BatchTensorDescriptor, ExpertBackend
from hivemind.server.checkpoints import store_experts, load_weights

EXPERT_WEIGHT_UPDATES = 3
BACKWARD_PASSES_BEFORE_SAVE = 2
BACKWARD_PASSES_AFTER_SAVE = 2


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

        for i in range(1, EXPERT_WEIGHT_UPDATES + 1):
            expert.weight.data[0] = i
            store_experts(experts, tmp_path)

        checkpoints_dir = tmp_path / expert_name

        assert checkpoints_dir.exists()
        # include checkpoint_last.pt
        assert len(list(checkpoints_dir.iterdir())) == EXPERT_WEIGHT_UPDATES + 1

        expert.weight.data[0] = 0

        load_weights(experts, tmp_path)
        assert expert.weight.data[0] == EXPERT_WEIGHT_UPDATES


def test_restore_update_count():
    experts = {}
    expert = Linear(1, 1)
    opt = torch.optim.SGD(expert.parameters(), 0.0)
    expert_name = f'test_expert'
    args_schema = (BatchTensorDescriptor(1),)
    expert_backend = ExpertBackend(name=expert_name, expert=expert, opt=opt,
                                   args_schema=args_schema,
                                   outputs_schema=BatchTensorDescriptor(1),
                                   max_batch_size=1,
                                   )
    experts[expert_name] = expert_backend

    batch = torch.randn(1, 1)
    loss_grad = torch.randn(1, 1)

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for _ in range(BACKWARD_PASSES_BEFORE_SAVE):
            expert_backend.backward(batch, loss_grad)

        store_experts(experts, tmp_path)

        for _ in range(BACKWARD_PASSES_AFTER_SAVE):
            expert_backend.backward(batch, loss_grad)

        load_weights(experts, tmp_path)
        assert experts[expert_name].update_count == BACKWARD_PASSES_BEFORE_SAVE
