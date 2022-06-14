from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from torch.nn import Linear

from hivemind import BatchTensorDescriptor, ModuleBackend
from hivemind.moe.server.checkpoints import load_experts, store_experts
from hivemind.moe.server.layers.lr_schedule import get_linear_schedule_with_warmup

EXPERT_WEIGHT_UPDATES = 3
BACKWARD_PASSES_BEFORE_SAVE = 2
BACKWARD_PASSES_AFTER_SAVE = 2
EXPERT_NAME = "test_expert"
PEAK_LR = 1.0


@pytest.fixture
def example_experts():
    expert = Linear(1, 1)
    opt = torch.optim.SGD(expert.parameters(), PEAK_LR)

    args_schema = (BatchTensorDescriptor(1),)
    expert_backend = ModuleBackend(
        name=EXPERT_NAME,
        module=expert,
        optimizer=opt,
        scheduler=get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=BACKWARD_PASSES_BEFORE_SAVE,
            num_training_steps=BACKWARD_PASSES_BEFORE_SAVE + BACKWARD_PASSES_AFTER_SAVE,
        ),
        args_schema=args_schema,
        outputs_schema=BatchTensorDescriptor(1),
        max_batch_size=1,
    )
    experts = {EXPERT_NAME: expert_backend}
    yield experts


@pytest.mark.forked
def test_save_load_checkpoints(example_experts):
    expert = example_experts[EXPERT_NAME].module

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for i in range(1, EXPERT_WEIGHT_UPDATES + 1):
            expert.weight.data[0] = i
            store_experts(example_experts, tmp_path)

        checkpoints_dir = tmp_path / EXPERT_NAME

        assert checkpoints_dir.exists()
        # include checkpoint_last.pt
        assert len(list(checkpoints_dir.iterdir())) == EXPERT_WEIGHT_UPDATES + 1

        expert.weight.data[0] = 0

        load_experts(example_experts, tmp_path)
        assert expert.weight.data[0] == EXPERT_WEIGHT_UPDATES


@pytest.mark.forked
def test_restore_update_count(example_experts):
    expert_backend = example_experts[EXPERT_NAME]

    batch = torch.randn(1, 1)
    loss_grad = torch.randn(1, 1)

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for _ in range(BACKWARD_PASSES_BEFORE_SAVE):
            expert_backend.backward(batch, loss_grad)

        store_experts(example_experts, tmp_path)

        for _ in range(BACKWARD_PASSES_AFTER_SAVE):
            expert_backend.backward(batch, loss_grad)

        load_experts(example_experts, tmp_path)
        assert expert_backend.scheduler._step_count == BACKWARD_PASSES_BEFORE_SAVE + 1


@pytest.mark.forked
def test_lr_schedule(example_experts):
    expert_backend = example_experts[EXPERT_NAME]
    optimizer = expert_backend.optimizer

    batch = torch.randn(1, 1)
    loss_grad = torch.randn(1, 1)

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        assert optimizer.param_groups[0]["lr"] == 0.0

        for i in range(BACKWARD_PASSES_BEFORE_SAVE):
            assert optimizer.param_groups[0]["lr"] == PEAK_LR * i / BACKWARD_PASSES_BEFORE_SAVE
            expert_backend.backward(batch, loss_grad)

        assert optimizer.param_groups[0]["lr"] == PEAK_LR
        store_experts(example_experts, tmp_path)

        for i in range(BACKWARD_PASSES_AFTER_SAVE):
            assert optimizer.param_groups[0]["lr"] == PEAK_LR * (1 - (i / BACKWARD_PASSES_AFTER_SAVE))
            expert_backend.backward(batch, loss_grad)

        assert optimizer.param_groups[0]["lr"] == 0.0
        load_experts(example_experts, tmp_path)
        assert optimizer.param_groups[0]["lr"] == PEAK_LR
