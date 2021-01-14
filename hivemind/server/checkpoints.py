import threading
from datetime import datetime
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from typing import Dict
import os

import torch

from hivemind.server.expert_backend import ExpertBackend


def dir_is_correct(directory: Path):
    assert directory is not None
    assert directory.exists()
    assert directory.is_dir()
    return True


class CheckpointSaver(threading.Thread):
    def __init__(self, expert_backends: Dict[str, ExpertBackend], checkpoint_dir: Path, update_period: int):
        super().__init__()
        assert dir_is_correct(checkpoint_dir)
        self.expert_backends = expert_backends
        self.update_period = update_period
        self.checkpoint_dir = checkpoint_dir
        self.stop = threading.Event()

        # create expert directories to ensure that the directory is writable and checkpoints can be loaded
        store_experts(self.expert_backends, self.checkpoint_dir)

    def run(self) -> None:
        while not self.stop.wait(self.update_period):
            store_experts(self.expert_backends, self.checkpoint_dir)


def store_experts(experts: Dict[str, ExpertBackend], checkpoint_dir: Path):
    assert dir_is_correct(checkpoint_dir)
    timestamp = datetime.now().isoformat(sep='_')
    with TemporaryDirectory() as tmpdirname:
        for expert_name, expert_backend in experts.items():
            expert_dir = Path(tmpdirname) / expert_name
            expert_dir.mkdir()
            checkpoint_name = expert_dir / f'checkpoint_{timestamp}.pt'
            torch.save(expert_backend.state_dict(), checkpoint_name)
            os.symlink(checkpoint_name, expert_dir / 'checkpoint_last.pt')
        copytree(tmpdirname, str(checkpoint_dir), dirs_exist_ok=True)


def load_weights(experts: Dict[str, ExpertBackend], checkpoint_dir: Path):
    assert dir_is_correct(checkpoint_dir)
    for expert_name, expert in experts.items():
        checkpoints_folder = checkpoint_dir / expert_name
        latest_checkpoint = checkpoints_folder / 'checkpoint_last.pt'
        expert.load_state_dict(torch.load(latest_checkpoint))
