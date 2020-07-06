import threading
import time
from datetime import datetime
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from typing import Dict

import torch

from hivemind.runtime import ExpertBackend


class CheckpointSaver(threading.Thread):
    def __init__(self, expert_backends: Dict[str, ExpertBackend], dir: Path, update_period: int):
        super().__init__()
        self.expert_backends = expert_backends
        self.update_period = update_period
        self.dir = dir

    def run(self) -> None:
        while True:
            store_experts(self.expert_backends, self.dir)
            time.sleep(self.update_period)


def store_experts(experts: Dict[str, ExpertBackend], checkpoints_dir: Path):
    timestamp = datetime.now().isoformat(sep='_')
    with TemporaryDirectory() as tmpdirname:
        for expert_name, expert_backend in experts.items():
            expert_dir = Path(tmpdirname) / expert_name
            expert_dir.mkdir()
            torch.save(expert_backend.state_dict(), expert_dir / f'checkpoint_{timestamp}.pt')
        copytree(tmpdirname, str(checkpoints_dir), dirs_exist_ok=True)


def load_weights(experts: Dict[str, ExpertBackend], checkpoints_dir: Path):
    for expert_name, expert in experts.items():
        checkpoints_folder = checkpoints_dir / expert_name
        latest_checkpoint = max(checkpoints_folder.glob('checkpoint_*.pt'))
        expert.load_state_dict(torch.load(latest_checkpoint))
