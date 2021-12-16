import hivemind
from hivemind.optim.experimental.grad_averager import GradientAverager
from hivemind.optim.experimental.power_ef_averager import PowerEFGradientAverager

import faulthandler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST

import multiprocessing as mp
import threading
import os
import time


print_step = 10


class Peer(threading.Thread):
    def __init__(self, idx, *, start: bool):
        super().__init__(daemon=True)
        self.dht = hivemind.DHT(initial_peers=dht_root.get_visible_maddrs(), start=True)
        self.model = SmallCNN()
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param).share_memory_()

        self.averager = PowerEFGradientAverager(
            self.model.parameters(), 1, dht=self.dht, target_group_size=4, prefix='my_mega_exp', start=True,
        )
        if start:
            self.start()

        self.idx = idx
        
    def run(self):
        torch.manual_seed(self.idx)
        print('started', self.dht.peer_id)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_data = MNIST(f".", download=True, transform=transform)

        def data():
            while True:
                train_dataloader = torch.utils.data.DataLoader(train_data, num_workers=0, batch_size=1024, shuffle=True)
                for batch in train_dataloader:
                    yield batch
        
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        next_step_time = hivemind.get_dht_time() + 5
        next_step_control = None
        for i, (xb, yb) in enumerate(data()):
            logits = self.model(xb)
            loss = F.cross_entropy(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if next_step_control is None and (next_step_time - hivemind.get_dht_time() <= 1):
                next_step_control = self.averager.schedule_step(scheduled_time=next_step_time)
            
            self.averager.accumulate_grads_(batch_size=1024)

            if hivemind.get_dht_time() >= next_step_time:
                self.averager.step(control=next_step_control)
                next_step_control.result()
                with self.averager.use_averaged_gradients():
                    with torch.no_grad():
                        param = next(iter(self.model.parameters()))
                        grad = param.grad.detach().cpu().norm().item()
                        print_param = param.flatten()[-3:].detach().cpu().numpy()
                        print(i, self.dht.peer_id.pretty()[-3:],f"{loss.item():.3f}", f"{hivemind.get_dht_time():.3f}", print_param, grad)
                    opt.step()
                self.averager.reset_accumulated_grads_()
                next_step_time = hivemind.get_dht_time() + 5
                next_step_control = None
            if i > 10000: break


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 4, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(4, 16, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(16, 64, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.cls = nn.Sequential(
            nn.Linear(64 * 6 * 6, 400),
            nn.ReLU(),
            nn.Linear(400, 10)
        )

    def forward(self, x):
        feature = self.features(x)
        return self.cls(feature.view(x.size(0), -1))


if __name__ == "__main__":
    dht_root = hivemind.DHT(start=True)

    peers = [
        Peer(0, start=False), Peer(1, start=False),
        Peer(2, start=False), Peer(3, start=False)
    ]
    peers[1].model.load_state_dict(peers[0].model.state_dict())
    peers[2].model.load_state_dict(peers[0].model.state_dict())
    peers[3].model.load_state_dict(peers[0].model.state_dict())

    for peer in peers:
        peer.start()
    for p in peers:
        p.join()
