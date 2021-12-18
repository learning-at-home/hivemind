import hivemind
from hivemind.optim.experimental.grad_averager import GradientAverager
from hivemind.optim.experimental.power_ef_averager import PowerEFGradientAverager
from hivemind.optim.experimental.power_sgd_averager import PowerSGDGradientAverager

import faulthandler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST

import multiprocessing as mp
import threading
import os
import random
import time


print_step = 10


class Peer(threading.Thread):
    def __init__(self, idx, *, start: bool):
        super().__init__(daemon=True)
        self.dht = hivemind.DHT(initial_peers=dht_root.get_visible_maddrs(), start=True)
        self.model = SmallCNN()
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param).share_memory_()

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
                train_dataloader = torch.utils.data.DataLoader(train_data, num_workers=0, batch_size=64, shuffle=True)
                for batch in train_dataloader:
                    yield batch
        
        opt = hivemind.Optimizer(
            dht=self.dht,
            prefix="my_super_run",
            params=self.model.parameters(),
            optimizer=torch.optim.SGD,
            lr=0.1,
            train_batch_size=256,
            batch_size=64
        )
        opt.load_state_from_peers()

        for i, (xb, yb) in enumerate(data()):
            logits = self.model(xb)
            loss = F.cross_entropy(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.averager.accumulate_grads_(batch_size=64)

            opt.step()
            opt.zero_grad()
            if i > 100000: break


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, (9, 9)),
            nn.ReLU(),
            nn.Conv2d(16, 16, (9, 9)),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.cls = nn.Sequential(
            nn.Linear(16 * 6 * 6, 400),
            nn.ReLU(),
            nn.Linear(400, 10)
        )

    def forward(self, x):
        feature = self.features(x)
        return self.cls(feature.view(x.size(0), -1))


if __name__ == "__main__":
    dht_root = hivemind.DHT(start=True)

    peers = [
        Peer(i, start=False) for i in range(4)
    ]
    for i in range(1, 4):
        peers[i].model.load_state_dict(peers[0].model.state_dict())

    for peer in peers:
        peer.start()
    for p in peers:
        p.join()
