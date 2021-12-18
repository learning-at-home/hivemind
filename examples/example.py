import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import hivemind


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
    # Create dataset and model, same as in the basic tutorial
    # For this basic tutorial, we download only the training set
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create DHT: a decentralized key-value storage shared between peers
    dht = hivemind.DHT(start=True, initial_peers=["/ip4/127.0.0.1/tcp/36805/p2p/Qmc7nJt6Pc3Eii4X1ZqtkxbiRWvf97nNfuD4CJpAep5THU"])
    print("To join the training, use initial_peers =", [str(addr) for addr in dht.get_visible_maddrs()])

    # Set up a decentralized optimizer that will average with peers in background
    opt = hivemind.Optimizer(
        dht=dht,                  # use a DHT that is connected with other peers
        run_id='my_cifar_run',    # unique identifier of this collaborative run
        batch_size_per_step=16,   # each call to opt.step adds this many samples towards the next epoch
        target_batch_size=1000,  # after peers collectively process this many samples, average weights and begin the next epoch 
        optimizer=opt,            # wrap the SGD optimizer defined above
        use_local_updates=False,  # perform optimizer steps with averaged gradients
        matchmaking_time=3.0,     # when averaging parameters, gather peers in background for up to this many seconds
        averaging_timeout=10.0,   # give up on averaging if not successful in this many seconds
        verbose=True,             # print logs incessently
        grad_rank_averager="power_sgd",
        grad_averager_opts={"averager_rank": 1}
    )
    opt.load_state_from_peers()

    # Note: if you intend to use GPU, switch to it only after the decentralized optimizer is created
    with tqdm() as progressbar:
        while True:
            for x_batch, y_batch in torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=16):
                time.sleep(0.1)
                opt.zero_grad()
                loss = F.cross_entropy(model(x_batch), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                progressbar.desc = f"loss = {loss.item():.3f}"
                progressbar.update()
