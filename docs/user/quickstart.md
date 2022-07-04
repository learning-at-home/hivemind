# Quick Start

This tutorial will teach you how to install `hivemind`, host your own experts and train them remotely.

## Installation

Just `pip install hivemind` to get the latest release (requires Python 3.7 or newer). 

You can also install the bleeding edge version from GitHub:

```
git clone https://github.com/learning-at-home/hivemind
cd hivemind
pip install -e . --no-use-pep517
```
 
## Decentralized Training

Hivemind is a set of building blocks for decentralized training.
In this tutorial, we'll use two of these blocks to train a simple neural network to classify CIFAR-10 images.
We assume that you are already familiar with the official [CIFAR-10 example](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
from the PyTorch website.

We build on top of the official example to spin up distributed training of a two-layer neural network by averaging weights.
For simplicity, this tutorial will use two non-GPU peers running on the same machine. If you try to run this example on two
separate machines with different IPs, this example will not work. To read more about how to perform training on more
than one machine check out [DHT - Running Across the Internet](https://learning-at-home.readthedocs.io/en/latest/user/dht.html#running-across-the-internet).
If you get to the end of this tutorial, we'll give you an example of actual distributed training of Transformers ;)

For now, let's run our first training peer:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import hivemind

# Create dataset and model, same as in the basic tutorial
# For this basic tutorial, we download only the training set
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

model = nn.Sequential(nn.Conv2d(3, 16, (5, 5)), nn.MaxPool2d(2, 2), nn.ReLU(),
                      nn.Conv2d(16, 32, (5, 5)), nn.MaxPool2d(2, 2), nn.ReLU(),
                      nn.Flatten(), nn.Linear(32 * 5 * 5, 10))
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Create DHT: a decentralized key-value storage shared between peers
dht = hivemind.DHT(start=True)
print("To join the training, use initial_peers =", [str(addr) for addr in dht.get_visible_maddrs()])

# Set up a decentralized optimizer that will average with peers in background
opt = hivemind.Optimizer(
    dht=dht,                  # use a DHT that is connected with other peers
    run_id='my_cifar_run',    # unique identifier of this collaborative run
    batch_size_per_step=32,   # each call to opt.step adds this many samples towards the next epoch
    target_batch_size=10000,  # after peers collectively process this many samples, average weights and begin the next epoch 
    optimizer=opt,            # wrap the SGD optimizer defined above
    use_local_updates=True,   # perform optimizer steps with local gradients, average parameters in background
    matchmaking_time=3.0,     # when averaging parameters, gather peers in background for up to this many seconds
    averaging_timeout=10.0,   # give up on averaging if not successful in this many seconds
    verbose=True              # print logs incessently
)

# Note: if you intend to use GPU, switch to it only after the decentralized optimizer is created
with tqdm() as progressbar:
    while True:
        for x_batch, y_batch in torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=32):
            opt.zero_grad()
            loss = F.cross_entropy(model(x_batch), y_batch)
            loss.backward()
            opt.step()

            progressbar.desc = f"loss = {loss.item():.3f}"
            progressbar.update()
```


As you can see, this code is regular PyTorch with one notable exception: it wraps your regular optimizer with a
`hivemind.Optimizer`. This optimizer uses `DHT` to find other peers and tries to exchange parameters them. When you run
the code (please do so), you will see the following output:

```shell
To join the training, use initial_peers = ['/ip4/127.0.0.1/tcp/XXX/p2p/YYY']
[...] Starting a new averaging round with current parameters.
```

This is `hivemind.Optimizer` telling you that it's looking for peers. Since there are no peers, we'll need to create 
them ourselves.

Copy the entire script (or notebook) and modify this line:

```python
# old version:
dht = hivemind.DHT(start=True)

# new version: added initial_peers
dht = hivemind.DHT(initial_peers=['/ip4/127.0.0.1/tcp/COPY_FULL_ADDRESS_FROM_PEER1_OUTPUTS'], start=True)
```
<details style="margin-top:-16px; margin-bottom: 16px;">
  <summary>Here's the full code of the second peer</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import hivemind

# Create dataset and model, same as in the basic tutorial
# For this basic tutorial, we download only the training set
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

model = nn.Sequential(nn.Conv2d(3, 16, (5, 5)), nn.MaxPool2d(2, 2), nn.ReLU(),
                      nn.Conv2d(16, 32, (5, 5)), nn.MaxPool2d(2, 2), nn.ReLU(),
                      nn.Flatten(), nn.Linear(32 * 5 * 5, 10))
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Create DHT: a decentralized key-value storage shared between peers
dht = hivemind.DHT(initial_peers=[COPY_FROM_OTHER_PEERS_OUTPUTS], start=True)
print("To join the training, use initial_peers =", [str(addr) for addr in dht.get_visible_maddrs()])

# Set up a decentralized optimizer that will average with peers in background
opt = hivemind.Optimizer(
    dht=dht,                  # use a DHT that is connected with other peers
    run_id='my_cifar_run',    # unique identifier of this collaborative run
    batch_size_per_step=32,   # each call to opt.step adds this many samples towards the next epoch
    target_batch_size=10000,  # after peers collectively process this many samples, average weights and begin the next epoch
    optimizer=opt,            # wrap the SGD optimizer defined above
    use_local_updates=True,   # perform optimizer steps with local gradients, average parameters in background
    matchmaking_time=3.0,     # when averaging parameters, gather peers in background for up to this many seconds
    averaging_timeout=10.0,   # give up on averaging if not successful in this many seconds
    verbose=True              # print logs incessently
)

opt.load_state_from_peers()

# Note: if you intend to use GPU, switch to it only after the optimizer is created
with tqdm() as progressbar:
    while True:
        for x_batch, y_batch in torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=32):
            opt.zero_grad()
            loss = F.cross_entropy(model(x_batch), y_batch)
            loss.backward()
            opt.step()

            progressbar.desc = f"loss = {loss.item():.3f}"
            progressbar.update()
```
</details>


Instead of setting up a new DHT, the second peer will link up with the existing DHT node from the first peer.
If you run the second peer, you will see that both first and second peer will periodically report averaging parameters:

```shell
[...] Starting a new averaging round with current parameters.
[...] Finished averaging round in with 2 peers.
```

This message means that the optimizer has averaged model parameters with another peer in background and applied them
during one of the calls to `opt.step()`. You can start more peers by replicating the same code as the second peer,
using either the first or second peer as `initial_peers`.

Each new peer starts with an untrained network and must download the latest training state before it can contribute.
By default, peer will automatically detect that it is out of sync and start ``Downloading parameters from peer <...>``.
To avoid wasting the first optimizer step, one can manually download the latest model/optimizer state right before it begins training on minibatches:
```python
opt.load_state_from_peers()
```

Congrats, you've just started a pocket-sized experiment with decentralized deep learning!

However, this is only the basics of what hivemind can do. In [this example](https://github.com/learning-at-home/hivemind/tree/master/examples/albert),
we show how to use a more advanced version of DecentralizedOptimizer to collaboratively train a large Transformer over the internet.

If you want to learn more about each individual component,
- Learn how to use `hivemind.DHT` using this basic [DHT tutorial](https://learning-at-home.readthedocs.io/en/latest/user/dht.html),
- Read more on how to use `hivemind.Optimizer` in its [documentation page](https://learning-at-home.readthedocs.io/en/latest/modules/optim.html), 
- Learn the underlying math behind hivemind.Optimizer in [Diskin et al., (2021)](https://arxiv.org/abs/2106.10207), 
  [Li et al. (2020)](https://arxiv.org/abs/2005.00124) and [Ryabinin et al. (2021)](https://arxiv.org/abs/2103.03239).
- Read about setting up Mixture-of-Experts training in [this guide](https://learning-at-home.readthedocs.io/en/latest/user/moe.html),
