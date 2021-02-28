# Quick Start

This tutorial will teach you how to install `hivemind`, host your own experts and train them remotely.

## Installation

Just `pip install hivemind` to get the latest release.

You can also install the bleeding edge version from GitHub:

```
git clone https://github.com/learning-at-home/hivemind
cd hivemind
pip install .
```

You can also install it in the editable mode with `pip install -e .`.

* __Dependencies:__ Hivemind requires Python 3.7+.
  The [requirements](https://github.com/learning-at-home/hivemind/blob/master/requirements.txt) are installed
  automatically.
* __OS support:__ Linux and macOS should just work. We do not officially support Windows, but you are welcome to
  contribute your windows build :)

## Host a server

`hivemind.Server` hosts one or several experts (PyTorch modules) for remote access. These experts are responsible for
most of the model parameters and computation. The server can be started using either Python or
[a shell script](https://github.com/learning-at-home/hivemind/blob/master/scripts/run_server.py). We'll use the shell
for now. To host a server with default experts, run this in your shell:

```sh
python scripts/run_server.py --expert_cls ffn --hidden_dim 512 --num_experts 5 --expert_pattern expert.[0:5] \
                             --listen_on 0.0.0.0:1337 --dht_port 1338
# note: if you omit listen_on and/or dht_port, they will be chosen automatically and printed to stdout.
```

<details style="margin-top:-24px; margin-bottom: 16px;">
  <summary><i>Console outputs</i></summary>

```sh
[2020/08/26 11:54:52.645][INFO][server.create:101] Bootstrapping DHT node, initial peers = []
[2020/08/26 11:54:52.660][INFO][server.create:105] Running dht node on port 1338
[2020/08/26 11:54:53.182][INFO][server.task_pool.run:130] expert.0_forward starting, pid=19382
[2020/08/26 11:54:53.182][INFO][server.task_pool.run:130] expert.0_forward starting, pid=19382
[2020/08/26 11:54:53.189][INFO][server.task_pool.run:130] expert.0_backward starting, pid=19384
[2020/08/26 11:54:53.189][INFO][server.task_pool.run:130] expert.0_backward starting, pid=19384
[2020/08/26 11:54:53.196][INFO][server.task_pool.run:130] expert.1_forward starting, pid=19386
[2020/08/26 11:54:53.196][INFO][server.task_pool.run:130] expert.1_forward starting, pid=19386
[2020/08/26 11:54:53.206][INFO][server.task_pool.run:130] expert.1_backward starting, pid=19388
[2020/08/26 11:54:53.206][INFO][server.task_pool.run:130] expert.1_backward starting, pid=19388
[2020/08/26 11:54:53.212][INFO][server.task_pool.run:130] expert.2_forward starting, pid=19390
[2020/08/26 11:54:53.212][INFO][server.task_pool.run:130] expert.2_forward starting, pid=19390
[2020/08/26 11:54:53.218][INFO][server.task_pool.run:130] expert.2_backward starting, pid=19392
[2020/08/26 11:54:53.218][INFO][server.task_pool.run:130] expert.2_backward starting, pid=19392
[2020/08/26 11:54:53.225][INFO][server.task_pool.run:130] expert.3_forward starting, pid=19394
[2020/08/26 11:54:53.225][INFO][server.task_pool.run:130] expert.3_forward starting, pid=19394
[2020/08/26 11:54:53.232][INFO][server.task_pool.run:130] expert.3_backward starting, pid=19396
[2020/08/26 11:54:53.232][INFO][server.task_pool.run:130] expert.3_backward starting, pid=19396
[2020/08/26 11:54:53.235][INFO][server.task_pool.run:130] expert.4_forward starting, pid=19398
[2020/08/26 11:54:53.235][INFO][server.task_pool.run:130] expert.4_forward starting, pid=19398
[2020/08/26 11:54:53.241][INFO][server.task_pool.run:130] expert.4_backward starting, pid=19400
[2020/08/26 11:54:53.241][INFO][server.task_pool.run:130] expert.4_backward starting, pid=19400
[2020/08/26 11:54:53.244][INFO][server.runtime.run:60] Started
[2020/08/26 11:54:53.244][INFO][server.runtime.run:60] Started
[2020/08/26 11:54:53.245][INFO][server.create:136] Server started at 0.0.0.0:1337
[2020/08/26 11:54:53.245][INFO][server.create:137] Got 5 active experts of type ffn: ['expert.0', 'expert.1', 'expert.2', 'expert.3', 'expert.4']
```

</details>


This server accepts requests to experts on port 1337 and start a DHT peer on port 1338. In total, it serves 5
feedforward experts with ReLU and LayerNorm
(see
architecture [here](https://github.com/learning-at-home/hivemind/blob/master/hivemind/server/layers/__init__.py#L7-L21))
.

You can create additional servers in the same decentralized network using `--initial_peers` argument:

```sh
python scripts/run_server.py --expert_cls ffn --hidden_dim 512 --num_experts 10 --expert_pattern "expert.[5:250]" \
                              --initial_peers localhost:1338
```

<details style="margin-top:-24px; margin-bottom: 16px;">
  <summary>Console outputs</summary>

```sh
[2020/08/26 13:15:05.078][INFO][server.create:103] Bootstrapping DHT node, initial peers = ['localhost:1338']
[2020/08/26 13:15:05.101][INFO][server.create:107] Running dht node on port 44291
expert.[5:250]
[2020/08/26 13:15:06.326][INFO][server.task_pool.run:130] expert.113_forward starting, pid=29517
[2020/08/26 13:15:06.326][INFO][server.task_pool.run:130] expert.113_forward starting, pid=29517
[2020/08/26 13:15:06.333][INFO][server.task_pool.run:130] expert.113_backward starting, pid=29519
[2020/08/26 13:15:06.333][INFO][server.task_pool.run:130] expert.113_backward starting, pid=29519
[2020/08/26 13:15:06.340][INFO][server.task_pool.run:130] expert.149_forward starting, pid=29521
[2020/08/26 13:15:06.340][INFO][server.task_pool.run:130] expert.149_forward starting, pid=29521
[2020/08/26 13:15:06.352][INFO][server.task_pool.run:130] expert.149_backward starting, pid=29523
[2020/08/26 13:15:06.352][INFO][server.task_pool.run:130] expert.149_backward starting, pid=29523
[2020/08/26 13:15:06.363][INFO][server.task_pool.run:130] expert.185_forward starting, pid=29525
[2020/08/26 13:15:06.363][INFO][server.task_pool.run:130] expert.185_forward starting, pid=29525
[2020/08/26 13:15:06.375][INFO][server.task_pool.run:130] expert.185_backward starting, pid=29527
[2020/08/26 13:15:06.375][INFO][server.task_pool.run:130] expert.185_backward starting, pid=29527
[2020/08/26 13:15:06.381][INFO][server.task_pool.run:130] expert.189_forward starting, pid=29529
[2020/08/26 13:15:06.381][INFO][server.task_pool.run:130] expert.189_forward starting, pid=29529
[2020/08/26 13:15:06.388][INFO][server.task_pool.run:130] expert.189_backward starting, pid=29531
[2020/08/26 13:15:06.388][INFO][server.task_pool.run:130] expert.189_backward starting, pid=29531
[2020/08/26 13:15:06.400][INFO][server.task_pool.run:130] expert.191_forward starting, pid=29533
[2020/08/26 13:15:06.400][INFO][server.task_pool.run:130] expert.191_forward starting, pid=29533
[2020/08/26 13:15:06.407][INFO][server.task_pool.run:130] expert.191_backward starting, pid=29535
[2020/08/26 13:15:06.407][INFO][server.task_pool.run:130] expert.191_backward starting, pid=29535
[2020/08/26 13:15:06.415][INFO][server.task_pool.run:130] expert.196_forward starting, pid=29537
[2020/08/26 13:15:06.415][INFO][server.task_pool.run:130] expert.196_forward starting, pid=29537
[2020/08/26 13:15:06.426][INFO][server.task_pool.run:130] expert.196_backward starting, pid=29539
[2020/08/26 13:15:06.426][INFO][server.task_pool.run:130] expert.196_backward starting, pid=29539
[2020/08/26 13:15:06.435][INFO][server.task_pool.run:130] expert.225_forward starting, pid=29541
[2020/08/26 13:15:06.435][INFO][server.task_pool.run:130] expert.225_forward starting, pid=29541
[2020/08/26 13:15:06.445][INFO][server.task_pool.run:130] expert.225_backward starting, pid=29543
[2020/08/26 13:15:06.445][INFO][server.task_pool.run:130] expert.225_backward starting, pid=29543
[2020/08/26 13:15:06.454][INFO][server.task_pool.run:130] expert.227_forward starting, pid=29545
[2020/08/26 13:15:06.454][INFO][server.task_pool.run:130] expert.227_forward starting, pid=29545
[2020/08/26 13:15:06.467][INFO][server.task_pool.run:130] expert.227_backward starting, pid=29547
[2020/08/26 13:15:06.467][INFO][server.task_pool.run:130] expert.227_backward starting, pid=29547
[2020/08/26 13:15:06.475][INFO][server.task_pool.run:130] expert.36_forward starting, pid=29549
[2020/08/26 13:15:06.475][INFO][server.task_pool.run:130] expert.36_forward starting, pid=29549
[2020/08/26 13:15:06.482][INFO][server.task_pool.run:130] expert.36_backward starting, pid=29551
[2020/08/26 13:15:06.482][INFO][server.task_pool.run:130] expert.36_backward starting, pid=29551
[2020/08/26 13:15:06.497][INFO][server.task_pool.run:130] expert.58_forward starting, pid=29553
[2020/08/26 13:15:06.497][INFO][server.task_pool.run:130] expert.58_forward starting, pid=29553
[2020/08/26 13:15:06.507][INFO][server.task_pool.run:130] expert.58_backward starting, pid=29555
[2020/08/26 13:15:06.507][INFO][server.task_pool.run:130] expert.58_backward starting, pid=29555
[2020/08/26 13:15:06.509][INFO][server.runtime.run:60] Started
[2020/08/26 13:15:06.509][INFO][server.runtime.run:60] Started
[2020/08/26 13:15:06.510][INFO][server.create:166] Server started at 0.0.0.0:40089
[2020/08/26 13:15:06.510][INFO][server.create:167] Got 10 active experts of type ffn: ['expert.113', 'expert.149', 'expert.185', 'expert.189', 'expert.191', 'expert.196', 'expert.225', 'expert.227', 'expert.36', 'expert.58']
```

</details>

Here and below, if you are running on a different machine, replace `localhost:1338` with your original server's public
IP address (e.g. `12.34.56.78:1338`). Hivemind supports both ipv4 and ipv6 protocols and uses the same notation
as [gRPC](https://grpc.io/docs/languages/python/basics/#starting-the-server).

## Train the experts

Now let's put these experts to work. Create a python console (or a jupyter) and run:

```python
import torch
import hivemind

dht = hivemind.DHT(initial_peers=["localhost:1338"], listen=False, start=True)
# note: listen=False means that your peer will operate in "client only" mode: 
# this means that it can request other peers, but will not accept requests in return 

expert1, expert4 = dht.get_experts(["expert.1", "expert.4"])
assert expert1 is not None and expert4 is not None, "server hasn't declared experts (yet?)"
```

The experts (e.g. `expert1`) can be used as a pytorch module with autograd support:

```python
dummy = torch.randn(3, 512)
out = expert1(dummy)  # forward pass
out.sum().backward()  # backward pass
```

When called, expert1 will submit a request to the corresponding server (which you created above) and return the output
tensor(s) or raise an exception. During backward, pytorch will submit the backward requests for the experts as they
appear in the computation graph.

By default, the experts will automatically update their parameters with one step of SGD after each backward pass. This
allows you to quickly run training using both local and remote layers:

```python
# generate dummy data
x = torch.randn(3, 512)
y = 0.01 * x.sum(dim=-1, keepdim=True)

# local torch module
proj_out = torch.nn.Sequential(
    torch.nn.Linear(512, 3)
)
opt = torch.optim.SGD(proj_out.parameters(), lr=0.01)

for i in range(100):
    prediction = proj_out(expert1(expert4(x)))
    loss = torch.mean(abs(prediction - y))
    print(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()
```

Finally, you can create a Mixture-of-Experts layer over these experts:

```python
import nest_asyncio

nest_asyncio.apply()  # asyncio patch for jupyter. for now, we recommend using MoE from console
dmoe = hivemind.RemoteMixtureOfExperts(in_features=512, uid_prefix="expert", grid_size=(5,),
                                       dht=dht, k_best=2)

out = dmoe(torch.randn(3, 512))
out.sum().backward()
```

The `dmoe` layer dynamically selects the right experts using a linear gating function. It will then dispatch parallel
forward (and backward) requests to those experts and collect results. You can find more details on how DMoE works in
Section 2.3 of the [paper](https://arxiv.org/abs/2002.04013)

Congratulations, you've made it through the basic tutorial. Give yourself a pat on the back :)

More advanced tutorials are coming soon :)
