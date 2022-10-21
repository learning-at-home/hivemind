# Mixture-of-Experts

This tutorial covers the basics of Decentralized Mixture-of-Experts (DMoE).
From the infrastructure standpoint, DMoE consists of two parts: experts hosted on peer devices, and client-side modules to access those experts.

## Host experts with a server

`hivemind.moe.Server` hosts one or several experts (PyTorch modules) for remote access. These experts are responsible for
most of the model parameters and computation. The server can be started using either Python or
[a shell script](https://github.com/learning-at-home/hivemind/blob/master/hivemind/hivemind_cli/run_server.py). We'll use the shell
for now. To host a server with default experts, run this in your shell:

```sh
hivemind-server --expert_cls ffn --hidden_dim 512 --num_experts 5 --expert_pattern "expert.[0:5]"
# note: server will listen to a random port. To specify interface & port, add --host_maddrs and --announce_maddrs
```

<details style="margin-top:-24px; margin-bottom: 16px;">
  <summary><i>Console outputs</i></summary>

```sh
[2021/07/15 18:52:01.424][INFO][moe.server.create:156] Running DHT node on ['/ip4/127.0.0.1/tcp/42513/p2p/QmacLgRkAHSqdWYdQ8TePioMxQCNV2JeD3AUDmbVd69gNL'], initial peers = []
[2021/07/15 18:52:01.424][INFO][moe.server.create:181] Generating 5 expert uids from pattern expert.[0:5]
[2021/07/15 18:52:01.658][INFO][moe.server.run:233] Server started with 5 experts
[2021/07/15 18:52:01.658][INFO][moe.server.run:237] expert.4: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:52:01.658][INFO][moe.server.run:237] expert.0: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:52:01.659][INFO][moe.server.run:237] expert.3: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:52:01.659][INFO][moe.server.run:237] expert.2: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:52:01.659][INFO][moe.server.run:237] expert.1: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:52:02.447][INFO][moe.server.task_pool.run:145] expert.4_forward starting, pid=14038
[2021/07/15 18:52:02.468][INFO][moe.server.task_pool.run:145] expert.4_backward starting, pid=14042
[2021/07/15 18:52:02.469][INFO][moe.server.task_pool.run:145] expert.0_forward starting, pid=14044
[2021/07/15 18:52:02.484][INFO][moe.server.task_pool.run:145] expert.0_backward starting, pid=14051
[2021/07/15 18:52:02.501][INFO][moe.server.task_pool.run:145] expert.3_forward starting, pid=14057
[2021/07/15 18:52:02.508][INFO][moe.server.task_pool.run:145] expert.3_backward starting, pid=14058
[2021/07/15 18:52:02.508][INFO][moe.server.task_pool.run:145] expert.2_forward starting, pid=14060
[2021/07/15 18:52:02.521][INFO][moe.server.task_pool.run:145] expert.2_backward starting, pid=14070
[2021/07/15 18:52:02.521][INFO][moe.server.task_pool.run:145] expert.1_forward starting, pid=14075
[2021/07/15 18:52:02.532][INFO][moe.server.task_pool.run:145] expert.1_backward starting, pid=14081
[2021/07/15 18:52:02.532][INFO][moe.server.runtime.run:80] Started
```

</details>


This server serves 5 feedforward experts with ReLU and LayerNorm
(see
architecture [here](https://github.com/learning-at-home/hivemind/blob/master/hivemind/moe/server/layers/common.py#L19))
. In order to connect to this server, you should copy its address from console outputs:
```shell
[...][INFO][moe.server.create:156] Running DHT node on ['ADDRESS_WILL_BE_PRINTED_HERE']
```


You can create additional servers in the same decentralized network using the `--initial_peers` argument:

```sh
hivemind-server --expert_cls ffn --hidden_dim 512 --num_experts 10 --expert_pattern "expert.[5:250]" \
                --initial_peers /ip4/127.0.0.1/tcp/42513/p2p/COPY_FULL_ADDRESS_HERE
```

<details style="margin-top:-24px; margin-bottom: 16px;">
  <summary>Console outputs</summary>

```sh
[2021/07/15 18:53:41.700][INFO][moe.server.create:156] Running DHT node on ['/ip4/127.0.0.1/tcp/34487/p2p/QmcJ3jgbdwphLAiwGjvwrjimJJrdMyhLHf6tFj9viCFFGn'], initial peers = ['/ip4/127.0.0.1/tcp/42513/p2p/QmacLgRkAHSqdWYdQ8TePioMxQCNV2JeD3AUDmbVd69gNL']
[2021/07/15 18:53:41.700][INFO][moe.server.create:181] Generating 10 expert uids from pattern expert.[5:250]
[2021/07/15 18:53:42.085][INFO][moe.server.run:233] Server started with 10 experts:
[2021/07/15 18:53:42.086][INFO][moe.server.run:237] expert.55: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:53:42.086][INFO][moe.server.run:237] expert.173: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:53:42.086][INFO][moe.server.run:237] expert.164: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:53:42.086][INFO][moe.server.run:237] expert.99: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:53:42.086][INFO][moe.server.run:237] expert.149: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:53:42.087][INFO][moe.server.run:237] expert.66: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:53:42.087][INFO][moe.server.run:237] expert.106: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:53:42.087][INFO][moe.server.run:237] expert.31: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:53:42.087][INFO][moe.server.run:237] expert.95: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:53:42.087][INFO][moe.server.run:237] expert.167: FeedforwardBlock, 2100736 parameters
[2021/07/15 18:53:43.892][INFO][moe.server.task_pool.run:145] expert.55_forward starting, pid=14854
[2021/07/15 18:53:43.901][INFO][moe.server.task_pool.run:145] expert.55_backward starting, pid=14858
[2021/07/15 18:53:43.915][INFO][moe.server.task_pool.run:145] expert.173_forward starting, pid=14862
[2021/07/15 18:53:43.929][INFO][moe.server.task_pool.run:145] expert.173_backward starting, pid=14864
[2021/07/15 18:53:43.930][INFO][moe.server.task_pool.run:145] expert.164_forward starting, pid=14869
[2021/07/15 18:53:43.948][INFO][moe.server.task_pool.run:145] expert.164_backward starting, pid=14874
[2021/07/15 18:53:43.968][INFO][moe.server.task_pool.run:145] expert.99_forward starting, pid=14883
[2021/07/15 18:53:43.977][INFO][moe.server.task_pool.run:145] expert.99_backward starting, pid=14888
[2021/07/15 18:53:43.995][INFO][moe.server.task_pool.run:145] expert.149_forward starting, pid=14889
[2021/07/15 18:53:44.007][INFO][moe.server.task_pool.run:145] expert.149_backward starting, pid=14898
[2021/07/15 18:53:44.021][INFO][moe.server.task_pool.run:145] expert.66_forward starting, pid=14899
[2021/07/15 18:53:44.034][INFO][moe.server.task_pool.run:145] expert.106_forward starting, pid=14909
[2021/07/15 18:53:44.036][INFO][moe.server.task_pool.run:145] expert.66_backward starting, pid=14904
[2021/07/15 18:53:44.058][INFO][moe.server.task_pool.run:145] expert.106_backward starting, pid=14919
[2021/07/15 18:53:44.077][INFO][moe.server.task_pool.run:145] expert.31_forward starting, pid=14923
[2021/07/15 18:53:44.077][INFO][moe.server.task_pool.run:145] expert.31_backward starting, pid=14925
[2021/07/15 18:53:44.095][INFO][moe.server.task_pool.run:145] expert.95_forward starting, pid=14932
[2021/07/15 18:53:44.106][INFO][moe.server.task_pool.run:145] expert.95_backward starting, pid=14935
[2021/07/15 18:53:44.118][INFO][moe.server.task_pool.run:145] expert.167_forward starting, pid=14943
[2021/07/15 18:53:44.119][INFO][moe.server.task_pool.run:145] expert.167_backward starting, pid=14944
[2021/07/15 18:53:44.123][INFO][moe.server.runtime.run:80] Started
```

</details>

By default, the server will only accept connections from your local network. 
To enable training over the Internet (or some other network), you should set `--host_maddrs` and `--announce_maddrs`.
These options also allow you to select IPv4/IPv6 network protocols and TCP and QUIC transport protocols.
You can find more details in the [DHT tutorial](https://learning-at-home.readthedocs.io/en/latest/user/dht.html).

## Train the experts

Now let's put these experts to work. Create a python console (or a jupyter) and run:

```python
import torch
import hivemind

dht = hivemind.DHT(
    initial_peers=["/ip4/127.0.0.1/tcp/TODO/COPYFULL_ADDRESS/FROM_ONE_OF_THE_SERVERS"],
    client_mode=True, start=True)

# note: client_mode=True means that your peer will operate in a "client-only" mode: 
# this means that it can request other peers, but will not accept requests in return 

expert1, expert4 = hivemind.moe.get_experts(dht, ["expert.1", "expert.4"])
assert expert1 is not None and expert4 is not None, "experts not found. Please double-check initial peers"
```

Each expert (e.g. `expert1`) can be used as a pytorch module with autograd support:

```python
dummy = torch.randn(3, 512)
out = expert1(dummy)  # forward pass
out.sum().backward()  # backward pass
```

When called, `expert1` will submit a request to the corresponding server (which you created above) and return the output
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
import nest_asyncio; nest_asyncio.apply()  # asyncio patch for jupyter. for now, we recommend using MoE from console

dmoe = hivemind.RemoteMixtureOfExperts(in_features=512, uid_prefix="expert.", grid_size=(5,),
                                       dht=dht, k_best=2)

out = dmoe(torch.randn(3, 512))
out.sum().backward()
```

The `dmoe` layer dynamically selects the right experts using a linear gating function. It will then dispatch parallel
forward (and backward) requests to those experts and collect results. You can find more details on how DMoE works in
Section 2.3 of [(Ryabinin et al, 2020)](https://arxiv.org/abs/2002.04013). In addition to traditional MoE, hivemind
implements `hivemind.RemoteSwitchMixtureOfExperts` using the simplified routing algorithm [(Fedus et al 2021)](https://arxiv.org/abs/2101.03961).

For more code examples related to DMoE, such as defining custom experts or using switch-based routing, please refer to
[`hivemind/tests/test_training.py`](https://github.com/learning-at-home/hivemind/blob/master/tests/test_training.py).
