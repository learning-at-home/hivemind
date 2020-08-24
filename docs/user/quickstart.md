# Quickstart

This tutorial will teach you how to install `hivemind`, host your own experts and train them remotely.


#### Installation

Just `pip install hivemind` to get the latest release. 

You can also install the bleeding edge version from github:
```
git clone https://github.com/learning-at-home/hivemind
cd hivemind
python setup.py install
```

You can also install it in editable mode with `python setup.py develop`.

* __Dependencies:__ Hivemind requires python 3.7+ (3.8 is recommended), it will install [requirements](https://github.com/learning-at-home/hivemind/blob/master/requirements.txt) automatically; 
* __OS support:__ Linux and Mac OS should [just work](https://github.com/learning-at-home/hivemind/issues).
We do not officially support Windows, but you are welcome to try and contribute your windows build :)


#### Host a server

Hivemind.Server hosts one or several experts (torch modules) for remote access. These experts are responsible for 
most of the model parameters and computation.

To host a server with default experts, run this in your shell:
```sh
python -m TODOPATH.run_server --expert_cls ffn --hidden_dim 512 --num_experts 5 --uid_space TODO \
                              --listen_on 0.0.0.0:1337 --dht_port 1338
# note: if you omit listen_on and/or dht_port, they will be chosen automatically and printed to stdout.
```

This server accepts requests to experts on port 1337 and start a DHT peer on port 1338.
In total, it serves 5 feedforward experts with ReLU and LayerNorm (see architecture [TODOhere][TODO]).

You (and anyone) can create additional servers in the same decentralized network using `--initial_peers` argument:
```sh
python -m TODOPATH.run_server --expert_cls ffn --hidden_dim 512 --num_experts 5 --uid_space TODO \
                              --initial-peers localhost:1338
```

Here and below, if you are running on a different machine, replace `localhost:1338` with your original server's
public IP address (e.g. `12.34.56.78:1338`). Hivemind supports both ipv4 and ipv6 protocols and uses the same notation
as [gRPC](https://grpc.io/docs/languages/python/basics/#starting-the-server).

#### Run the experts

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

When called, expert1 will submit a request to the corresponding server (which you created above) and return
 the outputs tensor(s) or raise an exception. During backward, pytorch will submit the backward requests
 for the experts as they appear in the computation graph.
 
By default, the experts will automatically update their parameters with one step of SGD after each backward pass.
This allows you to quickly run training using a mixture of local and remote layers:
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

Finally, you can create a Mixture-of-Experts layer over our humble band of experts:
```python
import nest_asyncio;  nest_asyncio.apply()  # asyncio patch for jupyter. for now, we recommend using MoE from console
dmoe = hivemind.RemoteMixtureOfExperts(in_features=512, uid_prefix="expert", grid_size=(5,),
                                       dht=dht, k_best=2)

out = dmoe(torch.randn(3, 512))
out.sum().backward()
```

The `dmoe` layer dynamically selects the right experts using a linear gating function. It will then dispatch parallel
forward (and backward) requests to those experts and collect results.
You can find more details on how MoE works in Section 2.3 of the [paper](https://arxiv.org/abs/2002.04013)

Congratulations, you've made it through the basic tutorial. Time to give yourself a pat on the back and decide what's next:
* Run a small training experiment in TODO
* Set up custom experts in TODO
* TODO
