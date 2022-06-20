# Training PPO with decentralized averaging

This tutorial will walk you through the steps to set up collaborative training of an off-policy reinforcement learning algorighm [PPO](https://arxiv.org/pdf/1707.06347.pdf) to play Atari Breakout. It uses [stable-baselines3 implementation of PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), hyperparameters for the algorithm are taken from [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml), collaborative training is built on `hivemind.Optimizer` to exchange information between peers.

## Preparation

* Install hivemind: `pip install git+https://github.com/learning-at-home/hivemind.git`
* Dependencies: `pip install -r requirements.txt`

## Running an experiment

### First peer
Run the first DHT peer to welcome trainers and record training statistics (e.g., loss and performance):
- In this example, we use [tensorboard](https://www.tensorflow.org/tensorboard) to plot training metrics. If you're unfamiliar with Tensorboard, here's a [quickstart tutorial](https://www.tensorflow.org/tensorboard/get_started).
- Run `python3 ppo.py`

```
$ python3 ppo.py
To connect other peers to this one, use --initial_peers /ip4/127.0.0.1/tcp/41926/p2p/QmUmiebP4BxdEPEpQb28cqyhaheDugFRn7M
CoLJr556xYt
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
Using cuda device
Wrapping the env in a VecTransposeImage.
[W NNPACK.cpp:51] Could not initialize NNPACK! Reason: Unsupported hardware.
Jun 20 13:23:20.515 [INFO] Found no active peers: None
Jun 20 13:23:20.533 [INFO] Initializing optimizer manually since it has no tensors in state dict. To override this, prov
ide initialize_optimizer=False
Logging to logs/bs-256.target_bs-32768.n_envs-8.n_steps-128.n_epochs-1_1
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 521      |
|    ep_rew_mean     | 0        |
| time/              |          |
|    fps             | 582      |
|    iterations      | 1        |
|    time_elapsed    | 1        |
|    total_timesteps | 1024     |
| train/             |          |
|    timesteps       | 1024     |
---------------------------------
Jun 20 13:23:23.525 [INFO] ppo_hivemind accumulated 1024 samples for epoch #0 from 1 peers. ETA 52.20 sec (refresh in 1$
.00 sec)

```