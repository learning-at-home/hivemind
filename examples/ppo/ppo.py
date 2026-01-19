import argparse
from importlib.resources import path
import pathlib
import torch

import hivemind
from hivemind import Float16Compression, SizeAdaptiveCompression, Uniform8BitQuantization

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-steps', type=int, default=128, help='Number of rollout steps per each agent')
    parser.add_argument('--n-envs', type=int, default=8, help='Number of training envs')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--target-batch-size', type=int, default=32768)
    parser.add_argument('--n-epochs', type=int, default=1, help='Number of training epochs per each rollout')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--tb-logs-path', type=pathlib.Path, default='./logs', help='Path to tensorboard logs folder')
    parser.add_argument('--experiment-prefix', type=str, help='Experiment prefix for tensorboard logs')
    parser.add_argument('--initial-peers', nargs='+', default=[])
    parser.add_argument('--averaging-compression', action='store_true')
    args = parser.parse_args()
    return args

def generate_experiment_name(args):
    exp_name_dict = {
        'bs': args.batch_size,
        'target_bs': args.target_batch_size,
        'n_envs': args.n_envs,
        'n_steps': args.n_steps,
        'n_epochs': args.n_epochs,
    }

    exp_name = [f'{key}-{value}' for key, value in exp_name_dict.items()]
    exp_name = '.'.join(exp_name)

    if args.experiment_prefix:
        exp_name = f'{args.experiment_prefix}.{exp_name}'
    exp_name = exp_name.replace('000.', 'k.')
    return exp_name


class AdamWithClipping(torch.optim.Adam):
    def __init__(self, *args, max_grad_norm: float, **kwargs):
        self.max_grad_norm = max_grad_norm
        super().__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        iter_params = (param for group in self.param_groups for param in group["params"])
        torch.nn.utils.clip_grad_norm_(iter_params, self.max_grad_norm)
        return super().step(*args, **kwargs)


def configure_dht_opts(args):
    opts = {
        'start': True,
    }
    if args.initial_peers:
        opts['initial_peers'] = args.initial_peers

    return opts


if __name__ == "__main__":
    args = parse_args()

    dht_opts = configure_dht_opts(args)
    dht = hivemind.DHT(**dht_opts)
    print("To connect other peers to this one, use --initial_peers", *[str(addr) for addr in dht.get_visible_maddrs()])

    env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=args.n_envs)
    env = VecFrameStack(env, n_stack=4)

    model = PPO(
        'CnnPolicy', env,
        verbose=1,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.01,
        tensorboard_log=args.tb_logs_path,
        max_grad_norm=10000.0,
        policy_kwargs={'optimizer_class': AdamWithClipping, 'optimizer_kwargs': {'max_grad_norm': 0.5}}
    )

    compression_opts = {}
    if args.averaging_compression:
        averaging_compression = SizeAdaptiveCompression(
            threshold=2 ** 10 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization()
        )
        compression_opts.update({
            'grad_compression': averaging_compression,
            'state_averaging_compression': averaging_compression
        })

    model.policy.optimizer_class = hivemind.Optimizer
    model.policy.optimizer = hivemind.Optimizer(
        dht=dht,
        optimizer=model.policy.optimizer,
        run_id='ppo_hivemind',
        batch_size_per_step=args.batch_size,
        target_batch_size=args.target_batch_size,
        offload_optimizer=False,
        verbose=True,
        use_local_updates=False,
        matchmaking_time=4,
        averaging_timeout=15,
        **compression_opts,
    )
    model.policy.optimizer.load_state_from_peers()
    model.learn(total_timesteps=int(5e11), tb_log_name=generate_experiment_name(args))
