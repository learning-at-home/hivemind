import multiprocessing as mp
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import torch
import torchvision
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

import hivemind
from hivemind.optim.experimental.optimizer import Optimizer
from hivemind.utils.crypto import RSAPrivateKey


@dataclass(frozen=True)
class TrainingArguments:
    seed: int = 42
    prefix: str = "my_exp"

    num_peers: int = 8
    num_clients: int = 3
    target_batch_size: int = 256
    reuse_grad_buffers: bool = True
    delay_grad_averaging: bool = True
    delay_optimizer_step: bool = True
    average_state_every: int = 3
    use_amp: bool = False

    lr_base: float = 0.1
    lr_gamma: int = 0.1
    lr_step_size: int = 10
    max_epoch: int = 25

    batch_size_min: int = 2
    batch_size_max: int = 16
    batch_time_min: float = 1.0
    batch_time_max: float = 4.5
    batch_time_std: float = 0.5

    matchmaking_time: float = 5.0
    max_refresh_period: float = 5.0
    averaging_timeout: float = 15.0
    winddown_time: float = 5.0
    verbose: bool = True

    device: str = "cpu"
    make_dataset: Callable[[], Dataset] = lambda: torchvision.datasets.MNIST(train=True, root=".", download=True)
    make_model: Callable[[int, int], nn.Module] = lambda num_features, num_classes: nn.Sequential(
        nn.Linear(num_features, 64), nn.ReLU(), nn.Linear(64, num_classes)
    )


def benchmark_optimizer(args: TrainingArguments):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    dht = hivemind.DHT(start=True)

    train_dataset = args.make_dataset()
    num_features = np.prod(train_dataset.data[0].shape)
    num_classes = len(train_dataset.classes)
    X_train = torch.as_tensor(train_dataset.data, dtype=torch.float32)
    X_train = X_train.sub_(X_train.mean((0, 1, 2))).div_(X_train.std((0, 1, 2))).reshape((-1, num_features))
    y_train = torch.as_tensor(train_dataset.targets, dtype=torch.int64)
    del train_dataset

    def run_trainer(batch_size: int, batch_time: float, client_mode: bool, verbose: bool):
        model = args.make_model(num_features, num_classes).to(args.device)

        assert isinstance(model, torch.nn.Module), "model_arch must evaluate to a pytorch module"

        optimizer = Optimizer(
            prefix=args.prefix,
            target_batch_size=args.target_batch_size,
            batch_size_per_step=batch_size,
            params=model.parameters(),
            optimizer=partial(torch.optim.SGD, lr=args.lr_base),
            scheduler=partial(torch.optim.lr_scheduler.StepLR, gamma=args.lr_gamma, step_size=args.lr_step_size),
            dht=hivemind.DHT(initial_peers=dht.get_visible_maddrs(), client_mode=client_mode, start=True),
            tracker_opts=dict(private_key=RSAPrivateKey(), max_refresh_period=args.max_refresh_period),
            matchmaking_time=args.matchmaking_time,
            averaging_timeout=args.averaging_timeout,
            reuse_grad_buffers=args.reuse_grad_buffers,
            delay_grad_averaging=args.delay_grad_averaging,
            delay_optimizer_step=args.delay_optimizer_step,
            average_state_every=args.average_state_every,
            client_mode=client_mode,
            verbose=verbose,
        )

        if args.use_amp and args.reuse_grad_buffers:
            grad_scaler = hivemind.GradScaler()
        else:
            grad_scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

        prev_time = time.perf_counter()

        while optimizer.local_epoch < args.max_epoch:
            time.sleep(max(0.0, prev_time + random.gauss(batch_time, args.batch_time_std) - time.perf_counter()))

            batch = torch.randint(0, len(X_train), (batch_size,))

            with torch.cuda.amp.autocast() if args.use_amp else nullcontext():
                loss = F.cross_entropy(model(X_train[batch].to(args.device)), y_train[batch].to(args.device))
                grad_scaler.scale(loss).backward()

            grad_scaler.unscale_(optimizer)

            if args.use_amp:
                grad_scaler.step(optimizer)
            else:
                optimizer.step()

            grad_scaler.update()

            if not args.reuse_grad_buffers:
                optimizer.zero_grad()

            prev_time = time.perf_counter()

        time.sleep(args.winddown_time)
        optimizer.shutdown()

    peers = []

    for index in range(args.num_peers):
        batch_size = random.randint(args.batch_size_min, args.batch_size_max)
        batch_time = random.uniform(args.batch_time_min, args.batch_time_max)
        peers.append(
            mp.Process(
                target=run_trainer,
                name=f"trainer-{index}",
                daemon=False,
                kwargs=dict(
                    batch_size=batch_size,
                    batch_time=batch_time,
                    client_mode=(index >= args.num_peers - args.num_clients),
                    verbose=args.verbose and (index == 0),
                ),
            )
        )

    try:
        for peer in peers[1:]:
            peer.start()
        peers[0].run()
        for peer in peers[1:]:
            peer.join()
    finally:
        for peer in peers[1:]:
            peer.kill()


if __name__ == '__main__':
    benchmark_optimizer(TrainingArguments())