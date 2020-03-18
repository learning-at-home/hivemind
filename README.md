## Tesseract

[![<ORG_NAME>](https://circleci.com/gh/learning-at-home/tesseract.svg?style=svg)](https://circleci.com/gh/learning-at-home/tesseract)

Distributed training of large neural networks across volunteer computers.

![img](https://i.imgur.com/GPxolxb.gif)

**[WIP]** - this branch is a work in progress. If you're interested in
supplementary code for [Learning@home paper](https://arxiv.org/abs/2002.04013),
you can find it at https://github.com/mryab/learning-at-home .

## What do I need to run it?

- One or several computers, each equipped with at least one GPU
- Each computer should have at least two open ports (if not, consider ssh port
  forwarding)
- Some popular Linux x64 distribution
  - Tested on Ubuntu16.04, should work fine on any popular linux64 and even
    MacOS;
  - Running on Windows natively is not supported, please use vm or docker;

## How do I run it?

Currently, there is no way to do it easily. There are some tests (you can check [`./tests/benchmark_throughput.py`](./tests/benchmark_throughput.py)
 or look into CI logs) and we want to expand them. If you want to
do something complex with it, please contact us by opening an issue (less preferred: [telegram](https://t.me/justheuristic)).

## `tesseract` quick tour

**Trainer process:**

- **`RemoteExpert`**(`tesseract/client/remote_expert.py`) behaves like a pytorch
  module with autograd support but actually sends request to a remote runtime.
- **`GatingFunction`**(`tesseract/client/gating_function.py`) finds best experts
  for a given input and either returns them as `RemoteExpert` or applies them
  right away.

**Runtime process:**

- **`TesseractRuntime`** (`tesseract/runtime/__init__.py`) aggregates batches
  and performs inference/training of experts according to their priority.
- **`TesseractServer`** (`tesseract/server/__init__.py`) wraps runtime and
  periodically uploads experts into `TesseractNetwork`.

**DHT:**

- **`TesseractNetwork`**(`tesseract/network/__init__.py`) is a node of
  Kademlia-based DHT that stores metadata used by trainer and runtime.

## Limitations

**DHT**:

- DHT functionality is severely limited by its inability to traverse NAT.
- Because of this all the features that require DHT are in deep pre-alpha state
  and cannot be used without special setup.

**Runtime**:
* You can achieve 4x less network load by passing quantized uint8 activations across experts.
    Implement your own quantization or wait for tesseract v0.8.
* Currently runtime can form batches that exceed maximal batch_size by task_size - 1. 
    We will fix that in the nearest patch.