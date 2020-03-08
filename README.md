## Tesseract

[![<ORG_NAME>](https://circleci.com/gh/learning-at-home/tesseract.svg?style=svg)](https://circleci.com/gh/learning-at-home/tesseract)


Distributed training of large neural networks across volunteer computers.

![img](https://i.imgur.com/GPxolxb.gif)

__[WIP]__ - this branch is in progress of updating. If you're interested in supplementary code for [Learning@home paper](https://arxiv.org/abs/2002.04013), you can find it at https://github.com/mryab/learning-at-home .


## What do I need to run it?
* One or several computers, each equipped with at least one GPU
* Each computer should have at least two open ports (if not, consider ssh port forwarding)
* Some popular Linux x64 distribution
  * Tested on Ubuntu16.04, should work fine on any popular linux64 and even MacOS;
  * Running on Windows natively is not supported, please use vm or docker;

## How do I run it?
1. Clone or download this repo. `cd` to its root directory.
2. Grab or build a working python enviromnent. [Anaconda](https://www.anaconda.com/) works fine.
3. Install packages from `requirements.txt`
4. Go to [./experiments](./experiments) and follow the README.md from there


## tesseract quick tour

__Trainer process:__
  * __`RemoteExpert`__(`lib/client/remote_expert.py`) behaves like a pytorch module with autograd support but actually sends request to a remote runtime.
  * __`GatingFunction`__(`lib/client/gating_function.py`) finds best experts for a given input and either returns them as `RemoteExpert` or applies them right away.

__Runtime process:__
  * __`TesseractRuntime`__ (`lib/runtime/__init__.py`) aggregates batches and performs inference/training of experts according to their priority. 
  * __`TesseractServer`__ (`lib/server/__init__.py`) wraps runtime and periodically uploads experts into `TesseractNetwork`.

__DHT:__
   * __`TesseractNetwork`__(`lib/network/__init__.py`) is a node of Kademlia-based DHT that stores metadata used by trainer and runtime.

## Limitations
WIP
