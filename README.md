## Hivemind: decentralized deep learning in PyTorch

[![Documentation Status](https://readthedocs.org/projects/learning-at-home/badge/?version=latest)](https://learning-at-home.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/hivemind.svg)](https://pypi.org/project/hivemind/)
[![Discord](https://img.shields.io/static/v1?style=default&label=chat&logo=discord&message=join%20chat)](https://discord.gg/xC7ucM8j)
[![CI status](https://github.com/learning-at-home/hivemind/actions/workflows/run-tests.yml/badge.svg?branch=master)](https://github.com/learning-at-home/hivemind/actions)
![Codecov](https://img.shields.io/codecov/c/github/learning-at-home/hivemind)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Hivemind is a PyTorch library for decentralized deep learning across the Internet. Its intended usage is training one
large model on hundreds of computers from different universities, companies, and volunteers.

![img](https://i.imgur.com/GPxolxb.gif)

## Key Features

* Distributed training without a master node: Distributed Hash Table allows connecting computers in a decentralized
  network.
* Fault-tolerant backpropagation: forward and backward passes succeed even if some nodes are unresponsive or take too
  long to respond.
* [Decentralized parameter averaging](https://arxiv.org/abs/2103.03239): iteratively aggregate updates from multiple
  workers without the need to synchronize across the entire network.
* Train neural networks of arbitrary size: parts of their layers are distributed across the participants with
  [Decentralized Mixture-of-Experts](https://arxiv.org/abs/2002.04013).

To learn more about the ideas behind this library, see https://learning-at-home.github.io or read
the [NeurIPS 2020 paper](https://arxiv.org/abs/2002.04013).

## Installation

Before installing hivemind, make sure that your environment has Python 3.7+
and [PyTorch](https://pytorch.org/get-started/locally/#start-locally) with a version at least as new as 1.6.0.

To start using this library, you can either install [the latest release](https://pypi.org/project/hivemind/) with pip
or build it from source.

Note: for now, hivemind can be run on Windows only using
[WSL](https://docs.microsoft.com/ru-ru/windows/wsl/install-win10). If you want to configure WSL to work with the GPU,
use the [official guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) by NVIDIA.

### With pip

If your versions of Python and PyTorch match the requirements, you can install hivemind from pip:

```
pip install hivemind
```

### From source

To install hivemind from source, simply run the following:

```
git clone https://github.com/learning-at-home/hivemind.git
cd hivemind
pip install .
```

If you would like to verify that your installation is working properly, you can install with `pip install -e .[dev]`
instead. Then, you can run the tests with `pytest tests/`.

By default, hivemind uses the precompiled binary of
the [go-libp2p-daemon](https://github.com/learning-at-home/go-libp2p-daemon) library. If you face compatibility issues
or want to build the binary yourself, you can recompile it by running `pip install . --global-option="--buildgo"`.
Before running the compilation, please ensure that your machine has a recent version
of [Go toolchain](https://golang.org/doc/install) (1.15 or higher).

## Documentation

* [Quickstart](https://learning-at-home.readthedocs.io/en/latest/user/quickstart.html): install hivemind, set up a
  server and train experts
* Documentation & guides are available at [learning-at-home.readthedocs.io](https://learning-at-home.readthedocs.io)

## Contributing

Hivemind is currently at the active development stage, and we welcome all contributions. Everything, from bug fixes and
documentation improvements to entirely new features, is equally appreciated.

If you want to contribute to hivemind but don't know where to start, take a look at the
unresolved [issues](https://github.com/learning-at-home/hivemind/issues). Open a new issue or
join [our chat room](https://gitter.im/learning-at-home/hivemind) in case you want to discuss new functionality or
report a possible bug. Bug fixes are always welcome, but new features should be preferably discussed with maintainers
beforehand.

If you want to start contributing to the source code of hivemind, please see
the [contributing guidelines](https://github.com/learning-at-home/hivemind/blob/master/CONTRIBUTING.md) first. To learn
more about other ways to contribute, read
our [guide](https://learning-at-home.readthedocs.io/en/latest/user/contributing.html).

## Citation

If you found hivemind or its underlying algorithms useful for your experiments, please cite the following source:

```
@misc{hivemind,
  author = {Learning@home team},
  title = {{H}ivemind: a {L}ibrary for {D}ecentralized {D}eep {L}earning},
  year = 2020,
  howpublished = {\url{https://github.com/learning-at-home/hivemind}},
}
```

Also, you can cite [the paper](https://arxiv.org/abs/2002.04013) that inspired the creation of this library
(prototype implementation of hivemind available at [mryab/learning-at-home](https://github.com/mryab/learning-at-home)):

```
@inproceedings{ryabinin2020crowdsourced,
 author = {Ryabinin, Max and Gusev, Anton},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {3659--3672},
 publisher = {Curran Associates, Inc.},
 title = {Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts},
 url = {https://proceedings.neurips.cc/paper/2020/file/25ddc0f8c9d3e22e03d3076f98d83cb2-Paper.pdf},
 volume = {33},
 year = {2020}
}
```

<details>
 <summary>Additional publications</summary>

["Moshpit SGD: Communication-Efficient Decentralized Training on Heterogeneous Unreliable Devices"](https://arxiv.org/abs/2103.03239)

```
@misc{ryabinin2021moshpit,
      title={Moshpit SGD: Communication-Efficient Decentralized Training on Heterogeneous Unreliable Devices}, 
      author={Max Ryabinin and Eduard Gorbunov and Vsevolod Plokhotnyuk and Gennady Pekhimenko},
      year={2021},
      eprint={2103.03239},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

["Distributed Deep Learning in Open Collaborations"](https://arxiv.org/abs/2106.10207)

```
@misc{diskin2021distributed,
      title={Distributed Deep Learning in Open Collaborations}, 
      author={Michael Diskin and Alexey Bukhtiyarov and Max Ryabinin and Lucile Saulnier and Quentin Lhoest and Anton Sinitsin and Dmitry Popov and Dmitry Pyrkin and Maxim Kashirin and Alexander Borzunov and Albert Villanova del Moral and Denis Mazur and Ilia Kobelev and Yacine Jernite and Thomas Wolf and Gennady Pekhimenko},
      year={2021},
      eprint={2106.10207},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

["Secure Distributed Training at Scale"](https://arxiv.org/abs/2106.11257)

```
@misc{gorbunov2021secure,
      title={Secure Distributed Training at Scale}, 
      author={Eduard Gorbunov and Alexander Borzunov and Michael Diskin and Max Ryabinin},
      year={2021},
      eprint={2106.11257},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

</details>

In the documentation, we list
several [related](https://learning-at-home.readthedocs.io/en/latest/user/acknowledgements.html) projects and
acknowledgements.

