## Hivemind: decentralized deep learning in PyTorch

[![Build status](https://circleci.com/gh/learning-at-home/hivemind.svg?style=shield)](https://circleci.com/gh/learning-at-home/hivemind)
[![Documentation Status](https://readthedocs.org/projects/learning-at-home/badge/?version=latest)](https://learning-at-home.readthedocs.io/en/latest/?badge=latest)
[![Gitter](https://badges.gitter.im/learning-at-home/hivemind.svg)](https://gitter.im/learning-at-home/hivemind?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Hivemind is a PyTorch library to train large neural networks across the Internet. Its intended usage is training a
single Transformer model on hundreds of computers from different universities, companies, and volunteers.

![img](https://i.imgur.com/GPxolxb.gif)

## Key Features

* Train neural networks of arbitrary size: parts of their layers are distributed across the participants.
* Distributed training without a master node: Distributed Hash Table allows connecting computers in a decentralized
  network.
* Fault-tolerant backpropagation: forward and backward passes succeed even if some nodes are unresponsive or take too
  long to respond.
* Decentralized parameter averaging: iteratively aggregate updates from multiple workers without the need to synchronize
  across the entire network.

To learn more about the ideas behind this library, see https://learning-at-home.github.io or read
the [NeurIPS 2020 paper](https://arxiv.org/abs/2002.04013).

## Installation

Before installing hivemind, make sure that your environment has Python 3.7+
and [PyTorch](https://pytorch.org/get-started/locally/#start-locally) with a version at least as new as 1.6.0.

To start using this library, you can either use the pip package manager or build it from source. Since currently the
release cycle is not established yet, we recommend installing hivemind from source to keep up with the latest bugfixes
and improvements.

### With pip

If your versions of Python and PyTorch match the requirements, you can install hivemind from pip:

```
pip install hivemind
```

### From source

To install hivemind from source, simply clone the repository and install

```
git clone https://github.com/learning-at-home/hivemind.git
cd hivemind
pip install .
```

If you would like to verify that your installation is working properly, you can install with `pip install -e .[dev]`
instead. Then, you can run the tests with `pytest tests/`.

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

Also, you can cite [the paper](https://arxiv.org/abs/2002.04013) that inspired the creation of this library:

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

The initial implementation of hivemind used for the paper is available
at [mryab/learning-at-home](https://github.com/mryab/learning-at-home).

In the documentation, we list
several [related](https://learning-at-home.readthedocs.io/en/latest/user/acknowledgements.html) projects and
acknowledgements.

