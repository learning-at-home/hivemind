## Hivemind: decentralized deep learning in PyTorch

[![Documentation Status](https://readthedocs.org/projects/learning-at-home/badge/?version=latest)](https://learning-at-home.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/hivemind.svg?color=blue)](https://pypi.org/project/hivemind/)
[![Discord](https://img.shields.io/static/v1?style=default&label=Discord&logo=discord&message=join)](https://discord.gg/uGugx9zYvN)
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
* Decentralized parameter averaging: iteratively aggregate updates from multiple workers without the need to
  synchronize across the entire network ([paper](https://arxiv.org/abs/2103.03239)).
* Train neural networks of arbitrary size: parts of their layers are distributed across the participants with the
  Decentralized Mixture-of-Experts ([paper](https://arxiv.org/abs/2002.04013)).

To learn more about the ideas behind this library,
see the [full list](https://github.com/learning-at-home/hivemind/tree/refer-to-discord-in-docs#citation) of our papers below.

## Hivemind in Action: Applications and Use Cases

This section lists external projects that leverage hivemind for decentralized training. 
If you have succesfully trained a model or created a downstream repository with the help of our library, 
feel free to submit a pull request that adds your project to this list.

* [sahajBERT](https://github.com/tanmoyio/sahajbert) — a collaboratively pretrained ALBERT-xlarge for the Bengali language
* [CALM](https://github.com/NCAI-Research/CALM) (Collaborative Arabic Language Model) — a masked language model trained on a combination of Arabic datasets
* [Training Transformers Together](https://training-transformers-together.github.io/) — a NeurIPS 2021 demonstration that trained a collaborative text-to-image Transformer model
* [HivemindStrategy](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.strategies.HivemindStrategy.html) in PyTorch Lightning allows adapting your existing pipelines to training over slow network with unreliable peers

## Installation

Before installing, make sure that your environment has Python 3.7+
and [PyTorch](https://pytorch.org/get-started/locally/#start-locally) 1.6.0 or newer. They can be installed either
natively or with [Anaconda](https://www.anaconda.com/products/individual).

You can get [the latest release](https://pypi.org/project/hivemind) with pip or build hivemind from source.

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

If you would like to verify that your installation is working properly, you can install with `pip install .[dev]`
instead. Then, you can run the tests with `pytest tests/`.

By default, hivemind uses the precompiled binary of
the [go-libp2p-daemon](https://github.com/learning-at-home/go-libp2p-daemon) library. If you face compatibility issues
or want to build the binary yourself, you can recompile it by running `pip install . --global-option="--buildgo"`.
Before running the compilation, please ensure that your machine has a recent version
of [Go toolchain](https://golang.org/doc/install) (1.15 or 1.16 are supported).

### System requirements

- __Linux__ is the default OS for which hivemind is developed and tested. We recommend Ubuntu 18.04+ (64-bit), but
  other 64-bit distros should work as well. Legacy 32-bit is not recommended.
- __macOS 10.x__ can run hivemind using [Docker](https://docs.docker.com/desktop/mac/install/).
  We recommend using [our Docker image](https://hub.docker.com/r/learningathome/hivemind).
- __Windows 10+ (experimental)__ can run hivemind
  using [WSL](https://docs.microsoft.com/ru-ru/windows/wsl/install-win10). You can configure WSL to use GPU by
  following sections 1–3 of [this guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) by NVIDIA. After
  that, you can simply follow the instructions above to install with pip or from source.

## Documentation

* The [quickstart tutorial](https://learning-at-home.readthedocs.io/en/latest/user/quickstart.html) walks through
  installation and a training a simple neural network with several peers.
* [examples/albert](https://github.com/learning-at-home/hivemind/tree/master/examples/albert) contains the starter kit
  and instructions for training a Transformer masked language model collaboratively.
* The [Mixture-of-Experts tutorial](https://learning-at-home.readthedocs.io/en/latest/user/moe.html)
  covers the usage of Decentralized Mixture-of-Experts layers.
* API reference and additional tutorials are available
  at [learning-at-home.readthedocs.io](https://learning-at-home.readthedocs.io)

If you have any questions about installing and using hivemind, feel free to ask them in
[our Discord chat](https://discord.gg/uGugx9zYvN) or file an [issue](https://github.com/learning-at-home/hivemind/issues).

## Contributing

Hivemind is currently at the active development stage, and we welcome all contributions. Everything, from bug fixes and
documentation improvements to entirely new features, is appreciated.

If you want to contribute to hivemind but don't know where to start, take a look at the
unresolved [issues](https://github.com/learning-at-home/hivemind/issues). Open a new issue or
join [our chat room](https://discord.gg/uGugx9zYvN) in case you want to discuss new functionality or report a possible
bug. Bug fixes are always welcome, but new features should be preferably discussed with maintainers beforehand.

If you want to start contributing to the source code of hivemind, please see
the [contributing guidelines](https://github.com/learning-at-home/hivemind/blob/master/CONTRIBUTING.md) first. To learn
more about other ways to contribute, read
our [guide](https://learning-at-home.readthedocs.io/en/latest/user/contributing.html).

## Citation

If you found hivemind or its underlying algorithms useful for your research, please cite the following source:

```bibtex
@misc{hivemind,
  author = {Learning{@}home team},
  title = {{H}ivemind: a {L}ibrary for {D}ecentralized {D}eep {L}earning},
  year = 2020,
  howpublished = {\url{https://github.com/learning-at-home/hivemind}},
}
```

Also, you can cite [the paper](https://arxiv.org/abs/2002.04013) that inspired the creation of this library
(prototype implementation of hivemind available
at [mryab/learning-at-home](https://github.com/mryab/learning-at-home)):

```bibtex
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

```bibtex
@inproceedings{ryabinin2021moshpit,
 author = {Ryabinin, Max and Gorbunov, Eduard and Plokhotnyuk, Vsevolod and Pekhimenko, Gennady},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {18195--18211},
 publisher = {Curran Associates, Inc.},
 title = {Moshpit SGD: Communication-Efficient Decentralized Training on Heterogeneous Unreliable Devices},
 url = {https://proceedings.neurips.cc/paper/2021/file/97275a23ca44226c9964043c8462be96-Paper.pdf},
 volume = {34},
 year = {2021}
}
```

["Distributed Deep Learning in Open Collaborations"](https://arxiv.org/abs/2106.10207)

```bibtex
@inproceedings{diskin2021distributed,
    title = {Distributed Deep Learning In Open Collaborations},
    author = {Michael Diskin and Alexey Bukhtiyarov and Max Ryabinin and Lucile Saulnier and Quentin Lhoest and Anton Sinitsin and Dmitry Popov and Dmitriy Pyrkin and Maxim Kashirin and Alexander Borzunov and Albert Villanova del Moral and Denis Mazur and Ilia Kobelev and Yacine Jernite and Thomas Wolf and Gennady Pekhimenko},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
    year = {2021},
    url = {https://openreview.net/forum?id=FYHktcK-7v}
}
```

["Secure Distributed Training at Scale"](https://arxiv.org/abs/2106.11257)

```bibtex
@InProceedings{pmlr-v162-gorbunov22a,
  title = 	 {Secure Distributed Training at Scale},
  author =       {Gorbunov, Eduard and Borzunov, Alexander and Diskin, Michael and Ryabinin, Max},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {7679--7739},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/gorbunov22a/gorbunov22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/gorbunov22a.html},
  abstract = 	 {Many areas of deep learning benefit from using increasingly larger neural networks trained on public data, as is the case for pre-trained models for NLP and computer vision. Training such models requires a lot of computational resources (e.g., HPC clusters) that are not available to small research groups and independent researchers. One way to address it is for several smaller groups to pool their computational resources together and train a model that benefits all participants. Unfortunately, in this case, any participant can jeopardize the entire training run by sending incorrect updates, deliberately or by mistake. Training in presence of such peers requires specialized distributed training algorithms with Byzantine tolerance. These algorithms often sacrifice efficiency by introducing redundant communication or passing all updates through a trusted server, making it infeasible to apply them to large-scale deep learning, where models can have billions of parameters. In this work, we propose a novel protocol for secure (Byzantine-tolerant) decentralized training that emphasizes communication efficiency.}
}

```

["Training Transformers Together"](https://arxiv.org/abs/2207.03481)

```
@misc{alex2022training,
    title={Training Transformers Together},
    author={Alexander Borzunov and Max Ryabinin and Tim Dettmers and Quentin Lhoest and Lucile Saulnier and Michael Diskin and Yacine Jernite and Thomas Wolf},
    year={2022},
    eprint={2207.03481},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

</details>

We also maintain a list
of [related projects and acknowledgements](https://learning-at-home.readthedocs.io/en/latest/user/acknowledgements.html).
