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
see the [full list](#citation) of our papers below.

## Example Use Cases

This section lists projects that leverage hivemind for decentralized training. 
If you have successfully trained a model or created a downstream repository with the help of our library, 
feel free to submit a pull request that adds your project to this list.

* **Petals** ([webpage](https://petals.ml), [code](https://github.com/bigscience-workshop/petals)) — a decentralized platform for inference and fine-tuning of 100B+ language models.
* **Training Transformers Together** ([webpage](https://training-transformers-together.github.io/), [code](https://github.com/learning-at-home/dalle-hivemind)) — a NeurIPS 2021 demonstration that trained a collaborative text-to-image Transformer model.
* **CALM** ([webpage](https://huggingface.co/CALM), [code](https://github.com/NCAI-Research/CALM)) — a masked language model trained on a combination of Arabic datasets.
* **sahajBERT** ([blog post](https://huggingface.co/blog/collaborative-training), [code](https://github.com/tanmoyio/sahajbert)) — a collaboratively pretrained ALBERT-xlarge for the Bengali language.
* **HivemindStrategy** ([docs](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.strategies.HivemindStrategy.html)) in PyTorch Lightning allows adapting your existing pipelines to training over slow network with unreliable peers.

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

Also, if you want to use blockwise 8-bit compression from [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 
during data transfer, you can install it with `pip install hivemind[bitsandbytes]`. 
After that, you can use the `BlockwiseQuantization` class in [hivemind.compression](./hivemind/compression)

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
  title = {{H}ivemind: a {L}ibrary for {D}ecentralized {D}eep {L}earning},
  author = {Learning{@}home team},
  year = 2020,
  howpublished = {\url{https://github.com/learning-at-home/hivemind}}
}
```

Also, you can cite [the paper](https://arxiv.org/abs/2002.04013) that inspired the creation of this library
(prototype implementation of hivemind available
at [mryab/learning-at-home](https://github.com/mryab/learning-at-home)):

```bibtex
@inproceedings{ryabinin2020crowdsourced,
  title = {Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts},
  author = {Ryabinin, Max and Gusev, Anton},
  year = 2020,
  booktitle = {Advances in Neural Information Processing Systems},
  volume = 33,
  url = {https://proceedings.neurips.cc/paper/2020/file/25ddc0f8c9d3e22e03d3076f98d83cb2-Paper.pdf}
}
```

<details>
 <summary>Additional publications</summary>

["Moshpit SGD: Communication-Efficient Decentralized Training on Heterogeneous Unreliable Devices"](https://arxiv.org/abs/2103.03239)

```bibtex
@inproceedings{ryabinin2021moshpit,
  title = {Moshpit SGD: Communication-Efficient Decentralized Training on Heterogeneous Unreliable Devices},
  author = {Ryabinin, Max and Gorbunov, Eduard and Plokhotnyuk, Vsevolod and Pekhimenko, Gennady},
  year = 2021,
  booktitle = {Advances in Neural Information Processing Systems},
  volume = 34,
  url = {https://proceedings.neurips.cc/paper/2021/file/97275a23ca44226c9964043c8462be96-Paper.pdf}
}
```

["Distributed Deep Learning in Open Collaborations"](https://arxiv.org/abs/2106.10207)

```bibtex
@inproceedings{diskin2021distributed,
  title = {Distributed Deep Learning In Open Collaborations},
  author = {Michael Diskin and Alexey Bukhtiyarov and Max Ryabinin and Lucile Saulnier and Quentin Lhoest and Anton Sinitsin and Dmitry Popov and Dmitriy Pyrkin and Maxim Kashirin and Alexander Borzunov and Albert Villanova del Moral and Denis Mazur and Ilia Kobelev and Yacine Jernite and Thomas Wolf and Gennady Pekhimenko},
  year = 2021,
  booktitle = {Advances in Neural Information Processing Systems},
  url = {https://openreview.net/forum?id=FYHktcK-7v}
}
```

["Secure Distributed Training at Scale"](https://arxiv.org/abs/2106.11257)

```bibtex
@inproceedings{gorbunov2022secure,
  title = {Secure Distributed Training at Scale},
  author = {Gorbunov, Eduard and Borzunov, Alexander and Diskin, Michael and Ryabinin, Max},
  year = 2022,
  month = {17--23 Jul},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = 162,
  url = {https://proceedings.mlr.press/v162/gorbunov22a.html}
}
```

["Training Transformers Together"](https://arxiv.org/abs/2207.03481)

```bibtex
@misc{borzunov2022training,
  title = {Training Transformers Together},
  author = {Alexander Borzunov and Max Ryabinin and Tim Dettmers and Quentin Lhoest and Lucile Saulnier and Michael Diskin and Yacine Jernite and Thomas Wolf},
  year = 2022,
  eprint = {2207.03481},
  archiveprefix = {arXiv},
  primaryclass = {cs.LG}
}
```

</details>

We also maintain a list
of [related projects and acknowledgements](https://learning-at-home.readthedocs.io/en/latest/user/acknowledgements.html).
