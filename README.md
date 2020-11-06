## hivemind: decentralized deep learning in PyTorch
[![Build status](https://circleci.com/gh/learning-at-home/hivemind.svg?style=shield)](https://circleci.com/gh/learning-at-home/hivemind)
[![Documentation Status](https://readthedocs.org/projects/learning-at-home/badge/?version=latest)](https://learning-at-home.readthedocs.io/en/latest/?badge=latest)
[![Gitter](https://badges.gitter.im/learning-at-home/hivemind.svg)](https://gitter.im/learning-at-home/hivemind?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Hivemind is a PyTorch library to train large neural networks across the Internet. Imagine training one huge Transformer model
  on thousands of computers from different universities, companies, and volunteers.

![img](https://i.imgur.com/GPxolxb.gif)

### Key Features
 * Train neural networks of arbitrary size: parts of their layers are distributed across the participants
 * Run distributed training without master node: Distributed Hash Table allows to connect computers in a decentralized network
 * Fault-tolerant backpropagation: forward and backward passes succeed even if some nodes are unresponsive or take too long to respond

To learn more about the idea behind this library and its components, see https://learning-at-home.github.io or read the [NeurIPS 2020 paper](https://arxiv.org/abs/2002.04013)

### Documentation
 * [Quickstart tutorial](https://learning-at-home.readthedocs.io/en/latest/user/quickstart.html): install hivemind, 
    set up a server and train experts  
 * Documentation & guides: [learning-at-home.readthedocs.io](https://learning-at-home.readthedocs.io)

### Contributing
Hivemind is currently at the active development stage, and we welcome all contributions from bug fixes and documentation improvements to entirely new features. 
If you want to contribute to hivemind, take a look at the issues or join [our chat room](https://gitter.im/learning-at-home/hivemind).
The [Developer's guide](https://learning-at-home.readthedocs.io/en/latest/user/contributing.html) page contains best practices, as well as description of tests and performance benchmarks.

### References
You can read the paper that inspired hivemind here:

[Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts](https://arxiv.org/abs/2002.04013) (Max Ryabinin and Anton Gusev, NeurIPS 2020).
```
@misc{ryabinin2020crowdsourced,
      title={Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts}, 
      author={Max Ryabinin and Anton Gusev},
      year={2020},
      eprint={2002.04013},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```
The initial implementation of hivemind used to conduct experiments for the paper is available here: [mryab/learning-at-home](https://github.com/mryab/learning-at-home).

In the docs, we list several [related](https://learning-at-home.readthedocs.io/en/latest/user/acknowledgements.html) projects and acknowledgements.

