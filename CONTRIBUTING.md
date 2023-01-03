# Contributing to hivemind

This document covers the technical details of making your contributions to the code of hivemind. For other ways to
contribute, read the [contributing guide](https://learning-at-home.readthedocs.io/en/latest/user/contributing.html) in
our documentation.

Before you begin, file a new issue on [GitHub](https://github.com/learning-at-home/hivemind/issues) or announce that
you are going to work on an existing one to avoid duplicate effort. After you finish, submit a pull request and wait
for it to be reviewed by the library maintainers (and possibly other community members).

## Environment setup

First, install hivemind in the development mode, preferably with Python 3.8+ on Linux.

```
git clone https://github.com/learning-at-home/hivemind
cd hivemind
pip install -e .[dev]
``` 

## Pull Request checklist

To make sure that the reviewers will request minimal changes in your PR, you can check that your contribution complies
with the following rules:

* All code changes are consistent with the repository [code style](#code-style).
* The title and the description of your pull request adhere to the formatting guidelines both
  for [pull requests](#pull-requests) and for [commit messages](#commit-messages).
* New modules or functions are sufficiently [documented](#building-documentation) and covered
  with [tests](#running-tests).
* The CI pipelines both for the documentation and for tests pass successfully.
* If you make performance-sensitive changes, their impact is measured with [benchmarks](#running-benchmarks) (the more,
  the better).

## Code style

* The code must follow [PEP8](https://www.python.org/dev/peps/pep-0008/) unless absolutely necessary. Also, each line
  cannot be longer than 119 characters.
* We use [black](https://github.com/psf/black) for code formatting and [isort](https://github.com/PyCQA/isort) for 
  import sorting. Before submitting a PR, make sure to install and run `black .` and `isort .` in the root of the
  repository. Also, you may want to check your code for typos by running `codespell --skip=".git"`, though there
  might be false positives.
* We highly encourage the use of [typing](https://docs.python.org/3/library/typing.html) where applicable.
* Use `get_logger` from `hivemind.utils.logging` to log any information instead of `print`ing directly to standard
  output/error streams.
* Comments should be used sparingly and never describe the obvious. Usually it's best to clean up the code logic
  instead of describing it, as it might lead to redundant (or worse, stale or incorrect) messages.
* In general, strive for code readability instead of compactness. In particular, prefer to create a new variable
  instead of a long one-liner and to break up a long method into several meaningful parts. This rule can be overridden
  in case of major performance considerations, but only if verified by benchmarks.
* Each user-facing function must have a [correct](#building-documentation) docstring that describes the intended usage,
  the input arguments and the return value. Both in comments and docstrings, please try to follow the capitalization
  rules for all terms and objects and to use proper grammar.

## Contribution formatting guidelines

To make sure that each change to hivemind is consistent across the entire project history and is easy to review by any
community member, follow these guidelines when submitting your pull request and writing commit messages. The library
maintainers use the same rules when merging your commits into the master branch after the PR approval.

### Commit messages

To ensure a consistent format across the entire repository history, please follow the following rules when formatting
your commits (especially for PR merge commits):

* Keep the subject line short, preferably under 50 characters.
* Capitalize the subject line and do not end it with a period.
* If possible, write a conceptual description of your commit in the body (why?) instead

It is not required to use this format while you are still working on your pull request. However, each merged PR commit
message has to adhere to these guidelines, and it will be easier for the maintainers to accept the PR if you have
already done most of the necessary formatting work.

For further reading on the commit message format, see
this [guide](https://chris.beams.io/posts/git-commit/#seven-rules) on good Git commit messages, as well as
this [repository](https://github.com/RomuloOliveira/commit-messages-guide).

### Pull requests

All commits from a pull request are squashed before merging to ensure a clean commit history in the master branch. The
merge commit title is the name of the pull request along with the PR number reference; the merge commit body is either
the pull request description (if it adheres to the format) or a cleaned up compilation of PR branch commit messages.

* As such, the name and the description of your PR should follow the same guidelines as commit messages.
* Try to make your pull requests more narrow in scope and split significant changes to the code base in separate
  pieces. This will ensure [faster and better](https://essenceofcode.com/2019/10/29/the-art-of-small-pull-requests/)
  feedback from the reviewers.
* In particular, try to separate functional and non-functional code changes, as well as independent functional changes
  if they make the pull request too large to review in a short period of time.
* In general, when naming a pull request instead of a commit, it's best to highlight the major change in its title
  instead of listing all modifications. Also, if a pull request makes significant changes to the library, it's best to
  give a high-level description in the title instead of a technical one:
  compare `Implement decentralized parameter averaging` with `Add hivemind.client.averaging`.

For more on the philosophy of easy-to-review pull requests, read these
guides: [1](https://mtlynch.io/code-review-love/)
[2](https://www.atlassian.com/blog/git/written-unwritten-guide-pull-requests). If the changelist is not very large
(more than a hundred lines) already, we encourage making small improvements to the codebase in the files already
changed by the PR; however, they should not dilute its major purpose.

## Running tests

Hivemind uses [pytest](https://github.com/pytest-dev/pytest/) for testing the behavior of the library modules. If you
implement a new part of the library, you are expected to write a test for the correctness of its implementation. If you
discovered a bug in the existing code base and intend to fix it, it's also best if you add the steps to reproduce it as
a new test to make sure it's not reintroduced by future changes.

To run tests, you need to install hivemind in development mode with additional dependencies: `pip install -e .[dev]`.
You can run all tests with `pytest tests/` or choose a specific subset, e.g., `pytest tests/test_dht.py`.

When investigating test behavior, please note that pytest automatically wraps all hivemind tests with fixtures defined
in a global configuration file [`tests/conftest.py`](./tests/conftest.py), some of which will run automatically. For
more informantion, refer to the [pytest documentation on fixtures](https://docs.pytest.org/en/6.2.x/fixture.html).

## Building documentation

Any function exposed to a user must have a docstring compatible
with [Sphinx](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html), which is used for building the
online documentation.

To build the docs locally,

1. `pip install -e .[docs]`
2. make sure you ran setup.py (see above)
3. `cd ./docs && make html`

The documentation root will be available in `./docs/_build/html/index.html`

## Running benchmarks

Currently, hivemind has three benchmark scripts for evaluating the impact of code changes on the most
performance-sensitive parts of the library. If you make a change that might introduce a regression, you may be asked by
the maintainers to provide the benchmarking results for your branch and a comparison with the master branch.

* `benchmarks/benchmark_averaging.py` measures the performance of decentralized parameter averaging across the DHT.
* `benchmarks/benchmark_dht.py` measures the performance of core DHT operations.
* `benchmarks/benchmark_throughput.py` measures the performance of a server hosting several expert layers under heavy
  load from multiple clients.

Example benchmark runs are available in
the [benchmarking](https://learning-at-home.readthedocs.io/en/latest/user/benchmarks.html) page of the documentation.

## See also

For more details on overall contributions, visit the contributing guide at:

https://learning-at-home.readthedocs.io/en/latest/user/contributing.html

This guide was inspired by several influential Python open source projects listed below:

* [PyTorch](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)
* [Scikit-learn](https://scikit-learn.org/dev/developers/contributing.html)
* [transformers](https://github.com/huggingface/transformers/blob/master/CONTRIBUTING.md)
