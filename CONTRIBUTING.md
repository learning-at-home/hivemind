

### Rules for collaborating:
Hivemind is still in the early stage of development, we expect only a handful of collaborators with individual roles.

1. Before you write any code, please contact us to avoid duplicate work:
   * Report bugs and propose new features via issues. We don't have templates at this point;
   * If you decide to implement a feature or fix a bug, leave a comment in the appropriate issue or create a new one;
   * Please follow [Contributor Convent v2.0](https://www.contributor-covenant.org/version/2/0/code_of_conduct/).
2. When you code, follow the best practices:
   * We use [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)-style development;
   * The code itself must follow [PEP8](https://www.python.org/dev/peps/pep-0008/). We recommend using pycharm builtin linter;
   * We highly encourage the use of typing, where applicable; If not applicable, use other tools like docstrings;
3. After you write the code, make sure others can use it:
   * Any function exposed to a user must have a docstring compatible with [sphinx](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html);
   * For new features, please write test(s) to make sure your functionality won't be broken by subsequent changes;
   * If you face any challenges or want feedback, please submit pull request early with a [WIP] tag = work in progress.



### Tips & tricks
* You can find a wealth of pytorch debugging tricks at [their contributing page](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md).
* Hivemind is optimized for development in pycharm CE 2019.3 or newer.
  * When working on tests, please mark "tests" as sources root.
