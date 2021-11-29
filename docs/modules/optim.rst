**hivemind.optim**
==================

.. raw:: html

  This module contains decentralized optimizers that wrap regular pytorch optimizers to collaboratively train a shared model. Depending on the exact type, optimizer may average model parameters with peers, exchange gradients, or follow a more complicated distributed training strategy.
  <br><br>

.. automodule:: hivemind.optim.experimental.optimizer
.. currentmodule:: hivemind.optim.experimental.optimizer

**hivemind.Optimizer**
----------------------

.. autoclass:: Optimizer
   :members: step, zero_grad, load_state_from_peers, param_groups, shutdown
   :member-order: bysource

.. currentmodule:: hivemind.optim.grad_scaler
.. autoclass:: GradScaler
   :member-order: bysource


**CollaborativeOptimizer**
--------------------------

.. raw:: html

  CollaborativeOptimizer is a legacy version of hivemind.Optimizer. **For new projects, please use hivemind.Optimizer.**
  Currently, hivemind.Optimizer supports all the features of CollaborativeOptimizer and then some.
  CollaborativeOptimizer will still be supported for awhile, but will eventually be deprecated.
  <br><br>


.. automodule:: hivemind.optim.collaborative
.. currentmodule:: hivemind.optim

.. autoclass:: CollaborativeOptimizer
   :members: step
   :member-order: bysource

.. autoclass:: CollaborativeAdaptiveOptimizer
   :members:
   :member-order: bysource
