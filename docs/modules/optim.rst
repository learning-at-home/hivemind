**hivemind.optim**
==================

.. raw:: html

  This module contains decentralized optimizers that wrap your regular PyTorch Optimizer to train with peers.
  Depending on the exact configuration, Optimizer may perform large synchronous updates equivalent,
  or perform asynchrnous local updates and average model parameters.

  <br><br>

.. automodule:: hivemind.optim.experimental.optimizer
.. currentmodule:: hivemind.optim.experimental.optimizer

**hivemind.Optimizer**
----------------------

.. autoclass:: Optimizer
   :members: step, local_epoch, zero_grad, load_state_from_peers, param_groups, shutdown
   :member-order: bysource

.. currentmodule:: hivemind.optim.grad_scaler
.. autoclass:: GradScaler
   :member-order: bysource


**CollaborativeOptimizer**
--------------------------

.. raw:: html

  CollaborativeOptimizer is a legacy version of hivemind.Optimizer. <b>For new projects please use hivemind.Optimizer</b>.
  Currently, hivemind.Optimizer supports all the features of CollaborativeOptimizer and then some.
  CollaborativeOptimizer will still be supported for a while, but eventually it will be deprecated.
  <br><br>


.. automodule:: hivemind.optim.collaborative
.. currentmodule:: hivemind.optim

.. autoclass:: CollaborativeOptimizer
   :members: step
   :member-order: bysource

.. autoclass:: CollaborativeAdaptiveOptimizer
   :members:
   :member-order: bysource
