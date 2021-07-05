**hivemind.optim**
==================

.. automodule:: hivemind.optim
.. currentmodule:: hivemind.optim

.. raw:: html

  This module contains decentralized optimizers that wrap regular pytorch optimizers to collaboratively train a shared model. Depending on the exact type, optimizer may average model parameters with peers, exchange gradients, or follow a more complicated distributed training strategy.
  <br><br>

.. autoclass:: CollaborativeOptimizer
   :members: step
   :member-order: bysource

.. autoclass:: CollaborativeAdaptiveOptimizer
   :members:
   :member-order: bysource
