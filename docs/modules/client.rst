**hivemind.client**
====================

.. automodule:: hivemind.client

.. currentmodule:: hivemind.client

.. raw:: html

  This module lets you connect to distributed Mixture-of-Experts or individual experts hosted
  <strike>in the cloud cloud</strike> on someone else's computer.
  <br><br>

.. autoclass:: RemoteExpert
   :members: forward

.. autoclass:: RemoteMixtureOfExperts
   :members:
   :member-order: bysource

.. autoclass:: DecentralizedAverager
   :members:
   :member-order: bysource
   :exclude-members: get_tensors, get_tensors_async, update_tensors, rpc_join_group, rpc_aggregate_part
