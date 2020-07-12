**Hivemind Server**
========================================

.. automodule:: hivemind.server

.. currentmodule:: hivemind.server

.. autoclass:: Server
   :members:
   :member-order: bysource

.. autoclass:: Runtime
    :members:
    :member-order: bysource

.. autoclass:: ExpertBackend
    :members: forward, backward, apply_gradients, get_info, get_pools
    :member-order: bysource

.. autoclass:: TaskPool
    :members: submit_task, iterate_minibatches, load_batch_to_runtime, send_outputs_from_runtime, get_task_size, empty
    :member-order: bysource