**hivemind.moe.server**
========================================

A hivemind server hosts one or several experts and processes incoming requests to those experts. It periodically
re-publishes these experts to the dht via a dedicated **hivemind.dht.DHT** peer that runs in background.
The experts can be accessed directly as **hivemind.moe.client.RemoteExpert("addr:port", "expert.uid.here")**
or as a part of **hivemind.moe.client.RemoteMixtureOfExperts** that finds the most suitable experts across the DHT.

The hivemind.moe.server module is organized as follows:

- Server_ is the main class that publishes experts, accepts incoming requests, and passes them to Runtime_ for compute.
- ModuleBackend_ is a wrapper for `torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ \
  that can be accessed by remote clients. It has two TaskPool_ s for forward and backward requests.
- Runtime_ balances the device (GPU) usage between several ModuleBackend_ instances that each service one expert.
- TaskPool_ stores incoming requests for a batch-parallel computation (e.g. forward pass), groups them into batches \
  and offers those batches to Runtime_ for processing.


.. automodule:: hivemind.moe.server

.. currentmodule:: hivemind.moe.server

.. _Server:
.. autoclass:: Server
   :members:
   :member-order: bysource

.. _ModuleBackend:
.. autoclass:: ModuleBackend
    :members: forward, backward, on_backward, get_info, get_pools
    :member-order: bysource

.. currentmodule:: hivemind.moe.server.runtime

.. _Runtime:
.. autoclass:: Runtime
    :members:
    :member-order: bysource

.. currentmodule:: hivemind.moe.server.task_pool

.. _TaskPool:
.. autoclass:: TaskPool
    :members: submit_task, iterate_minibatches, load_batch_to_runtime, send_outputs_from_runtime, get_task_size, empty
    :member-order: bysource