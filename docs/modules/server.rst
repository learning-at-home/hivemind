``tesseract.server & tesseract.runtime``
========================================

.. automodule:: tesseract.server

.. currentmodule:: tesseract.server

.. autoclass:: TesseractServer
   :members:
   :member-order: bysource

.. currentmodule:: tesseract.runtime

.. autoclass:: TesseractRuntime
    :members:
    :member-order: bysource


.. autoclass:: ExpertBackend
    :members: forward, backward, apply_gradients, get_info, get_pools
    :member-order: bysource

.. autoclass:: TaskPool
    :members: submit_task, form_batch, load_batch_to_runtime, send_outputs_from_runtime, get_task_size, empty
    :member-order: bysource