**hivemind.dht**
====================

.. automodule:: hivemind.dht
.. currentmodule:: hivemind.dht

Here's a high level scheme of how these components interact with one another:

.. image:: ../_static/dht.png
   :width: 640
   :align: center


**Note:** hivemind.DHT is currently being updated to improve beam search latency
(see `issue 92 <https://github.com/learning-at-home/hivemind/issues>`__). New functionality will be documented
here by 2020.10.15 23:59:59 AOE (ping justheuristic for details).

DHT and DHTNode
###############

.. autoclass:: DHT
   :members:
   :exclude-members: make_key
   :member-order: bysource

.. autoclass:: DHTNode
   :members:
   :member-order: bysource

DHT communication protocol
##########################
.. automodule:: hivemind.dht.protocol
.. currentmodule:: hivemind.dht.protocol

.. autoclass:: DHTProtocol
   :members:
   :member-order: bysource

.. currentmodule:: hivemind.dht.routing

.. autoclass:: RoutingTable
   :members:
   :member-order: bysource

.. autoclass:: KBucket
   :members:
   :member-order: bysource

.. autoclass:: DHTID
   :members:
   :exclude-members: HASH_FUNC
   :member-order: bysource

Traverse (crawl) DHT
####################

.. automodule:: hivemind.dht.traverse
.. currentmodule:: hivemind.dht.traverse

.. autofunction:: simple_traverse_dht

.. autofunction:: traverse_dht