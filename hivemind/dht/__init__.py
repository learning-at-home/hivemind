"""
This is a Distributed Hash Table optimized for rapidly accessing a lot of lightweight metadata.
Hivemind DHT is based on Kademlia [1] with added support for improved bulk store/get operations and caching.

The code is organized as follows:

 * **class DHT (dht.py)** - high-level class for model training. Runs DHTNode in a background process.
 * **class DHTNode (node.py)** - an asyncio implementation of dht server, stores AND gets keys.
 * **class DHTProtocol (protocol.py)** - an RPC protocol to request data from dht nodes.
 * **async def traverse_dht (traverse.py)** - a search algorithm that crawls DHT peers.

- [1] Maymounkov P., Mazieres D. (2002) Kademlia: A Peer-to-Peer Information System Based on the XOR Metric.
- [2] https://github.com/bmuller/kademlia , Brian, if you're reading this: THANK YOU! you're awesome :)
"""

from hivemind.dht.dht import DHT
from hivemind.dht.node import DEFAULT_NUM_WORKERS, DHTNode
from hivemind.dht.routing import DHTID, DHTValue
from hivemind.dht.validation import CompositeValidator, RecordValidatorBase
