import asyncio
import time
from typing import Optional, List, Tuple, Dict
from rpcudp.protocol import RPCProtocol

from .routing import RoutingTable, DHTID, DHTValue, DHTExpiration, BinaryDHTID
from ..utils import Endpoint


class KademliaProtocol(RPCProtocol):
    """
    A protocol that allows DHT nodes to request keys/neighbors from other DHT nodes.
    As a side-effect, KademliaProtocol also maintains a routing table as described in
    https://pdos.csail.mit.edu/~petar/papers/maymounkov-kademlia-lncs.pdf

    See DHTNode (node.py) for a more detailed description.

    :note: the rpc_* methods defined in this class will be automatically exposed to other DHT nodes,
     for instance, def rpc_ping can be called as protocol.call_ping(addr, dht_id) from a remote machine
     Only the call_* methods are meant to be called publicly, e.g. from DHTNode
     Read more: https://github.com/bmuller/rpcudp/tree/master/rpcudp
    """

    def __init__(self, node_id: DHTID, bucket_size: int, depth_modulo: int, wait_timeout: float):
        super().__init__(wait_timeout)
        self.node_id, self.bucket_size = node_id, bucket_size
        self.routing_table = RoutingTable(node_id, bucket_size, depth_modulo)
        self.storage = LocalStorage()

    def rpc_ping(self, sender: Endpoint, sender_id_bytes: BinaryDHTID) -> BinaryDHTID:
        """ Some dht node wants us to add it to our routing table. """
        asyncio.ensure_future(self.update_routing_table(DHTID.from_bytes(sender_id_bytes), sender))
        return bytes(self.node_id)

    async def call_ping(self, recipient: Endpoint) -> Optional[DHTID]:
        """ Get recipient's node id and add him to the routing table. If recipient doesn't respond, return None """
        responded, response = await self.ping(recipient, bytes(self.node_id))
        recipient_node_id = DHTID.from_bytes(response) if responded else None
        asyncio.ensure_future(self.update_routing_table(recipient_node_id, recipient, responded=responded))
        return recipient_node_id

    def rpc_store(self, sender: Endpoint, sender_id_bytes: BinaryDHTID,
                  key_bytes: BinaryDHTID, value: DHTValue, expiration_time: DHTExpiration) -> Tuple[bool, BinaryDHTID]:
        """ Some node wants us to store this (key, value) pair """
        asyncio.ensure_future(self.update_routing_table(DHTID.from_bytes(sender_id_bytes), sender))
        store_accepted = self.storage.store(DHTID.from_bytes(key_bytes), value, expiration_time)
        return store_accepted, bytes(self.node_id)

    async def call_store(self, recipient: Endpoint, key: DHTID, value: DHTValue,
                         expiration_time: DHTExpiration) -> Optional[bool]:
        """
        Ask a recipient to store (key, value) pair until expiration time or update their older value
        :returns: True if value was accepted, False if it was rejected (recipient has newer value), None if no response
        """
        responded, response = await self.store(recipient, bytes(self.node_id), bytes(key), value, expiration_time)
        if responded:
            store_accepted, recipient_node_id = response[0], DHTID.from_bytes(response[1])
            asyncio.ensure_future(self.update_routing_table(recipient_node_id, recipient, responded=responded))
            return store_accepted
        return None

    def rpc_find_node(self, sender: Endpoint, sender_id_bytes: BinaryDHTID,
                      query_id_bytes: BinaryDHTID) -> Tuple[List[Tuple[BinaryDHTID, Endpoint]], BinaryDHTID]:
        """
        Someone wants to find :key_node: in the DHT. Give him k nearest neighbors from our routing table
        :returns: a list of pairs (node_id, address) of :bucket_size: nearest to key_node according to XOR distance,
         also returns our own node id for routing table maintenance
        """
        query_id, sender_id = DHTID.from_bytes(query_id_bytes), DHTID.from_bytes(sender_id_bytes)
        asyncio.ensure_future(self.update_routing_table(sender_id, sender))
        peer_ids_and_addr = self.routing_table.get_nearest_neighbors(query_id, k=self.bucket_size, exclude=sender_id)
        return [(bytes(peer_id), peer_addr) for peer_id, peer_addr in peer_ids_and_addr], bytes(self.node_id)

    async def call_find_node(self, recipient: Endpoint, query_id: DHTID) -> Dict[DHTID, Endpoint]:
        """
        Ask a recipient to give you nearest neighbors to key_node. If recipient knows key_node directly,
         it will be returned as first of the neighbors; if recipient does not respond, return empty dict.
        :returns: a dicitionary[node id => address] as per Section 2.3 of the paper
        """
        responded, response = await self.find_node(recipient, bytes(self.node_id), bytes(query_id))
        if responded:
            peers = {DHTID.from_bytes(peer_id_bytes): tuple(addr) for peer_id_bytes, addr in response[0]}
            # Note: we convert addr from list to tuple here --^ because some msgpack versions convert tuples to lists
            recipient_node_id = DHTID.from_bytes(response[1])
            asyncio.ensure_future(self.update_routing_table(recipient_node_id, recipient, responded=responded))
            return peers
        return {}

    def rpc_find_value(self, sender: Endpoint, sender_id_bytes: BinaryDHTID, key_bytes: BinaryDHTID) -> \
            Tuple[Optional[DHTValue], Optional[DHTExpiration], List[Tuple[BinaryDHTID, Endpoint]], BinaryDHTID]:
        """
        Someone wants to find value corresponding to key. If we have the value, return the value and its expiration time
         Either way, return :bucket_size: nearest neighbors to that node.
        :note: this is a deviation from Section 2.3 of the paper, original kademlia returner EITHER value OR neighbors
        :returns: (value or None if we have no value, nearest neighbors, our own dht id)
        """
        maybe_value, maybe_expiration = self.storage.get(DHTID.from_bytes(key_bytes))
        nearest_neighbors, my_id = self.rpc_find_node(sender, sender_id_bytes, key_bytes)
        return maybe_value, maybe_expiration, nearest_neighbors, my_id

    async def call_find_value(self, recipient: Endpoint, key: BinaryDHTID) -> \
            Tuple[Optional[DHTValue], Optional[DHTExpiration], Dict[DHTID, Endpoint]]:
        """
        Ask a recipient to give you the value, if it has one, or nearest neighbors to your key.
        :returns: (optional value, optional expiration time, and neighbors)
         value: whatever was the latest value stored by the recipient with that key (see DHTNode contract)
         expiration time: expiration time of the returned value, None if no value was found
         neighbors:  a dictionary[node id => address] as per Section 2.3 of the paper;
        Note: if no response, returns None, None, {}
        """
        responded, response = await self.find_value(recipient, bytes(self.node_id), bytes(key))
        if responded:
            (value, expiration_time, peers_bytes), recipient_id = response[:-1], DHTID.from_bytes(response[-1])
            peers = {DHTID.from_bytes(peer_id_bytes): tuple(addr) for peer_id_bytes, addr in peers_bytes}
            asyncio.ensure_future(self.update_routing_table(recipient_id, recipient, responded=responded))
            return value, expiration_time, peers
        return None, None, {}

    async def update_routing_table(self, node_id: Optional[DHTID], addr: Endpoint, responded=True):
        """
        This method is called on every incoming AND outgoing request to update the routing table
        :param addr: sender endpoint for incoming requests, recipient endpoint for outgoing requests
        :param node_id: sender node id for incoming requests, recipient node id for outgoing requests
        :param responded: for outgoing requests, this indicated whether recipient responded or not.
          For incoming requests, this should always be True
        """
        if responded:  # incoming request or outgoing request with response
            maybe_node_to_ping = self.routing_table.add_or_update_node(node_id, addr)
            if maybe_node_to_ping is not None:
                # we couldn't add new node because the table was full. Check if existing peers are alive (Section 2.2)
                # ping one least-recently updated peer: if it won't respond, remove it from the table, else update it
                await self.call_ping(maybe_node_to_ping[1])  # [1]-th element is that node's endpoint

        else:  # outgoing request and peer did not respond
            if node_id is not None and node_id in self.routing_table:
               del self.routing_table[node_id]


class LocalStorage(dict):
    def __init__(self, maxsize: Optional[int] = None, keep_expired: bool = True):
        self.maxsize = maxsize or float("inf")
        self.keep_expired = keep_expired
        super().__init__()

    def store(self, key: DHTID, value: DHTValue, expiration_time: DHTExpiration) -> bool:
        """
        Store a (key, value) pair locally at least until expiration_time. See class docstring for details.
        :returns: True if new value was stored, False it was rejected (current value is newer)
        """
        if len(self) >= self.maxsize:
            del self[min(self, key=lambda k:self[k][1])]
        if key in self:
            if self[key][1] < expiration_time:
                self[key] = (value, expiration_time)
                return True
            return False
        self[key] = (value, expiration_time)
        return True

    def get(self, key: DHTID) -> (Optional[DHTValue], Optional[DHTExpiration]):
        """ Get a value corresponding to a key if that (key, value) pair was previously stored here. """
        if key in self:
            if self[key][1] >= time.monotonic():
                if self.keep_expired:
                    return self[key]
            else:
                return self[key]
        return None, None
