from typing import Optional, Union, List, Tuple
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

    def __init__(self, node_id: DHTID, bucket_size: int, depth_modulo: int,
                 staleness_timeout: float, wait_timeout: float):
        super().__init__(wait_timeout)
        self.node_id, self.bucket_size = node_id, bucket_size
        self.routing_table = RoutingTable(
            node_id, bucket_size, depth_modulo, staleness_timeout)
        self.storage = LocalStorage()

    def rpc_ping(self, sender: Endpoint, sender_id_bytes: BinaryDHTID) -> BinaryDHTID:
        """ Some dht node wants us to add it to our routing table. """
        self.routing_table.register_request_from(
            sender, DHTID.from_bytes(sender_id_bytes))
        return bytes(self.node_id)

    async def call_ping(self, recipient: Endpoint) -> Optional[DHTID]:
        """ Get recipient's node id and add him to the routing table. If recipient doesn't respond, return None """
        responded, response = await self.ping(recipient, bytes(self.node_id))
        recipient_node_id = DHTID.from_bytes(response) if responded else None
        self.routing_table.register_request_to(
            recipient, recipient_node_id, responded=responded)
        return recipient_node_id

    def rpc_store(self, sender: Endpoint, sender_id_bytes: BinaryDHTID,
                  key_bytes: BinaryDHTID, value: DHTValue, expiration_time: DHTExpiration) -> Tuple[bool, BinaryDHTID]:
        """ Some node wants us to store this (key, value) pair """
        self.routing_table.register_request_from(
            sender, DHTID.from_bytes(sender_id_bytes))
        store_accepted = self.storage.store(
            DHTID.from_bytes(key_bytes), value, expiration_time)
        return store_accepted, bytes(self.node_id)

    async def call_store(self, recipient: Endpoint, key: DHTID, value: DHTValue,
                         expiration_time: DHTExpiration) -> Optional[bool]:
        """
        Ask a recipient to store (key, value) pair until expiration time or update their older value
        :returns: True if value was accepted, False if it was rejected (recipient has newer value), None if no response
        """
        responded, response = await self.store(recipient, bytes(self.node_id), bytes(key), value, expiration_time)
        if responded:
            self.routing_table.register_request_to(
                recipient, DHTID.from_bytes(response[1]), responded=responded)
            # response[0] is True if an update was accepted, False if rejected
            return response[0]
        return None

    def rpc_find_node(self, sender: Endpoint, sender_id_bytes: BinaryDHTID,
                      key_bytes: BinaryDHTID) -> Tuple[List[Tuple[BinaryDHTID, Endpoint]], BinaryDHTID]:
        """
        Someone wants to find :key_node: in the DHT. Give him k nearest neighbors from our routing table
        :returns: a list of pairs (node_id, address) of :bucket_size: nearest to key_node according to XOR distance,
         also returns our own node id for routing table maintenance
        """
        key_node_id, sender_id = DHTID.from_bytes(
            key_bytes), DHTID.from_bytes(sender_id_bytes)
        self.routing_table.register_request_from(sender, sender_id)
        peer_ids = self.routing_table.get_nearest_neighbors(
            key_node_id, k=self.bucket_size, exclude=sender_id)
        return [(bytes(peer_id), self.routing_table[peer_id]) for peer_id in peer_ids], bytes(self.node_id)

    async def call_find_node(self, recipient: Endpoint, key_node_id: DHTID) -> List[Tuple[DHTID, Endpoint]]:
        """
        Ask a recipient to give you nearest neighbors to key_node. If recipient knows key_node directly,
         it will be returned as first of the neighbors; if recipient does not respond, return empty list.
        :returns: a list of pairs (node id, address) as per Section 2.3 of the paper
        """
        responded, response = await self.find_node(recipient, bytes(self.node_id), bytes(key_node_id))
        if responded:
            peers = [(DHTID.from_bytes(peer_id_bytes), tuple(addr))
                     for peer_id_bytes, addr in response[0]]
            # Note: we convert addr from list to tuple here --^ because some msgpack versions convert tuples to lists
            self.routing_table.register_request_to(
                recipient, DHTID.from_bytes(response[1]), responded=responded)
            return peers
        return []

    def rpc_find_value(self, sender: Endpoint, sender_id_bytes: BinaryDHTID, key_bytes: BinaryDHTID) -> \
            Tuple[Optional[DHTValue], Optional[DHTExpiration], List[Tuple[BinaryDHTID, Endpoint]], BinaryDHTID]:
        """
        Someone wants to find value corresponding to key. If we have the value, return the value and its expiration time
         Either way, return :bucket_size: nearest neighbors to that node.
        :note: this is a deviation from Section 2.3 of the paper, original kademlia returner EITHER value OR neighbors
        :returns: (value or None if we have no value, nearest neighbors, our own dht id)
        """
        return self.storage.get(DHTID.from_bytes(key_bytes)) + self.rpc_find_node(sender, sender_id_bytes, key_bytes)

    async def call_find_value(self, recipient: Endpoint, key: BinaryDHTID) -> \
            Tuple[Optional[DHTValue], Optional[DHTExpiration], List[Tuple[DHTID, Endpoint]]]:
        """
        Ask a recipient to give you the value, if it has one, or nearest neighbors to your key.
        :returns: (optional value, optional expiration time, and neighbors)
         value: whatever was the latest value stored by the recepient with that key (see DHTNode contract)
         expiration time: expiration time of the returned value, None if no value was found
         neighbors:  a list of pairs (node id, address) as per Section 2.3 of the paper;
        Note: if no response, returns None, None, []
        """
        responded, response = await self.find_value(recipient, bytes(self.node_id), bytes(key))
        if responded:
            value, expiration_time, peers_bytes, recipient_id_bytes = response
            peers = [(DHTID.from_bytes(peer_id_bytes), tuple(addr))
                     for peer_id_bytes, addr in peers_bytes]
            self.routing_table.register_request_to(
                recipient, DHTID.from_bytes(recipient_id_bytes), responded=responded)
            return value, expiration_time, peers
        return None, None, []


class LocalStorage(dict):
    def store(self, key: DHTID, value: DHTValue, expiration_time: DHTExpiration) -> bool:
        """
        Store a (key, value) pair locally at least until expiration_time. See class docstring for details.
        :returns: True if new value was stored, False it was rejected (current value is newer)
        """
        self[key] = (value, expiration_time)
        return True  # TODO implement actual local storage, test that the logic is correct

    def get(self, key: DHTID) -> (Optional[DHTValue], Optional[DHTExpiration]):
        """ Get a value corresponding to a key if that (key, value) pair was previously stored here. """
        return self[key] if key in self else (None, None)
