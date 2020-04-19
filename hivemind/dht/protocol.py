from typing import Optional
from rpcudp.protocol import RPCProtocol

from .node import DHTNode
from .routing import RoutingTable, DHTID
from ..utils import Hostname, Port, Tuple, Endpoint


class KademliaProtocol(RPCProtocol):
    """
    A protocol that allows DHT nodes to request keys/neighbors from other DHT nodes.
    As a side-effect, KademliaProtocol also maintains a routing table as described in
    https://pdos.csail.mit.edu/~petar/papers/maymounkov-kademlia-lncs.pdf

    Node: the rpc_* methods defined in this class will be automatically exposed to other DHT nodes,
     for instance, def rpc_ping can be called as protocol.ping() from a remote machine
     Read more: https://github.com/bmuller/rpcudp/tree/master/rpcudp
    """

    def __init__(self, node: DHTNode, bucket_size: int, modulo: int):
        self.node, self.bucket_size = node, bucket_size
        self.routing_table = RoutingTable(node, bucket_size, modulo)

    def rpc_ping(self, sender: Endpoint, sender_node_id: DHTID) -> DHTID:
        """ Some dht node wants us to add it to our routing table. """
        self.routing_table.register_request_from(sender, sender_node_id)
        return self.node.id

    def call_ping(self, recipient: Endpoint) -> Optional[DHTID]:
        """ Get recipient's node id and add him to the routing table. If recipient doesn't respond, return None """
        responded, recipient_node_id = self.ping(recipient, self.node.id)
        self.routing_table.register_request_to(
            recipient, recipient_node_id, responded=responded)
        return recipient_node_id
