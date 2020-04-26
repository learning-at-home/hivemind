from __future__ import annotations

import hashlib
import os
import random

import time
import heapq
from itertools import chain
from typing import Tuple, Optional, List, Dict, Set, Union, Any

from ..utils import Hostname, Port, Endpoint


class RoutingTable:
    """
    A data structure that contains DHT peers bucketed according to their distance to node_id
    :param node_id: node id used to measure distance
    :param bucket_size: parameter $k$ from Kademlia paper Section 2.2
    :param depth_modulo: parameter $b$ from Kademlia paper Section 2.2.
     In short, RoutingTable will can split bucket containing root indefinitely but will only split other buckets
      up to the nearest multiple of :depth_modulo:
    :param staleness_timeout: a bucket is considered stale if no node from that bucket
        was updated for this many seconds

    :note: kademlia paper refers to https://pdos.csail.mit.edu/~petar/papers/maymounkov-kademlia-lncs.pdf
    """

    def __init__(self, node_id: DHTID, bucket_size: int, depth_modulo: int, staleness_timeout: Union[float, int]):
        self.node_id, self.bucket_size = node_id, bucket_size
        self.depth_modulo, self.staleness_timeout = depth_modulo, staleness_timeout
        self.buckets = [KBucket(node_id.MIN, node_id.MAX, bucket_size)]

    def get_bucket_index(self, node_id: DHTID) -> int:
        """ Get the index of the bucket that the given node would fall into. """
        # TODO use binsearch aka from bisect import bisect
        for index, bucket in enumerate(self.buckets):
            if bucket.lower <= node_id < bucket.upper:
                return index
        raise ValueError(f"Failed to get bucket for node_id={node_id}, this should not be possible.")

    def try_add_node(self, node_id: DHTID, addr: Endpoint) -> bool:
        """
        Add a node to the routing table, split buckets if necessary.
        :returns: True if node was added (with or without splitting), False if node was rejected
        """
        bucket_index = self.get_bucket_index(node_id)
        bucket = self.buckets[bucket_index]

        if bucket.try_add_node(node_id, addr):
            return True  # this will succeed unless the bucket is full

        # Per section 4.2 of paper, split if the bucket has node's own id in its range
        # or if bucket depth is not congruent to 0 mod $b$
        if bucket.has_in_range(self.node_id) or bucket.depth % self.depth_modulo != 0:
            self.split_bucket(bucket_index)
            return self.try_add_node(node_id, addr)
        return False

    def __getitem__(self, node_id: DHTID) -> Endpoint:
        return self.buckets[self.get_bucket_index(node_id)][node_id]

    def __setitem__(self, node_id: DHTID, addr: Endpoint) -> NotImplementedError:
        raise NotImplementedError("KBucket doesn't support direct item assignment. Use KBucket.try_add_node instead")

    def __contains__(self, node_id: DHTID) -> bool:
        return node_id in self.buckets[self.get_bucket_index(node_id)]

    def __delitem__(self, node_id: DHTID):
        node_bucket = self.buckets[self.get_bucket_index(node_id)]
        del node_bucket[node_id]

    def split_bucket(self, index: int) -> None:
        """ Split bucket range in two equal parts and reassign nodes to the appropriate half """
        first, second = self.buckets[index].split()
        self.buckets[index] = first
        self.buckets.insert(index + 1, second)

    def get_nearest_neighbors(self, query_node_id: DHTID, k: int, exclude: Optional[DHTID] = None) -> List[DHTID]:
        """
        Find k nearest neighbors according to XOR distance
        :param query_node_id: find neighbors of this node
        :param k: find this many neighbors. If there aren't enough nodes in the table, returns all nodes
        :param exclude: if True, results will not contain query_node_id even if it is in table
        :returns: a list of node_ids for up to k neighbors sorted from nearest to farthest

        :note: this is a semi-exhaustive search of nodes that takes O(n * log k) time.
            One can implement a more efficient knn search using a binary skip-tree in some
            more elegant language such as c++ / cython / numba.
            Here's a sketch

            Preparation: construct a non-regular binary tree of depth (2 * DHTID.HASH_NBYTES)
             Each leaf corresponds to a binary DHTID with '0' for every left turn and '1' for right turn
             Each non-leaf corresponds to a certain prefix, e.g. 0010110???...???
             If there are no nodes under some prefix xxxY???..., the corresponding node xxx????...
             will only have one child.
            Add(node):
             Traverse down a tree, on i-th level go left if node_i == 0, right if node_i == 1
             If the corresponding node is missing, simply create it
            Search(query, k):
             Traverse the tree with a depth-first search, on i-th level go left if query_i == 0, else right
             If the corresponding node is missing, go the other way. Proceed until you found a leaf.
             This leaf is your nearest neighbor. Now add more neighbors by considering alternative paths
             bottom-up, i.e. if your nearest neighbor is 01011, first try 01010, then 0100x, then 011xx, ...

            This results in O(num_nodes * bit_length) complexity for add and search
            Better yet: use binary tree with skips for O(num_nodes * log(num_nodes))
        """
        all_nodes = chain(*map(KBucket.get_nodes, self.buckets))
        nearest_neighbors = heapq.nsmallest(k + int(exclude is not None), all_nodes, key=query_node_id.xor_distance)
        return [node_id for node_id in nearest_neighbors if (exclude is None or node_id != exclude)]

    # Protocol methods for DHTNode and KademliaProtocol

    def register_request_from(self, sender: Tuple[Hostname, Port], node_id: Optional[DHTID]) -> None:
        """ Update routing table on incoming request from host:port """
        raise NotImplementedError("TODO")

    def register_request_to(self, recepient: Tuple[Hostname, Port], node_id: Optional[DHTID],
                            *, responded: bool) -> None:
        """ Update routing table upon receiving response from a remote node """
        if node_id in self.nodes_to_ping:
            self.nodes_to_ping.remove(node_id)
        raise NotImplementedError("TODO")

    def get_nodes_to_refresh(self) -> List[Tuple[DHTID, Endpoint]]:
        """ return a list of nodes that should be queried """
        staleness_threshold = time.monotonic() - self.staleness_timeout
        stale_buckets = [bucket for bucket in self.buckets if bucket.last_updated < staleness_threshold]
        staleness_ids = [DHTID(random.randint(bucket.lower, bucket.upper)) for bucket in stale_buckets]


        raise NotImplementedError("TODO")

    def __repr__(self):
        bucket_info = "\n".join(repr(bucket) for bucket in self.buckets)
        return f"{self.__class__.__name__}(node_id={self.node_id}, bucket_size={self.bucket_size}," \
               f" modulo={self.depth_modulo}, timeout={self.staleness_timeout},\nbuckets=[\n{bucket_info})"


class KBucket:
    """
    A bucket containing up to :size: of DHTIDs in [lower, upper) semi-interval.
    Maps DHT node ids to their endpoints (hostname, addr)
    """
    def __init__(self, lower: int, upper: int, size: int, depth: int = 0):
        assert upper - lower == 2 ** (DHTID.HASH_NBYTES * 8 - depth)
        self.lower, self.upper, self.size, self.depth = lower, upper, size, depth
        self.nodes_to_addr: Dict[DHTID, Endpoint] = {}
        self.replacement_nodes: Dict[DHTID, Endpoint] = {}
        self.last_updated = time.monotonic()

    def register_update(self) -> None:
        self.last_updated = time.monotonic()

    def has_in_range(self, node_id: DHTID):
        """ Check if node_id is between this bucket's lower and upper bounds """
        return self.lower <= node_id < self.upper

    def try_add_node(self, node_id: DHTID, addr: Endpoint) -> bool:
        """
        Add node to KBucket, return True if successful, False if the bucket is full.
        If the bucket is full, keep track of node in a replacement list, per section 4.1 of the paper.
        :param node_id: dht node identifier
        :param addr: a pair of (hostname, port) associated with that node id
        """
        if node_id in self.nodes_to_addr:
            del self.nodes_to_addr[node_id]
            self.nodes_to_addr[node_id] = addr
        elif len(self) < self.size:
            self.nodes_to_addr[node_id] = addr
        else:
            if node_id in self.replacement_nodes:
                del self.replacement_nodes[node_id]
            self.replacement_nodes[node_id] = addr
            return False
        return True

    def get_nodes(self) -> List[DHTID]:
        return list(self.nodes_to_addr.keys())

    def __getitem__(self, node_id: DHTID) -> Endpoint:
        return self.nodes_to_addr[node_id] if node_id in self.nodes_to_addr else self.replacement_nodes[node_id]

    def __delitem__(self, node_id: DHTID):
        if not (node_id in self.nodes_to_addr or node_id in self.replacement_nodes):
            raise KeyError(f"KBucket does not contain node id={node_id}.")

        if node_id in self.replacement_nodes:
            del self.replacement_nodes[node_id]

        if node_id in self.nodes_to_addr:
            del self.nodes_to_addr[node_id]

            if self.replacement_nodes:
                newnode_id, newnode = self.replacement_nodes.popitem()
                self.nodes_to_addr[newnode_id] = newnode

    def __len__(self):
        return len(self.nodes_to_addr)

    def split(self) -> Tuple[KBucket, KBucket]:
        """ Split bucket over midpoint, rounded down, assign nodes to according to their id """
        midpoint = (self.lower + self.upper) // 2
        assert self.lower < midpoint < self.upper, f"Bucket to small to be split: [{self.lower}: {self.upper})"
        left = KBucket(self.lower, midpoint, self.size, depth=self.depth + 1)
        right =KBucket(midpoint, self.upper, self.size, depth=self.depth + 1)
        for node_id, addr in chain(self.nodes_to_addr.items(), self.replacement_nodes.items()):
            bucket = left if int(node_id) <= midpoint else right
            bucket.try_add_node(node_id, addr)
        return left, right

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self.nodes_to_addr)} nodes" \
               f" with {len(self.replacement_nodes)} replacements, depth={self.depth}, max size={self.size}" \
               f" lower={hex(self.lower)}, upper={hex(self.upper)})"


class DHTID(int):
    HASH_FUNC = hashlib.sha1
    HASH_NBYTES = 20  # SHA1 produces a 20-byte (aka 160bit) number
    RANGE = MIN, MAX = 0, 2 ** (HASH_NBYTES * 8)  # inclusive min, exclusive max

    def __new__(cls, value: int):
        assert cls.MIN <= value < cls.MAX, f"DHTID must be in [{cls.MIN}, {cls.MAX}) but got {value}"
        return super().__new__(cls, value)

    def to_bytes(self, length=HASH_NBYTES, byteorder='big', *, signed=False) -> bytes:
        return super().to_bytes(length, byteorder, signed=signed)

    @classmethod
    def generate(cls, seed: Optional[int] = None, nbits: int = 255):
        """
        Generates random uid based on SHA1
        """
        randbytes = (seed or random.getrandbits(nbits)).to_bytes(nbits, byteorder='big')
        raw_uid = hashlib.sha1(randbytes).digest()
        return cls(int(raw_uid.hex(), 16))

    def xor_distance(self, other: DHTID) -> int:
        """ Return a number which binary representation equals bitwise xor between the two DHTIDs """
        return int(self) ^ int(other)

    @classmethod
    def longest_common_prefix_length(cls, *ids: DHTID) -> int:
        ids_bits = [bin(uid)[2:].rjust(8 * cls.HASH_NBYTES, '0') for uid in ids]
        return len(os.path.commonprefix(ids_bits))

    def __repr__(self):
        return f"{self.__class__.__name__}({hex(self)})"

DHTValue, DHTExpirationTime = Any, float  # flavour types