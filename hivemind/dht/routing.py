""" Utlity data structures to represent DHT nodes (peers), data keys, and routing tables. """
from __future__ import annotations

import hashlib
import heapq
import os
import random
from collections.abc import Iterable
from itertools import chain
from typing import Tuple, Optional, List, Dict, Set, Union, Any, Sequence
from hivemind.utils import Endpoint, PickleSerializer, get_dht_time, DHTExpiration

DHTKey, Subkey, DHTValue, BinaryDHTID, BinaryDHTValue, = Any, Any, Any, bytes, bytes


class RoutingTable:
    """
    A data structure that contains DHT peers bucketed according to their distance to node_id.
    Follows Kademlia routing table as described in https://pdos.csail.mit.edu/~petar/papers/maymounkov-kademlia-lncs.pdf

    :param node_id: node id used to measure distance
    :param bucket_size: parameter $k$ from Kademlia paper Section 2.2
    :param depth_modulo: parameter $b$ from Kademlia paper Section 2.2.
    :note: you can find a more detailed description of parameters in DHTNode, see node.py
    """

    def __init__(self, node_id: DHTID, bucket_size: int, depth_modulo: int):
        self.node_id, self.bucket_size, self.depth_modulo = node_id, bucket_size, depth_modulo
        self.buckets = [KBucket(node_id.MIN, node_id.MAX, bucket_size)]
        self.endpoint_to_uid: Dict[Endpoint, DHTID] = dict()  # all nodes currently in buckets, including replacements
        self.uid_to_endpoint: Dict[DHTID, Endpoint] = dict()  # all nodes currently in buckets, including replacements

    def get_bucket_index(self, node_id: DHTID) -> int:
        """ Get the index of the bucket that the given node would fall into. """
        lower_index, upper_index = 0, len(self.buckets)
        while upper_index - lower_index > 1:
            pivot_index = (lower_index + upper_index + 1) // 2
            if node_id >= self.buckets[pivot_index].lower:
                lower_index = pivot_index
            else:  # node_id < self.buckets[pivot_index].lower
                upper_index = pivot_index
        assert upper_index - lower_index == 1
        return lower_index

    def add_or_update_node(self, node_id: DHTID, endpoint: Endpoint) -> Optional[Tuple[DHTID, Endpoint]]:
        """
        Update routing table after an incoming request from :endpoint: or outgoing request to :endpoint:

        :returns: If we cannot add node_id to the routing table, return the least-recently-updated node (Section 2.2)
        :note: DHTProtocol calls this method for every incoming and outgoing request if there was a response.
          If this method returned a node to be ping-ed, the protocol will ping it to check and either move it to
          the start of the table or remove that node and replace it with
        """
        bucket_index = self.get_bucket_index(node_id)
        bucket = self.buckets[bucket_index]
        store_success = bucket.add_or_update_node(node_id, endpoint)

        if node_id in bucket.nodes_to_endpoint or node_id in bucket.replacement_nodes:
            # if we added node to bucket or as a replacement, throw it into lookup dicts as well
            self.uid_to_endpoint[node_id] = endpoint
            self.endpoint_to_uid[endpoint] = node_id

        if not store_success:
            # Per section 4.2 of paper, split if the bucket has node's own id in its range
            # or if bucket depth is not congruent to 0 mod $b$
            if bucket.has_in_range(self.node_id) or bucket.depth % self.depth_modulo != 0:
                self.split_bucket(bucket_index)
                return self.add_or_update_node(node_id, endpoint)

            # The bucket is full and won't split further. Return a node to ping (see this method's docstring)
            return bucket.request_ping_node()

    def split_bucket(self, index: int) -> None:
        """ Split bucket range in two equal parts and reassign nodes to the appropriate half """
        first, second = self.buckets[index].split()
        self.buckets[index] = first
        self.buckets.insert(index + 1, second)

    def get(self, *, node_id: Optional[DHTID] = None, endpoint: Optional[Endpoint] = None, default=None):
        """ Find endpoint for a given DHTID or vice versa """
        assert (node_id is None) != (endpoint is None), "Please specify either node_id or endpoint, but not both"
        if node_id is not None:
            return self.uid_to_endpoint.get(node_id, default)
        else:
            return self.endpoint_to_uid.get(endpoint, default)

    def __getitem__(self, item: Union[DHTID, Endpoint]) -> Union[Endpoint, DHTID]:
        """ Find endpoint for a given DHTID or vice versa """
        return self.uid_to_endpoint[item] if isinstance(item, DHTID) else self.endpoint_to_uid[item]

    def __setitem__(self, node_id: DHTID, endpoint: Endpoint) -> NotImplementedError:
        raise NotImplementedError("RoutingTable doesn't support direct item assignment. Use table.try_add_node instead")

    def __contains__(self, item: Union[DHTID, Endpoint]) -> bool:
        return (item in self.uid_to_endpoint) if isinstance(item, DHTID) else (item in self.endpoint_to_uid)

    def __delitem__(self, node_id: DHTID):
        del self.buckets[self.get_bucket_index(node_id)][node_id]
        node_endpoint = self.uid_to_endpoint.pop(node_id)
        if self.endpoint_to_uid.get(node_endpoint) == node_id:
            del self.endpoint_to_uid[node_endpoint]

    def get_nearest_neighbors(
            self, query_id: DHTID, k: int, exclude: Optional[DHTID] = None) -> List[Tuple[DHTID, Endpoint]]:
        """
        Find k nearest neighbors from routing table according to XOR distance, does NOT include self.node_id

        :param query_id: find neighbors of this node
        :param k: find this many neighbors. If there aren't enough nodes in the table, returns all nodes
        :param exclude: if True, results will not contain query_node_id even if it is in table
        :return: a list of tuples (node_id, endpoint) for up to k neighbors sorted from nearest to farthest
        """
        # algorithm: first add up all buckets that can contain one of k nearest nodes, then heap-sort to find best
        candidates: List[Tuple[int, DHTID, Endpoint]] = []  # min-heap based on xor distance to query_id

        # step 1: add current bucket to the candidates heap
        nearest_index = self.get_bucket_index(query_id)
        nearest_bucket = self.buckets[nearest_index]
        for node_id, endpoint in nearest_bucket.nodes_to_endpoint.items():
            heapq.heappush(candidates, (query_id.xor_distance(node_id), node_id, endpoint))

        # step 2: add adjacent buckets by ascending code tree one level at a time until you have enough nodes
        left_index, right_index = nearest_index, nearest_index + 1  # bucket indices considered, [left, right)
        current_lower, current_upper, current_depth = nearest_bucket.lower, nearest_bucket.upper, nearest_bucket.depth

        while current_depth > 0 and len(candidates) < k + int(exclude is not None):
            split_direction = current_lower // 2 ** (DHTID.HASH_NBYTES * 8 - current_depth) % 2
            # ^-- current leaf direction from pre-leaf node, 0 = left, 1 = right
            current_depth -= 1  # traverse one level closer to the root and add all child nodes to the candidates heap

            if split_direction == 0:  # leaf was split on the left, merge its right peer(s)
                current_upper += current_upper - current_lower
                while right_index < len(self.buckets) and self.buckets[right_index].upper <= current_upper:
                    for node_id, endpoint in self.buckets[right_index].nodes_to_endpoint.items():
                        heapq.heappush(candidates, (query_id.xor_distance(node_id), node_id, endpoint))
                    right_index += 1  # note: we may need to add more than one bucket if they are on a lower depth level
                assert self.buckets[right_index - 1].upper == current_upper

            else:  # split_direction == 1, leaf was split on the right, merge its left peer(s)
                current_lower -= current_upper - current_lower
                while left_index > 0 and self.buckets[left_index - 1].lower >= current_lower:
                    left_index -= 1  # note: we may need to add more than one bucket if they are on a lower depth level
                    for node_id, endpoint in self.buckets[left_index].nodes_to_endpoint.items():
                        heapq.heappush(candidates, (query_id.xor_distance(node_id), node_id, endpoint))
                assert self.buckets[left_index].lower == current_lower

        # step 3: select k nearest vertices from candidates heap
        heap_top: List[Tuple[int, DHTID, Endpoint]] = heapq.nsmallest(k + int(exclude is not None), candidates)
        return [(node, endpoint) for _, node, endpoint in heap_top if node != exclude][:k]

    def __repr__(self):
        bucket_info = "\n".join(repr(bucket) for bucket in self.buckets)
        return f"{self.__class__.__name__}(node_id={self.node_id}, bucket_size={self.bucket_size}," \
               f" modulo={self.depth_modulo},\nbuckets=[\n{bucket_info})"


class KBucket:
    """
    A bucket containing up to :size: of DHTIDs in [lower, upper) semi-interval.
    Maps DHT node ids to their endpoints
    """

    def __init__(self, lower: int, upper: int, size: int, depth: int = 0):
        assert upper - lower == 2 ** (DHTID.HASH_NBYTES * 8 - depth)
        self.lower, self.upper, self.size, self.depth = lower, upper, size, depth
        self.nodes_to_endpoint: Dict[DHTID, Endpoint] = {}
        self.replacement_nodes: Dict[DHTID, Endpoint] = {}
        self.nodes_requested_for_ping: Set[DHTID] = set()
        self.last_updated = get_dht_time()

    def has_in_range(self, node_id: DHTID):
        """ Check if node_id is between this bucket's lower and upper bounds """
        return self.lower <= node_id < self.upper

    def add_or_update_node(self, node_id: DHTID, endpoint: Endpoint) -> bool:
        """
        Add node to KBucket or update existing node, return True if successful, False if the bucket is full.
        If the bucket is full, keep track of node in a replacement list, per section 4.1 of the paper.

        :param node_id: dht node identifier that should be added or moved to the front of bucket
        :param endpoint: network address associated with that node id
        :note: this function has a side-effect of resetting KBucket.last_updated time
        """
        if node_id in self.nodes_requested_for_ping:
            self.nodes_requested_for_ping.remove(node_id)
        self.last_updated = get_dht_time()
        if node_id in self.nodes_to_endpoint:
            del self.nodes_to_endpoint[node_id]
            self.nodes_to_endpoint[node_id] = endpoint
        elif len(self.nodes_to_endpoint) < self.size:
            self.nodes_to_endpoint[node_id] = endpoint
        else:
            if node_id in self.replacement_nodes:
                del self.replacement_nodes[node_id]
            self.replacement_nodes[node_id] = endpoint
            return False
        return True

    def request_ping_node(self) -> Optional[Tuple[DHTID, Endpoint]]:
        """ :returns: least-recently updated node that isn't already being pinged right now -- if such node exists """
        for uid, endpoint in self.nodes_to_endpoint.items():
            if uid not in self.nodes_requested_for_ping:
                self.nodes_requested_for_ping.add(uid)
                return uid, endpoint

    def __getitem__(self, node_id: DHTID) -> Endpoint:
        return self.nodes_to_endpoint[node_id] if node_id in self.nodes_to_endpoint else self.replacement_nodes[node_id]

    def __delitem__(self, node_id: DHTID):
        if not (node_id in self.nodes_to_endpoint or node_id in self.replacement_nodes):
            raise KeyError(f"KBucket does not contain node id={node_id}.")

        if node_id in self.replacement_nodes:
            del self.replacement_nodes[node_id]

        if node_id in self.nodes_to_endpoint:
            del self.nodes_to_endpoint[node_id]

            if self.replacement_nodes:
                newnode_id, newnode = self.replacement_nodes.popitem()
                self.nodes_to_endpoint[newnode_id] = newnode

    def split(self) -> Tuple[KBucket, KBucket]:
        """ Split bucket over midpoint, rounded down, assign nodes to according to their id """
        midpoint = (self.lower + self.upper) // 2
        assert self.lower < midpoint < self.upper, f"Bucket to small to be split: [{self.lower}: {self.upper})"
        left = KBucket(self.lower, midpoint, self.size, depth=self.depth + 1)
        right = KBucket(midpoint, self.upper, self.size, depth=self.depth + 1)
        for node_id, endpoint in chain(self.nodes_to_endpoint.items(), self.replacement_nodes.items()):
            bucket = left if int(node_id) <= midpoint else right
            bucket.add_or_update_node(node_id, endpoint)
        return left, right

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self.nodes_to_endpoint)} nodes" \
               f" with {len(self.replacement_nodes)} replacements, depth={self.depth}, max size={self.size}" \
               f" lower={hex(self.lower)}, upper={hex(self.upper)})"


class DHTID(int):
    HASH_FUNC = hashlib.sha1
    HASH_NBYTES = 20  # SHA1 produces a 20-byte (aka 160bit) number
    RANGE = MIN, MAX = 0, 2 ** (HASH_NBYTES * 8)  # inclusive min, exclusive max

    def __new__(cls, value: int):
        assert cls.MIN <= value < cls.MAX, f"DHTID must be in [{cls.MIN}, {cls.MAX}) but got {value}"
        return super().__new__(cls, value)

    @classmethod
    def generate(cls, source: Optional[Any] = None, nbits: int = 255):
        """
        Generates random uid based on SHA1

        :param source: if provided, converts this value to bytes and uses it as input for hashing function;
            by default, generates a random dhtid from :nbits: random bits
        """
        source = random.getrandbits(nbits).to_bytes(nbits, byteorder='big') if source is None else source
        source = PickleSerializer.dumps(source) if not isinstance(source, bytes) else source
        raw_uid = cls.HASH_FUNC(source).digest()
        return cls(int(raw_uid.hex(), 16))

    def xor_distance(self, other: Union[DHTID, Sequence[DHTID]]) -> Union[int, List[int]]:
        """
        :param other: one or multiple DHTIDs. If given multiple DHTIDs as other, this function
         will compute distance from self to each of DHTIDs in other.
        :return: a number or a list of numbers whose binary representations equal bitwise xor between DHTIDs.
        """
        if isinstance(other, Iterable):
            return list(map(self.xor_distance, other))
        return int(self) ^ int(other)

    @classmethod
    def longest_common_prefix_length(cls, *ids: DHTID) -> int:
        ids_bits = [bin(uid)[2:].rjust(8 * cls.HASH_NBYTES, '0') for uid in ids]
        return len(os.path.commonprefix(ids_bits))

    def to_bytes(self, length=HASH_NBYTES, byteorder='big', *, signed=False) -> bytes:
        """ A standard way to serialize DHTID into bytes """
        return super().to_bytes(length, byteorder, signed=signed)

    @classmethod
    def from_bytes(cls, raw: bytes, byteorder='big', *, signed=False) -> DHTID:
        """ reverse of to_bytes """
        return DHTID(super().from_bytes(raw, byteorder=byteorder, signed=signed))

    def __repr__(self):
        return f"{self.__class__.__name__}({hex(self)})"

    def __bytes__(self):
        return self.to_bytes()
