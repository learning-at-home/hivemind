import heapq
import operator
import random
from itertools import chain

from hivemind.dht.routing import DHTID
from hivemind.dht.routing import RoutingTable
from hivemind.utils.serializer import PickleSerializer


def test_ids_basic():
    # basic functionality tests
    for i in range(100):
        id1, id2 = DHTID.generate(), DHTID.generate()
        assert DHTID.MIN <= id1 < DHTID.MAX and DHTID.MIN <= id2 <= DHTID.MAX
        assert DHTID.xor_distance(id1, id1) == DHTID.xor_distance(id2, id2) == 0
        assert DHTID.xor_distance(id1, id2) > 0 or (id1 == id2)
        assert (
            len(PickleSerializer.dumps(id1)) - len(PickleSerializer.dumps(int(id1)))
            < 40
        )
        assert (
            DHTID.from_bytes(bytes(id1)) == id1
            and DHTID.from_bytes(id2.to_bytes()) == id2
        )


def test_ids_depth():
    for i in range(100):
        ids = [random.randint(0, 4096) for i in range(random.randint(1, 256))]
        ours = DHTID.longest_common_prefix_length(*map(DHTID, ids))

        ids_bitstr = [
            "".join(bin(bite)[2:].rjust(8, "0") for bite in uid.to_bytes(20, "big"))
            for uid in ids
        ]
        reference = len(shared_prefix(*ids_bitstr))
        assert reference == ours, f"ours {ours} != reference {reference}, ids: {ids}"


def test_routing_table_basic():
    node_id = DHTID.generate()
    routing_table = RoutingTable(
        node_id, bucket_size=20, depth_modulo=5, staleness_timeout=300
    )

    for phony_neighbor_port in random.sample(range(10000), 100):
        phony_id = DHTID.generate()
        routing_table.try_add_node(phony_id, ("localhost", phony_neighbor_port))
        assert routing_table[phony_id] == ("localhost", phony_neighbor_port)

    assert (
        routing_table.buckets[0].lower == DHTID.MIN
        and routing_table.buckets[-1].upper == DHTID.MAX
    )
    for bucket in routing_table.buckets:
        assert (
            len(bucket.replacement_nodes) == 0
        ), "There should be no replacement nodes in a table with 100 entries"
    assert 3 <= len(routing_table.buckets) <= 10, len(routing_table.buckets)


def test_routing_table_parameters():
    for (bucket_size, modulo, min_nbuckets, max_nbuckets) in [
        (20, 5, 45, 65),
        (50, 5, 35, 45),
        (20, 10, 650, 800),
        (20, 1, 7, 15),
    ]:
        node_id = DHTID.generate()
        routing_table = RoutingTable(
            node_id, bucket_size=bucket_size, depth_modulo=modulo, staleness_timeout=300
        )
        for phony_neighbor_port in random.sample(range(1_000_000), 10_000):
            routing_table.try_add_node(
                DHTID.generate(), ("localhost", phony_neighbor_port)
            )
        for bucket in routing_table.buckets:
            assert (
                len(bucket.replacement_nodes) == 0
                or len(bucket.nodes_to_addr) <= bucket.size
            )
        assert (
            min_nbuckets <= len(routing_table.buckets) <= max_nbuckets
        ), f"Unexpected number of buckets: {min_nbuckets} <= {len(routing_table.buckets)} <= {max_nbuckets}"


def test_routing_table_search():
    node_id = DHTID.generate()
    routing_table = RoutingTable(
        node_id, bucket_size=20, depth_modulo=5, staleness_timeout=300
    )
    num_added = 0
    for phony_neighbor_port in random.sample(range(1_000_000), 10_000):
        num_added += routing_table.try_add_node(
            DHTID.generate(), ("localhost", phony_neighbor_port)
        )
    num_replacements = sum(
        len(bucket.replacement_nodes) for bucket in routing_table.buckets
    )

    all_active_neighbors = list(
        chain(*(bucket.nodes_to_addr.keys() for bucket in routing_table.buckets))
    )
    assert 800 <= len(all_active_neighbors) <= 1100
    assert len(all_active_neighbors) == num_added
    assert num_added + num_replacements == 10_000

    # random queries
    for i in range(500):
        k = random.randint(1, 100)
        query_id = DHTID.generate()
        exclude = query_id if random.random() < 0.5 else None
        our_knn = routing_table.get_nearest_neighbors(query_id, k=k, exclude=exclude)
        reference_knn = heapq.nsmallest(
            k, all_active_neighbors, key=query_id.xor_distance
        )
        assert all(our == ref for our, ref in zip(our_knn, reference_knn))

    # queries from table
    for i in range(500):
        k = random.randint(1, 100)
        query_id = random.choice(all_active_neighbors)
        our_knn = routing_table.get_nearest_neighbors(query_id, k=k, exclude=query_id)
        reference_knn = heapq.nsmallest(
            k,
            all_active_neighbors,
            key=lambda uid: query_id.xor_distance(uid)
            if uid != query_id
            else float("inf"),
        )

        assert query_id not in our_knn
        assert all(
            query_id.xor_distance(our) == query_id.xor_distance(ref)
            for our, ref in zip(our_knn, reference_knn)
        )
        assert (
            routing_table.get_nearest_neighbors(query_id, k=k, exclude=None)[0]
            == query_id
        )


def shared_prefix(*strings: str):
    for i in range(min(map(len, strings))):
        if len(set(map(operator.itemgetter(i), strings))) != 1:
            return strings[0][:i]
    return min(strings, key=len)
