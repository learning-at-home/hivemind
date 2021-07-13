import asyncio
import heapq
import multiprocessing as mp
import random
import signal
from itertools import product
from typing import List, Sequence, Tuple

import numpy as np
import pytest
from multiaddr import Multiaddr

import hivemind
from hivemind import get_dht_time
from hivemind.dht.node import DHTID, DHTNode
from hivemind.dht.protocol import DHTProtocol
from hivemind.dht.storage import DictionaryDHTValue
from hivemind.p2p import P2P, PeerID
from hivemind.utils.logging import get_logger
from test_utils.dht_swarms import launch_swarm_in_separate_processes, launch_star_shaped_swarm


logger = get_logger(__name__)


def maddrs_to_peer_ids(maddrs: List[Multiaddr]) -> List[PeerID]:
    return list({PeerID.from_base58(maddr["p2p"]) for maddr in maddrs})


def run_protocol_listener(
    dhtid: DHTID, maddr_conn: mp.connection.Connection, initial_peers: Sequence[Multiaddr]
) -> None:
    loop = asyncio.get_event_loop()

    p2p = loop.run_until_complete(P2P.create(initial_peers=initial_peers))
    visible_maddrs = loop.run_until_complete(p2p.get_visible_maddrs())

    protocol = loop.run_until_complete(
        DHTProtocol.create(p2p, dhtid, bucket_size=20, depth_modulo=5, num_replicas=3, wait_timeout=5)
    )

    logger.info(f"Started peer id={protocol.node_id} visible_maddrs={visible_maddrs}")

    for peer_id in maddrs_to_peer_ids(initial_peers):
        loop.run_until_complete(protocol.call_ping(peer_id))

    maddr_conn.send((p2p.id, visible_maddrs))

    async def shutdown():
        await p2p.shutdown()
        logger.info(f"Finished peer id={protocol.node_id} maddrs={visible_maddrs}")
        loop.stop()

    loop.add_signal_handler(signal.SIGTERM, lambda: loop.create_task(shutdown()))
    loop.run_forever()


def launch_protocol_listener(
    initial_peers: Sequence[Multiaddr] = (),
) -> Tuple[DHTID, mp.Process, PeerID, List[Multiaddr]]:
    remote_conn, local_conn = mp.Pipe()
    dht_id = DHTID.generate()
    process = mp.Process(target=run_protocol_listener, args=(dht_id, remote_conn, initial_peers), daemon=True)
    process.start()
    peer_id, visible_maddrs = local_conn.recv()

    return dht_id, process, peer_id, visible_maddrs


# note: we run network-related tests in a separate process to re-initialize all global states from scratch
# this helps us avoid undesirable gRPC side-effects (e.g. segfaults) when running multiple tests in sequence


@pytest.mark.forked
def test_dht_protocol():
    peer1_node_id, peer1_proc, peer1_id, peer1_maddrs = launch_protocol_listener()
    peer2_node_id, peer2_proc, peer2_id, _ = launch_protocol_listener(initial_peers=peer1_maddrs)

    loop = asyncio.get_event_loop()
    for listen in [False, True]:  # note: order matters, this test assumes that first run uses listen=False
        p2p = loop.run_until_complete(P2P.create(initial_peers=peer1_maddrs))
        protocol = loop.run_until_complete(
            DHTProtocol.create(
                p2p, DHTID.generate(), bucket_size=20, depth_modulo=5, wait_timeout=5, num_replicas=3, listen=listen
            )
        )
        logger.info(f"Self id={protocol.node_id}")

        assert loop.run_until_complete(protocol.call_ping(peer1_id)) == peer1_node_id

        key, value, expiration = DHTID.generate(), [random.random(), {"ololo": "pyshpysh"}], get_dht_time() + 1e3
        store_ok = loop.run_until_complete(
            protocol.call_store(peer1_id, [key], [hivemind.MSGPackSerializer.dumps(value)], expiration)
        )
        assert all(store_ok), "DHT rejected a trivial store"

        # peer 1 must know about peer 2
        (recv_value_bytes, recv_expiration), nodes_found = loop.run_until_complete(
            protocol.call_find(peer1_id, [key])
        )[key]
        recv_value = hivemind.MSGPackSerializer.loads(recv_value_bytes)
        (recv_id, recv_peer_id) = next(iter(nodes_found.items()))
        assert (
            recv_id == peer2_node_id and recv_peer_id == peer2_id
        ), f"expected id={peer2_node_id}, peer={peer2_id} but got {recv_id}, {recv_peer_id}"

        assert recv_value == value and recv_expiration == expiration, (
            f"call_find_value expected {value} (expires by {expiration}) "
            f"but got {recv_value} (expires by {recv_expiration})"
        )

        # peer 2 must know about peer 1, but not have a *random* nonexistent value
        dummy_key = DHTID.generate()
        empty_item, nodes_found_2 = loop.run_until_complete(protocol.call_find(peer2_id, [dummy_key]))[dummy_key]
        assert empty_item is None, "Non-existent keys shouldn't have values"
        (recv_id, recv_peer_id) = next(iter(nodes_found_2.items()))
        assert (
            recv_id == peer1_node_id and recv_peer_id == peer1_id
        ), f"expected id={peer1_node_id}, peer={peer1_id} but got {recv_id}, {recv_peer_id}"

        # cause a non-response by querying a nonexistent peer
        assert loop.run_until_complete(protocol.call_find(PeerID.from_base58("fakeid"), [key])) is None

        # store/get a dictionary with sub-keys
        nested_key, subkey1, subkey2 = DHTID.generate(), "foo", "bar"
        value1, value2 = [random.random(), {"ololo": "pyshpysh"}], "abacaba"
        assert loop.run_until_complete(
            protocol.call_store(
                peer1_id,
                keys=[nested_key],
                values=[hivemind.MSGPackSerializer.dumps(value1)],
                expiration_time=[expiration],
                subkeys=[subkey1],
            )
        )
        assert loop.run_until_complete(
            protocol.call_store(
                peer1_id,
                keys=[nested_key],
                values=[hivemind.MSGPackSerializer.dumps(value2)],
                expiration_time=[expiration + 5],
                subkeys=[subkey2],
            )
        )
        (recv_dict, recv_expiration), nodes_found = loop.run_until_complete(
            protocol.call_find(peer1_id, [nested_key])
        )[nested_key]
        assert isinstance(recv_dict, DictionaryDHTValue)
        assert len(recv_dict.data) == 2 and recv_expiration == expiration + 5
        assert recv_dict.data[subkey1] == (protocol.serializer.dumps(value1), expiration)
        assert recv_dict.data[subkey2] == (protocol.serializer.dumps(value2), expiration + 5)

        if listen:
            loop.run_until_complete(p2p.shutdown())

    peer1_proc.terminate()
    peer2_proc.terminate()


@pytest.mark.forked
def test_empty_table():
    """Test RPC methods with empty routing table"""
    peer_id, peer_proc, peer_peer_id, peer_maddrs = launch_protocol_listener()

    loop = asyncio.get_event_loop()
    p2p = loop.run_until_complete(P2P.create(initial_peers=peer_maddrs))
    protocol = loop.run_until_complete(
        DHTProtocol.create(
            p2p, DHTID.generate(), bucket_size=20, depth_modulo=5, wait_timeout=5, num_replicas=3, listen=False
        )
    )

    key, value, expiration = DHTID.generate(), [random.random(), {"ololo": "pyshpysh"}], get_dht_time() + 1e3

    empty_item, nodes_found = loop.run_until_complete(protocol.call_find(peer_peer_id, [key]))[key]
    assert empty_item is None and len(nodes_found) == 0
    assert all(
        loop.run_until_complete(
            protocol.call_store(peer_peer_id, [key], [hivemind.MSGPackSerializer.dumps(value)], expiration)
        )
    ), "peer rejected store"

    (recv_value_bytes, recv_expiration), nodes_found = loop.run_until_complete(
        protocol.call_find(peer_peer_id, [key])
    )[key]
    recv_value = hivemind.MSGPackSerializer.loads(recv_value_bytes)
    assert len(nodes_found) == 0
    assert recv_value == value and recv_expiration == expiration

    assert loop.run_until_complete(protocol.call_ping(peer_peer_id)) == peer_id
    assert loop.run_until_complete(protocol.call_ping(PeerID.from_base58("fakeid"))) is None
    peer_proc.terminate()


@pytest.mark.forked
def test_dht_node():
    # step A: create a swarm of 50 dht nodes in separate processes
    #         (first 5 created sequentially, others created in parallel)
    processes, dht, swarm_maddrs = launch_swarm_in_separate_processes(n_peers=50, n_sequential_peers=5)

    # step B: run 51-st node in this process
    loop = asyncio.get_event_loop()
    initial_peers = random.choice(swarm_maddrs)
    me = loop.run_until_complete(
        DHTNode.create(initial_peers=initial_peers, parallel_rpc=10, cache_refresh_before_expiry=False)
    )

    # test 1: find self
    nearest = loop.run_until_complete(me.find_nearest_nodes([me.node_id], k_nearest=1))[me.node_id]
    assert len(nearest) == 1 and nearest[me.node_id] == me.peer_id

    # test 2: find others
    for _ in range(10):
        ref_peer_id, query_id = random.choice(list(dht.items()))
        nearest = loop.run_until_complete(me.find_nearest_nodes([query_id], k_nearest=1))[query_id]
        assert len(nearest) == 1
        found_node_id, found_peer_id = next(iter(nearest.items()))
        assert found_node_id == query_id and found_peer_id == ref_peer_id

    # test 3: find neighbors to random nodes
    accuracy_numerator = accuracy_denominator = 0  # top-1 nearest neighbor accuracy
    jaccard_numerator = jaccard_denominator = 0  # jaccard similarity aka intersection over union
    all_node_ids = list(dht.values())

    for _ in range(10):
        query_id = DHTID.generate()
        k_nearest = random.randint(1, 10)
        exclude_self = random.random() > 0.5
        nearest = loop.run_until_complete(
            me.find_nearest_nodes([query_id], k_nearest=k_nearest, exclude_self=exclude_self)
        )[query_id]
        nearest_nodes = list(nearest)  # keys from ordered dict

        assert len(nearest_nodes) == k_nearest, "beam search must return exactly k_nearest results"
        assert me.node_id not in nearest_nodes or not exclude_self, "if exclude, results shouldn't contain self"
        assert np.all(np.diff(query_id.xor_distance(nearest_nodes)) >= 0), "results must be sorted by distance"

        ref_nearest = heapq.nsmallest(k_nearest + 1, all_node_ids, key=query_id.xor_distance)
        if exclude_self and me.node_id in ref_nearest:
            ref_nearest.remove(me.node_id)
        if len(ref_nearest) > k_nearest:
            ref_nearest.pop()

        accuracy_numerator += nearest_nodes[0] == ref_nearest[0]
        accuracy_denominator += 1

        jaccard_numerator += len(set.intersection(set(nearest_nodes), set(ref_nearest)))
        jaccard_denominator += k_nearest

    accuracy = accuracy_numerator / accuracy_denominator
    logger.debug(f"Top-1 accuracy: {accuracy}")  # should be 98-100%
    jaccard_index = jaccard_numerator / jaccard_denominator
    logger.debug(f"Jaccard index (intersection over union): {jaccard_index}")  # should be 95-100%
    assert accuracy >= 0.9, f"Top-1 accuracy only {accuracy} ({accuracy_numerator} / {accuracy_denominator})"
    assert jaccard_index >= 0.9, f"Jaccard index only {accuracy} ({accuracy_numerator} / {accuracy_denominator})"

    # test 4: find all nodes
    dummy = DHTID.generate()
    nearest = loop.run_until_complete(me.find_nearest_nodes([dummy], k_nearest=len(dht) + 100))[dummy]
    assert len(nearest) == len(dht) + 1
    assert len(set.difference(set(nearest.keys()), set(all_node_ids) | {me.node_id})) == 0

    # test 5: node without peers
    detached_node = loop.run_until_complete(DHTNode.create())
    nearest = loop.run_until_complete(detached_node.find_nearest_nodes([dummy]))[dummy]
    assert len(nearest) == 1 and nearest[detached_node.node_id] == detached_node.peer_id
    nearest = loop.run_until_complete(detached_node.find_nearest_nodes([dummy], exclude_self=True))[dummy]
    assert len(nearest) == 0

    # test 6: store and get value
    true_time = get_dht_time() + 1200
    assert loop.run_until_complete(me.store("mykey", ["Value", 10], true_time))

    initial_peers = random.choice(swarm_maddrs)
    that_guy = loop.run_until_complete(
        DHTNode.create(
            initial_peers=initial_peers, parallel_rpc=10, cache_refresh_before_expiry=False, cache_locally=False
        )
    )

    for node in [me, that_guy]:
        val, expiration_time = loop.run_until_complete(node.get("mykey"))
        assert val == ["Value", 10], "Wrong value"
        assert expiration_time == true_time, f"Wrong time"

    assert loop.run_until_complete(detached_node.get("mykey")) is None

    # test 7: bulk store and bulk get
    keys = "foo", "bar", "baz", "zzz"
    values = 3, 2, "batman", [1, 2, 3]
    store_ok = loop.run_until_complete(me.store_many(keys, values, expiration_time=get_dht_time() + 999))
    assert all(store_ok.values()), "failed to store one or more keys"
    response = loop.run_until_complete(me.get_many(keys[::-1]))
    for key, value in zip(keys, values):
        assert key in response and response[key][0] == value

    # test 8: store dictionaries as values (with sub-keys)
    upper_key, subkey1, subkey2, subkey3 = "ololo", "k1", "k2", "k3"
    now = get_dht_time()
    assert loop.run_until_complete(me.store(upper_key, subkey=subkey1, value=123, expiration_time=now + 10))
    assert loop.run_until_complete(me.store(upper_key, subkey=subkey2, value=456, expiration_time=now + 20))
    for node in [that_guy, me]:
        value, time = loop.run_until_complete(node.get(upper_key))
        assert isinstance(value, dict) and time == now + 20
        assert value[subkey1] == (123, now + 10)
        assert value[subkey2] == (456, now + 20)
        assert len(value) == 2

    assert not loop.run_until_complete(me.store(upper_key, subkey=subkey2, value=345, expiration_time=now + 10))
    assert loop.run_until_complete(me.store(upper_key, subkey=subkey2, value=567, expiration_time=now + 30))
    assert loop.run_until_complete(me.store(upper_key, subkey=subkey3, value=890, expiration_time=now + 50))
    loop.run_until_complete(asyncio.sleep(0.1))  # wait for cache to refresh

    for node in [that_guy, me]:
        value, time = loop.run_until_complete(node.get(upper_key))
        assert isinstance(value, dict) and time == now + 50, (value, time)
        assert value[subkey1] == (123, now + 10)
        assert value[subkey2] == (567, now + 30)
        assert value[subkey3] == (890, now + 50)
        assert len(value) == 3

    for proc in processes:
        proc.terminate()
    # The nodes don't own their hivemind.p2p.P2P instances, so we shutdown them separately
    loop.run_until_complete(asyncio.wait([node.shutdown() for node in [me, detached_node, that_guy]]))


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dhtnode_replicas():
    num_replicas = random.randint(1, 20)
    peers = await launch_star_shaped_swarm(n_peers=20, num_replicas=num_replicas)

    you = random.choice(peers)
    assert await you.store("key1", "foo", get_dht_time() + 999)

    actual_key1_replicas = sum(len(peer.protocol.storage) for peer in peers)
    assert num_replicas == actual_key1_replicas

    assert await you.store("key2", "bar", get_dht_time() + 999)
    total_size = sum(len(peer.protocol.storage) for peer in peers)
    actual_key2_replicas = total_size - actual_key1_replicas
    assert num_replicas == actual_key2_replicas

    assert await you.store("key2", "baz", get_dht_time() + 1000)
    assert sum(len(peer.protocol.storage) for peer in peers) == total_size, "total size should not have changed"


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dhtnode_caching(T=0.05):
    node2 = await DHTNode.create(cache_refresh_before_expiry=5 * T, reuse_get_requests=False)
    node1 = await DHTNode.create(
        initial_peers=await node2.protocol.p2p.get_visible_maddrs(),
        cache_refresh_before_expiry=5 * T,
        listen=False,
        reuse_get_requests=False,
    )
    await node2.store("k", [123, "value"], expiration_time=hivemind.get_dht_time() + 7 * T)
    await node2.store("k2", [654, "value"], expiration_time=hivemind.get_dht_time() + 7 * T)
    await node2.store("k3", [654, "value"], expiration_time=hivemind.get_dht_time() + 15 * T)
    await node1.get_many(["k", "k2", "k3", "k4"])
    assert len(node1.protocol.cache) == 3
    assert len(node1.cache_refresh_queue) == 0

    await node1.get_many(["k", "k2", "k3", "k4"])
    assert len(node1.cache_refresh_queue) == 3

    await node2.store("k", [123, "value"], expiration_time=hivemind.get_dht_time() + 12 * T)
    await asyncio.sleep(4 * T)
    await node1.get("k")
    await asyncio.sleep(1 * T)

    assert len(node1.protocol.cache) == 3
    assert len(node1.cache_refresh_queue) == 2
    await asyncio.sleep(3 * T)

    assert len(node1.cache_refresh_queue) == 1

    await asyncio.sleep(5 * T)
    assert len(node1.cache_refresh_queue) == 0
    await asyncio.sleep(5 * T)
    assert len(node1.cache_refresh_queue) == 0

    await node2.store("k", [123, "value"], expiration_time=hivemind.get_dht_time() + 10 * T)
    await node1.get("k")
    await asyncio.sleep(1 * T)
    assert len(node1.cache_refresh_queue) == 0
    await node1.get("k")
    await asyncio.sleep(1 * T)
    assert len(node1.cache_refresh_queue) == 1

    await asyncio.sleep(5 * T)
    assert len(node1.cache_refresh_queue) == 0

    await asyncio.gather(node1.shutdown(), node2.shutdown())


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dhtnode_reuse_get():
    peers = await launch_star_shaped_swarm(n_peers=10, parallel_rpc=256)

    await asyncio.gather(
        random.choice(peers).store("k1", 123, hivemind.get_dht_time() + 999),
        random.choice(peers).store("k2", 567, hivemind.get_dht_time() + 999),
    )

    you = random.choice(peers)

    futures1 = await you.get_many(["k1", "k2"], return_futures=True)
    assert len(you.pending_get_requests[DHTID.generate("k1")]) == 1
    assert len(you.pending_get_requests[DHTID.generate("k2")]) == 1

    futures2 = await you.get_many(["k2", "k3"], return_futures=True)
    assert len(you.pending_get_requests[DHTID.generate("k2")]) == 2

    await asyncio.gather(*futures1.values(), *futures2.values())
    futures3 = await you.get_many(["k3"], return_futures=True)
    assert len(you.pending_get_requests[DHTID.generate("k1")]) == 0
    assert len(you.pending_get_requests[DHTID.generate("k2")]) == 0
    assert len(you.pending_get_requests[DHTID.generate("k3")]) == 1

    assert (await futures1["k1"])[0] == 123
    assert await futures1["k2"] == await futures2["k2"] and (await futures1["k2"])[0] == 567
    assert await futures2["k3"] == await futures3["k3"] and (await futures3["k3"]) is None


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dhtnode_blacklist():
    node1, node2, node3, node4 = await launch_star_shaped_swarm(n_peers=4, blacklist_time=999)

    assert await node2.store("abc", 123, expiration_time=hivemind.get_dht_time() + 99)
    assert len(node2.blacklist.ban_counter) == 0

    await asyncio.gather(node3.shutdown(), node4.shutdown())

    assert await node2.store("def", 456, expiration_time=hivemind.get_dht_time() + 99)

    assert set(node2.blacklist.ban_counter.keys()) == {node3.peer_id, node4.peer_id}

    assert await node1.get("abc", latest=True)  # force node1 to crawl dht and discover unresponsive peers
    assert node3.peer_id in node1.blacklist

    assert await node1.get("abc", latest=True)  # force node1 to crawl dht and discover unresponsive peers
    assert node2.peer_id not in node1.blacklist

    await asyncio.gather(node1.shutdown(), node2.shutdown())


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dhtnode_edge_cases():
    peers = await launch_star_shaped_swarm(n_peers=4, parallel_rpc=4)

    subkeys = [0, "", False, True, "abyrvalg", 4555]
    keys = subkeys + [()]
    values = subkeys + [[]]
    for key, subkey, value in product(keys, subkeys, values):
        await random.choice(peers).store(
            key=key, subkey=subkey, value=value, expiration_time=hivemind.get_dht_time() + 999
        ),

        stored = await random.choice(peers).get(key=key, latest=True)
        assert stored is not None
        assert subkey in stored.value
        assert stored.value[subkey].value == value

    await asyncio.wait([node.shutdown() for node in peers])
