import asyncio
import heapq
import random
from itertools import product

import numpy as np
import pytest

import hivemind
from hivemind import get_dht_time
from hivemind.dht.node import DHTID, DHTNode
from hivemind.utils.logging import get_logger

from test_utils.dht_swarms import launch_star_shaped_swarm, launch_swarm_in_separate_processes

logger = get_logger(__name__)

# note: we run network-related tests in a separate process to re-initialize all global states from scratch
# this helps us avoid undesirable gRPC side-effects (e.g. segfaults) when running multiple tests in sequence


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dht_node(
    n_peers: int = 20, n_sequential_peers: int = 5, parallel_rpc: int = 10, bucket_size: int = 5, num_replicas: int = 3
):
    # step A: create a swarm of 50 dht nodes in separate processes
    #         (first 5 created sequentially, others created in parallel)

    processes, dht, swarm_maddrs = launch_swarm_in_separate_processes(
        n_peers=n_peers, n_sequential_peers=n_sequential_peers, bucket_size=bucket_size, num_replicas=num_replicas
    )

    # step B: run 51-st node in this process
    initial_peers = random.choice(swarm_maddrs)
    me = await DHTNode.create(
        initial_peers=initial_peers,
        parallel_rpc=parallel_rpc,
        bucket_size=bucket_size,
        num_replicas=num_replicas,
        cache_refresh_before_expiry=False,
    )

    # test 1: find self
    nearest = (await me.find_nearest_nodes([me.node_id], k_nearest=1))[me.node_id]
    assert len(nearest) == 1 and nearest[me.node_id] == me.peer_id

    # test 2: find others
    for _ in range(10):
        ref_peer_id, query_id = random.choice(list(dht.items()))
        nearest = (await me.find_nearest_nodes([query_id], k_nearest=1))[query_id]
        assert len(nearest) == 1
        found_node_id, found_peer_id = next(iter(nearest.items()))
        assert found_node_id == query_id and found_peer_id == ref_peer_id

    # test 3: find neighbors to random nodes
    accuracy_numerator = accuracy_denominator = 0  # top-1 nearest neighbor accuracy
    jaccard_numerator = jaccard_denominator = 0  # jaccard similarity aka intersection over union
    all_node_ids = list(dht.values())

    for _ in range(20):
        query_id = DHTID.generate()
        k_nearest = random.randint(1, 10)
        exclude_self = random.random() > 0.5
        find_result = await me.find_nearest_nodes([query_id], k_nearest=k_nearest, exclude_self=exclude_self)
        nearest_nodes = list(find_result[query_id])  # keys from ordered dict

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
    logger.debug(f"Top-1 accuracy: {accuracy}")  # should be 90-100%
    jaccard_index = jaccard_numerator / jaccard_denominator
    logger.debug(f"Jaccard index (intersection over union): {jaccard_index}")  # should be 95-100%
    assert accuracy >= 0.8, f"Top-1 accuracy only {accuracy} ({accuracy_numerator} / {accuracy_denominator})"
    assert jaccard_index >= 0.9, f"Jaccard index only {accuracy} ({accuracy_numerator} / {accuracy_denominator})"

    # test 4: find all nodes
    dummy = DHTID.generate()
    nearest = (await me.find_nearest_nodes([dummy], k_nearest=len(dht) + 100))[dummy]
    assert len(nearest) == len(dht) + 1
    assert len(set.difference(set(nearest.keys()), set(all_node_ids) | {me.node_id})) == 0

    # test 5: node without peers
    detached_node = await DHTNode.create()
    nearest = (await detached_node.find_nearest_nodes([dummy]))[dummy]
    assert len(nearest) == 1 and nearest[detached_node.node_id] == detached_node.peer_id
    nearest = (await detached_node.find_nearest_nodes([dummy], exclude_self=True))[dummy]
    assert len(nearest) == 0

    # test 6: store and get value
    true_time = get_dht_time() + 1200
    assert await me.store("mykey", ["Value", 10], true_time)

    initial_peers = random.choice(swarm_maddrs)
    that_guy = await DHTNode.create(
        initial_peers=initial_peers,
        parallel_rpc=parallel_rpc,
        cache_refresh_before_expiry=False,
        cache_locally=False,
    )

    for node in [me, that_guy]:
        val, expiration_time = await node.get("mykey")
        assert val == ["Value", 10], "Wrong value"
        assert expiration_time == true_time, f"Wrong time"

    assert not await detached_node.get("mykey")

    # test 7: bulk store and bulk get
    keys = "foo", "bar", "baz", "zzz"
    values = 3, 2, "batman", [1, 2, 3]
    store_ok = await me.store_many(keys, values, expiration_time=get_dht_time() + 999)
    assert all(store_ok.values()), "failed to store one or more keys"
    response = await me.get_many(keys[::-1])
    for key, value in zip(keys, values):
        assert key in response and response[key][0] == value

    # test 8: store dictionaries as values (with sub-keys)
    upper_key, subkey1, subkey2, subkey3 = "ololo", "k1", "k2", "k3"
    now = get_dht_time()
    assert await me.store(upper_key, subkey=subkey1, value=123, expiration_time=now + 10)
    assert await me.store(upper_key, subkey=subkey2, value=456, expiration_time=now + 20)
    for node in [that_guy, me]:
        value, time = await node.get(upper_key)
        assert isinstance(value, dict) and time == now + 20
        assert value[subkey1] == (123, now + 10)
        assert value[subkey2] == (456, now + 20)
        assert len(value) == 2

    assert not await me.store(upper_key, subkey=subkey2, value=345, expiration_time=now + 10)
    assert await me.store(upper_key, subkey=subkey2, value=567, expiration_time=now + 30)
    assert await me.store(upper_key, subkey=subkey3, value=890, expiration_time=now + 50)

    for node in [that_guy, me]:
        value, time = await node.get(upper_key, latest=True)
        assert isinstance(value, dict) and time == now + 50, (value, time)
        assert value[subkey1] == (123, now + 10)
        assert value[subkey2] == (567, now + 30)
        assert value[subkey3] == (890, now + 50)
        assert len(value) == 3

    for proc in processes:
        proc.terminate()
    # The nodes don't own their hivemind.p2p.P2P instances, so we shutdown them separately
    await asyncio.gather(me.shutdown(), that_guy.shutdown(), detached_node.shutdown())


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
        client_mode=True,
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
