import asyncio
import multiprocessing as mp
import random
import heapq
from typing import Optional
import numpy as np
import pytest
from itertools import product
import hivemind
from typing import List, Dict

from hivemind import get_dht_time, replace_port
from hivemind.dht.node import DHTID, Endpoint, DHTNode, LOCALHOST
from hivemind.dht.protocol import DHTProtocol, ValidationError
from hivemind.dht.storage import DictionaryDHTValue


def run_protocol_listener(port: int, dhtid: DHTID, pipe_side: mp.connection.Connection, ping: Optional[Endpoint] = None):
    loop = asyncio.get_event_loop()
    protocol = loop.run_until_complete(DHTProtocol.create(
        dhtid, bucket_size=20, depth_modulo=5, num_replicas=3, wait_timeout=5, listen_on=f"{LOCALHOST}:{port}"))

    port = protocol.port
    print(f"Started peer id={protocol.node_id} port={port}", flush=True)

    if ping is not None:
        loop.run_until_complete(protocol.call_ping(ping))

    pipe_side.send((protocol.port, protocol.server.endpoint))

    loop.run_until_complete(protocol.server.wait_for_termination())
    print(f"Finished peer id={protocol.node_id} port={port}", flush=True)


@pytest.mark.forked
def test_dht_protocol():
    # create the first peer
    first_side, ours_side = mp.Pipe()
    peer1_port, peer1_id = hivemind.find_open_port(), DHTID.generate()
    peer1_proc = mp.Process(target=run_protocol_listener, args=(peer1_port, peer1_id, first_side), daemon=True)
    peer1_proc.start()
    peer1_port, peer1_endpoint = ours_side.recv()

    # create another peer that connects to the first peer
    second_side, ours_side = mp.Pipe()
    peer2_port, peer2_id = hivemind.find_open_port(), DHTID.generate()
    peer2_proc = mp.Process(target=run_protocol_listener, args=(peer2_port, peer2_id, second_side),
                            kwargs={'ping': peer1_endpoint}, daemon=True)
    peer2_proc.start()
    peer2_port, peer2_endpoint = ours_side.recv()

    loop = asyncio.get_event_loop()
    for listen in [False, True]:  # note: order matters, this test assumes that first run uses listen=False
        protocol = loop.run_until_complete(DHTProtocol.create(
            DHTID.generate(), bucket_size=20, depth_modulo=5, wait_timeout=5, num_replicas=3, listen=listen))
        print(f"Self id={protocol.node_id}", flush=True)

        assert loop.run_until_complete(protocol.call_ping(peer1_endpoint)) == peer1_id

        key, value, expiration = DHTID.generate(), [random.random(), {'ololo': 'pyshpysh'}], get_dht_time() + 1e3
        store_ok = loop.run_until_complete(protocol.call_store(
            peer1_endpoint, [key], [hivemind.MSGPackSerializer.dumps(value)], expiration)
        )
        assert all(store_ok), "DHT rejected a trivial store"

        # peer 1 must know about peer 2
        (recv_value_bytes, recv_expiration), nodes_found = loop.run_until_complete(
            protocol.call_find(peer1_endpoint, [key]))[key]
        recv_value = hivemind.MSGPackSerializer.loads(recv_value_bytes)
        (recv_id, recv_endpoint) = next(iter(nodes_found.items()))
        assert recv_id == peer2_id and recv_endpoint == peer2_endpoint, \
            f"expected id={peer2_id}, peer={peer2_endpoint} but got {recv_id}, {recv_endpoint}"

        assert recv_value == value and recv_expiration == expiration, \
            f"call_find_value expected {value} (expires by {expiration}) " \
            f"but got {recv_value} (expires by {recv_expiration})"

        # peer 2 must know about peer 1, but not have a *random* nonexistent value
        dummy_key = DHTID.generate()
        empty_item, nodes_found_2 = loop.run_until_complete(
            protocol.call_find(peer2_endpoint, [dummy_key]))[dummy_key]
        assert empty_item is None, "Non-existent keys shouldn't have values"
        (recv_id, recv_endpoint) = next(iter(nodes_found_2.items()))
        assert recv_id == peer1_id and recv_endpoint == peer1_endpoint, \
            f"expected id={peer1_id}, peer={peer1_endpoint} but got {recv_id}, {recv_endpoint}"

        # cause a non-response by querying a nonexistent peer
        dummy_port = hivemind.find_open_port()
        assert loop.run_until_complete(protocol.call_find(f"{LOCALHOST}:{dummy_port}", [key])) is None

        # store/get a dictionary with sub-keys
        nested_key, subkey1, subkey2 = DHTID.generate(), 'foo', 'bar'
        value1, value2 = [random.random(), {'ololo': 'pyshpysh'}], 'abacaba'
        assert loop.run_until_complete(protocol.call_store(
            peer1_endpoint, keys=[nested_key], values=[hivemind.MSGPackSerializer.dumps(value1)],
            expiration_time=[expiration], subkeys=[subkey1])
        )
        assert loop.run_until_complete(protocol.call_store(
            peer1_endpoint, keys=[nested_key], values=[hivemind.MSGPackSerializer.dumps(value2)],
            expiration_time=[expiration + 5], subkeys=[subkey2])
        )
        (recv_dict, recv_expiration), nodes_found = loop.run_until_complete(
            protocol.call_find(peer1_endpoint, [nested_key]))[nested_key]
        assert isinstance(recv_dict, DictionaryDHTValue)
        assert len(recv_dict.data) == 2 and recv_expiration == expiration + 5
        assert recv_dict.data[subkey1] == (protocol.serializer.dumps(value1), expiration)
        assert recv_dict.data[subkey2] == (protocol.serializer.dumps(value2), expiration + 5)

        assert protocol.client.endpoint == loop.run_until_complete(protocol.get_outgoing_request_endpoint(peer1_endpoint))

        if listen:
            loop.run_until_complete(protocol.shutdown())

    peer1_proc.terminate()
    peer2_proc.terminate()


@pytest.mark.forked
def test_empty_table():
    """ Test RPC methods with empty routing table """
    theirs_side, ours_side = mp.Pipe()
    peer_port, peer_id = hivemind.find_open_port(), DHTID.generate()
    peer_proc = mp.Process(target=run_protocol_listener, args=(peer_port, peer_id, theirs_side), daemon=True)
    peer_proc.start()
    peer_port, peer_endpoint = ours_side.recv()

    loop = asyncio.get_event_loop()
    protocol = loop.run_until_complete(DHTProtocol.create(
        DHTID.generate(), bucket_size=20, depth_modulo=5, wait_timeout=5, num_replicas=3, listen=False))

    key, value, expiration = DHTID.generate(), [random.random(), {'ololo': 'pyshpysh'}], get_dht_time() + 1e3

    empty_item, nodes_found = loop.run_until_complete(
        protocol.call_find(peer_endpoint, [key]))[key]
    assert empty_item is None and len(nodes_found) == 0
    assert all(loop.run_until_complete(protocol.call_store(
        peer_endpoint, [key], [hivemind.MSGPackSerializer.dumps(value)], expiration)
    )), "peer rejected store"

    (recv_value_bytes, recv_expiration), nodes_found = loop.run_until_complete(
        protocol.call_find(peer_endpoint, [key]))[key]
    recv_value = hivemind.MSGPackSerializer.loads(recv_value_bytes)
    assert len(nodes_found) == 0
    assert recv_value == value and recv_expiration == expiration

    assert loop.run_until_complete(protocol.call_ping(peer_endpoint)) == peer_id
    assert loop.run_until_complete(protocol.call_ping(f'{LOCALHOST}:{hivemind.find_open_port()}')) is None
    peer_proc.terminate()


def run_node(node_id, peers, status_pipe: mp.Pipe):
    if asyncio.get_event_loop().is_running():
        asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop
        asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    node = loop.run_until_complete(DHTNode.create(node_id, initial_peers=peers))
    status_pipe.send((node.port, node.endpoint))
    loop.run_until_complete(node.protocol.server.wait_for_termination())


@pytest.mark.forked
def test_dht_node():
    # create dht with 50 nodes + your 51-st node
    dht: Dict[Endpoint, DHTID] = {}
    processes: List[mp.Process] = []

    for i in range(5):
        node_id = DHTID.generate()
        peers = random.sample(dht.keys(), min(len(dht), 5))
        pipe_recv, pipe_send = mp.Pipe(duplex=False)
        proc = mp.Process(target=run_node, args=(node_id, peers, pipe_send), daemon=True)
        proc.start()
        port, endpoint = pipe_recv.recv()
        processes.append(proc)
        dht[endpoint] = node_id

    loop = asyncio.get_event_loop()
    me = loop.run_until_complete(DHTNode.create(initial_peers=random.sample(dht.keys(), min(len(dht), 5)), parallel_rpc=2,
                                                cache_refresh_before_expiry=False))

    # test 1: find self
    nearest = loop.run_until_complete(me.find_nearest_nodes([me.node_id], k_nearest=1))[me.node_id]
    assert len(nearest) == 1 and nearest[me.node_id] == me.endpoint

    # test 2: find others
    for i in range(10):
        ref_endpoint, query_id = random.choice(list(dht.items()))
        nearest = loop.run_until_complete(me.find_nearest_nodes([query_id], k_nearest=1))[query_id]
        assert len(nearest) == 1
        found_node_id, found_endpoint = next(iter(nearest.items()))
        assert found_node_id == query_id and found_endpoint == ref_endpoint

    # test 3: find neighbors to random nodes
    accuracy_numerator = accuracy_denominator = 0  # top-1 nearest neighbor accuracy
    jaccard_numerator = jaccard_denominator = 0  # jaccard similarity aka intersection over union
    all_node_ids = list(dht.values())

    for i in range(10):
        query_id = DHTID.generate()
        k_nearest = random.randint(1, len(dht))
        exclude_self = random.random() > 0.5
        nearest = loop.run_until_complete(
            me.find_nearest_nodes([query_id], k_nearest=k_nearest, exclude_self=exclude_self))[query_id]
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
    print("Top-1 accuracy:", accuracy)  # should be 98-100%
    jaccard_index = jaccard_numerator / jaccard_denominator
    print("Jaccard index (intersection over union):", jaccard_index)  # should be 95-100%
    assert accuracy >= 0.7, f"Top-1 accuracy only {accuracy} ({accuracy_numerator} / {accuracy_denominator})"
    assert jaccard_index >= 0.8, f"Jaccard index only {accuracy} ({accuracy_numerator} / {accuracy_denominator})"

    # test 4: find all nodes
    dummy = DHTID.generate()
    nearest = loop.run_until_complete(me.find_nearest_nodes([dummy], k_nearest=len(dht) + 100))[dummy]
    assert len(nearest) == len(dht) + 1
    assert len(set.difference(set(nearest.keys()), set(all_node_ids) | {me.node_id})) == 0

    # test 5: node without peers
    detached_node = loop.run_until_complete(DHTNode.create())
    nearest = loop.run_until_complete(detached_node.find_nearest_nodes([dummy]))[dummy]
    assert len(nearest) == 1 and nearest[detached_node.node_id] == detached_node.endpoint
    nearest = loop.run_until_complete(detached_node.find_nearest_nodes([dummy], exclude_self=True))[dummy]
    assert len(nearest) == 0

    # test 6 store and get value
    true_time = get_dht_time() + 1200
    assert loop.run_until_complete(me.store("mykey", ["Value", 10], true_time))
    that_guy = loop.run_until_complete(DHTNode.create(initial_peers=random.sample(dht.keys(), 3), parallel_rpc=10,
                                                      cache_refresh_before_expiry=False, cache_locally=False))

    for node in [me, that_guy]:
        val, expiration_time = loop.run_until_complete(node.get("mykey"))
        assert val == ["Value", 10], "Wrong value"
        assert expiration_time == true_time, f"Wrong time"

    assert loop.run_until_complete(detached_node.get("mykey")) is None

    # test 7: bulk store and bulk get
    keys = 'foo', 'bar', 'baz', 'zzz'
    values = 3, 2, 'batman', [1, 2, 3]
    store_ok = loop.run_until_complete(me.store_many(keys, values, expiration_time=get_dht_time() + 999))
    assert all(store_ok.values()), "failed to store one or more keys"
    response = loop.run_until_complete(me.get_many(keys[::-1]))
    for key, value in zip(keys, values):
        assert key in response and response[key][0] == value

    # test 8: store dictionaries as values (with sub-keys)
    upper_key, subkey1, subkey2, subkey3 = 'ololo', 'k1', 'k2', 'k3'
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


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dhtnode_replicas():
    dht_size = 5
    initial_peers = 3
    num_replicas = random.randint(1, 5)

    peers = []
    for i in range(dht_size):
        neighbors_i = [node.endpoint for node in random.sample(peers, min(initial_peers, len(peers)))]
        peers.append(await DHTNode.create(initial_peers=neighbors_i, num_replicas=num_replicas))

    you = random.choice(peers)
    assert await you.store('key1', 'foo', get_dht_time() + 999)

    actual_key1_replicas = sum(len(peer.protocol.storage) for peer in peers)
    assert num_replicas == actual_key1_replicas

    assert await you.store('key2', 'bar', get_dht_time() + 999)
    total_size = sum(len(peer.protocol.storage) for peer in peers)
    actual_key2_replicas = total_size - actual_key1_replicas
    assert num_replicas == actual_key2_replicas

    assert await you.store('key2', 'baz', get_dht_time() + 1000)
    assert sum(len(peer.protocol.storage) for peer in peers) == total_size, "total size should not have changed"


# @pytest.mark.forked
# @pytest.mark.asyncio
# async def test_dhtnode_caching(T=0.05):
#     node2 = await hivemind.DHTNode.create(cache_refresh_before_expiry=5 * T, reuse_get_requests=False)
#     node1 = await hivemind.DHTNode.create(initial_peers=[node2.endpoint],
#                                           cache_refresh_before_expiry=5 * T, listen=False, reuse_get_requests=False)
#     await node2.store('k', [123, 'value'], expiration_time=hivemind.get_dht_time() + 7 * T)
#     await node2.store('k2', [654, 'value'], expiration_time=hivemind.get_dht_time() + 7 * T)
#     await node2.store('k3', [654, 'value'], expiration_time=hivemind.get_dht_time() + 15 * T)
#     await node1.get_many(['k', 'k2', 'k3', 'k4'])
#     assert len(node1.protocol.cache) == 3
#     assert len(node1.cache_refresh_queue) == 0
#
#     await node1.get_many(['k', 'k2', 'k3', 'k4'])
#     assert len(node1.cache_refresh_queue) == 3
#
#     await node2.store('k', [123, 'value'], expiration_time=hivemind.get_dht_time() + 12 * T)
#     await asyncio.sleep(4 * T)
#     await node1.get('k')
#     await asyncio.sleep(1 * T)
#
#     assert len(node1.protocol.cache) == 3
#     assert len(node1.cache_refresh_queue) == 2
#     await asyncio.sleep(3 * T)
#
#     assert len(node1.cache_refresh_queue) == 1
#
#     await asyncio.sleep(5 * T)
#     assert len(node1.cache_refresh_queue) == 0
#     await asyncio.sleep(5 * T)
#     assert len(node1.cache_refresh_queue) == 0
#
#     await node2.store('k', [123, 'value'], expiration_time=hivemind.get_dht_time() + 10 * T)
#     await node1.get('k')
#     await asyncio.sleep(1 * T)
#     assert len(node1.cache_refresh_queue) == 0
#     await node1.get('k')
#     await asyncio.sleep(1 * T)
#     assert len(node1.cache_refresh_queue) == 1
#
#     await asyncio.sleep(5 * T)
#     assert len(node1.cache_refresh_queue) == 0
#
#     await asyncio.gather(node1.shutdown(), node2.shutdown())


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dhtnode_reuse_get():
    peers = []
    for i in range(5):
        neighbors_i = [node.endpoint for node in random.sample(peers, min(3, len(peers)))]
        peers.append(await hivemind.DHTNode.create(initial_peers=neighbors_i, parallel_rpc=32))

    await asyncio.gather(
        random.choice(peers).store('k1', 123, hivemind.get_dht_time() + 999),
        random.choice(peers).store('k2', 567, hivemind.get_dht_time() + 999)
    )

    you = random.choice(peers)

    futures1 = await you.get_many(['k1', 'k2'], return_futures=True)
    assert len(you.pending_get_requests[DHTID.generate('k1')]) == 1
    assert len(you.pending_get_requests[DHTID.generate('k2')]) == 1

    futures2 = await you.get_many(['k2', 'k3'], return_futures=True)
    assert len(you.pending_get_requests[DHTID.generate('k2')]) == 2

    await asyncio.gather(*futures1.values(), *futures2.values())
    futures3 = await you.get_many(['k3'], return_futures=True)
    assert len(you.pending_get_requests[DHTID.generate('k1')]) == 0
    assert len(you.pending_get_requests[DHTID.generate('k2')]) == 0
    assert len(you.pending_get_requests[DHTID.generate('k3')]) == 1

    assert (await futures1['k1'])[0] == 123
    assert await futures1['k2'] == await futures2['k2'] and (await futures1['k2'])[0] == 567
    assert await futures2['k3'] == await futures3['k3'] and (await futures3['k3']) is None


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dhtnode_blacklist():
    node1 = await hivemind.DHTNode.create(blacklist_time=999)
    node2 = await hivemind.DHTNode.create(blacklist_time=999, initial_peers=[node1.endpoint])
    node3 = await hivemind.DHTNode.create(blacklist_time=999, initial_peers=[node1.endpoint])
    node4 = await hivemind.DHTNode.create(blacklist_time=999, initial_peers=[node1.endpoint])

    assert await node2.store('abc', 123, expiration_time=hivemind.get_dht_time() + 99)
    assert len(node2.blacklist.ban_counter) == 0

    await node3.shutdown()
    await node4.shutdown()

    assert await node2.store('def', 456, expiration_time=hivemind.get_dht_time() + 99)

    assert len(node2.blacklist.ban_counter) == 2

    for banned_peer in node2.blacklist.ban_counter:
        assert any(banned_peer == endpoint for endpoint in [node3.endpoint, node4.endpoint])

    node3_endpoint = await node3.protocol.get_outgoing_request_endpoint(node1.endpoint)
    assert await node1.get('abc', latest=True)  # force node1 to crawl dht and discover unresponsive peers
    assert node3_endpoint in node1.blacklist

    node2_endpoint = await node2.protocol.get_outgoing_request_endpoint(node1.endpoint)
    assert await node1.get('abc', latest=True)  # force node1 to crawl dht and discover unresponsive peers
    assert node2_endpoint not in node1.blacklist


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dhtnode_validate(fake_endpoint='127.0.0.721:*'):

    node1 = await hivemind.DHTNode.create(blacklist_time=999)
    with pytest.raises(ValidationError):
        node2 = await hivemind.DHTNode.create(blacklist_time=999, initial_peers=[node1.endpoint],
                                              endpoint=fake_endpoint)


# @pytest.mark.forked
# @pytest.mark.asyncio
# async def test_dhtnode_edge_cases():
#     peers = []
#     for i in range(5):
#         neighbors_i = [node.endpoint for node in random.sample(peers, min(3, len(peers)))]
#         peers.append(await hivemind.DHTNode.create(initial_peers=neighbors_i, parallel_rpc=32))
#
#     subkeys = [0, '', False, True, 'abyrvalg', 4555]
#     keys = subkeys + [()]
#     values = subkeys + [[]]
#     for key, subkey, value in product(keys, subkeys, values):
#         await random.choice(peers).store(key=key, subkey=subkey, value=value,
#                                          expiration_time=hivemind.get_dht_time() + 999),
#
#         stored = await random.choice(peers).get(key=key, latest=True)
#         assert stored is not None
#         assert subkey in stored.value
#         assert stored.value[subkey].value == value
