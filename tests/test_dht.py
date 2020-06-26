import time
import asyncio
import multiprocessing as mp
import random
import heapq
import uuid
from functools import partial
from itertools import chain
from typing import Optional
import numpy as np

import hivemind
from typing import List, Dict

from hivemind import get_dht_time
from hivemind.dht.node import DHTID, Endpoint, DHTNode, LOCALHOST, DHTProtocol
from hivemind.dht.protocol import LocalStorage


def run_protocol_listener(port: int, dhtid: DHTID, started: mp.synchronize.Event,
                          ping: Optional[hivemind.Endpoint] = None):
    loop = asyncio.new_event_loop()
    protocol = partial(DHTProtocol, dhtid, bucket_size=20, depth_modulo=5, wait_timeout=5)
    listen = loop.create_datagram_endpoint(protocol, local_addr=('127.0.0.1', port))
    transport, protocol = loop.run_until_complete(listen)
    print(f"Started peer id={protocol.node_id} port={port}", flush=True)

    if ping is not None:
        loop.run_until_complete(protocol.call_ping(ping))
    started.set()
    loop.run_forever()
    print(f"Finished peer id={protocol.node_id} port={port}", flush=True)


def test_kademlia_protocol():
    try:
        # create the first peer
        peer1_port, peer1_id, peer1_started = hivemind.find_open_port(), DHTID.generate(), mp.Event()
        peer1_proc = mp.Process(target=run_protocol_listener, args=(peer1_port, peer1_id, peer1_started), daemon=True)
        peer1_proc.start(), peer1_started.wait()

        # create another peer that connects to the first peer
        peer2_port, peer2_id, peer2_started = hivemind.find_open_port(), DHTID.generate(), mp.Event()
        peer2_proc = mp.Process(target=run_protocol_listener, args=(peer2_port, peer2_id, peer2_started),
                                kwargs={'ping': ('127.0.0.1', peer1_port)}, daemon=True)
        peer2_proc.start(), peer2_started.wait()

        port = hivemind.find_open_port()
        loop = asyncio.new_event_loop()
        protocol = partial(DHTProtocol, DHTID.generate(), bucket_size=20, depth_modulo=5, wait_timeout=5)
        listen = loop.create_datagram_endpoint(protocol, local_addr=('127.0.0.1', port))
        transport, protocol = loop.run_until_complete(listen)
        print(f"Self id={protocol.node_id} port={port}", flush=True)

        assert loop.run_until_complete(protocol.call_ping(('127.0.0.1', peer1_port))) == peer1_id

        key, value, expiration = DHTID.generate(), [123, {'ololo': 'pyshpysh'}], get_dht_time() + 1e3
        assert loop.run_until_complete(protocol.call_store(('127.0.0.1', peer1_port), key, value, expiration))

        # peer 1 must know about peer 2
        nodes_found = loop.run_until_complete(
            protocol.call_find_node(('127.0.0.1', peer1_port), key))
        (recv_id, recv_endpoint) = next(iter(nodes_found.items()))
        assert recv_id == peer2_id and recv_endpoint == ('127.0.0.1', peer2_port), \
            f"expected id={peer2_id}, port={('127.0.0.1', peer2_port)} but got {recv_id}, {recv_endpoint}"

        # peer 2 must know about peer 1
        nodes_found_2 = loop.run_until_complete(protocol.call_find_node(('127.0.0.1', peer2_port), key))
        (recv_id, recv_endpoint) = next(iter(nodes_found_2.items()))
        assert recv_id == peer1_id and recv_endpoint == ('127.0.0.1', peer1_port), \
            f"expected id={peer1_id}, port={('127.0.0.1', peer1_port)} but got {recv_id}, {recv_endpoint}"

        recv_value, recv_expiration, recv_peers = loop.run_until_complete(
            protocol.call_find_value(('127.0.0.1', peer1_port), key))
        assert recv_value == value and recv_expiration == expiration, "call_find_value expected " \
                                                                      f"{value} (expires by {expiration}) but got {recv_value} (expires by {recv_expiration})"
        print(recv_peers, nodes_found)
        assert recv_peers == nodes_found, "call_find_value must return the same peers as call_find_node"
        print("Kademlia test finished sucessfully!")

    finally:
        peer1_proc.terminate()
        peer2_proc.terminate()


def run_node(node_id, port, peers, status_pipe: mp.Pipe):
    if asyncio.get_event_loop().is_running():
        asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop
    asyncio.set_event_loop(asyncio.new_event_loop())
    try:
        node = DHTNode(node_id, port, initial_peers=peers)
        status_pipe.send('STARTED')
        while True:
            asyncio.get_event_loop().run_forever()
    except BaseException as e:
        status_pipe.send(e)  # report exception to master
        if not isinstance(e, OSError):
            raise e


def test_dht():
    # create dht with 50 nodes + your 51-st node
    dht: Dict[Endpoint, DHTID] = {}
    processes: List[mp.Process] = []
    port_fails, max_port_fails = 0, 10

    while len(dht) < 50:
        node_id = DHTID.generate()
        peers = random.sample(dht.keys(), min(len(dht), 5))
        port = hivemind.find_open_port()
        pipe_recv, pipe_send = mp.Pipe(duplex=False)
        proc = mp.Process(target=run_node, args=(node_id, port, peers, pipe_send), daemon=True)
        proc.start()

        status = pipe_recv.recv()
        if status == 'STARTED':
            processes.append(proc)
            dht[(LOCALHOST, port)] = node_id
        else:
            assert isinstance(status, BaseException)
            proc.terminate()
            if isinstance(status, OSError):  # port already in use. It just happens sometimes.
                port_fails += 1
                if port_fails > max_port_fails:
                    raise OSError("Too many 'Address already in use' errors.")
            else:
                raise ValueError(f"Failed to create node due to an error {status}, see traceback above")

    loop = asyncio.get_event_loop()
    me = hivemind.dht.node.DHTNode(initial_peers=random.sample(peers, 5), port=0)  # port=0 means os-specified port

    # test 1: find self
    nearest = loop.run_until_complete(me.find_nearest_nodes(key_id=me.node_id, k_nearest=1))
    assert len(nearest) == 1 and nearest[me.node_id] == (LOCALHOST, me.port)

    # test 2: find others
    for i in range(10):
        ref_endpoint, query_id = random.choice(list(dht.items()))
        nearest = loop.run_until_complete(me.find_nearest_nodes(key_id=query_id, k_nearest=1))
        assert len(nearest) == 1 and next(iter(nearest.items())) == (query_id, ref_endpoint)

    # test 3: find neighbors to random nodes
    accuracy_numerator = accuracy_denominator = 0  # top-1 nearest neighbor accuracy
    jaccard_numerator = jaccard_denominator = 0  # jaccard similarity aka intersection over union
    all_node_ids = list(dht.values())

    for i in range(100):
        query_id = DHTID.generate()
        k_nearest = random.randint(1, 20)
        exclude_self = random.random() > 0.5
        nearest = loop.run_until_complete(
            me.find_nearest_nodes(key_id=query_id, k_nearest=k_nearest, exclude_self=exclude_self))
        nearest_nodes = list(nearest)  # keys from ordered dict

        assert len(nearest_nodes) == k_nearest, "beam search must return exactly k_nearest results"
        assert me.node_id not in nearest_nodes or not exclude_self, "if exclude, results should not contain own node id"
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
    assert accuracy >= 0.9, f"Top-1 accuracy only {accuracy} ({accuracy_numerator} / {accuracy_denominator})"
    assert jaccard_index >= 0.9, f"Jaccard index only {accuracy} ({accuracy_numerator} / {accuracy_denominator})"

    # test 4: find all nodes
    nearest = loop.run_until_complete(
        me.find_nearest_nodes(key_id=DHTID.generate(), k_nearest=len(dht) + 100))
    assert len(nearest) == len(dht) + 1
    assert len(set.difference(set(nearest.keys()), set(all_node_ids) | {me.node_id})) == 0

    # test 5: node without peers
    other_node = hivemind.dht.node.DHTNode()
    nearest = loop.run_until_complete(other_node.find_nearest_nodes(DHTID.generate()))
    assert len(nearest) == 1 and nearest[other_node.node_id] == (LOCALHOST, other_node.port)
    nearest = loop.run_until_complete(other_node.find_nearest_nodes(DHTID.generate(), exclude_self=True))
    assert len(nearest) == 0

    # test 6 store and get value
    true_time = get_dht_time() + 1200
    assert loop.run_until_complete(me.store("mykey", ["Value", 10], true_time))
    val, expiration_time = loop.run_until_complete(me.get("mykey"))
    assert expiration_time == true_time, "Wrong time"
    assert val == ["Value", 10], "Wrong value"

    # terminate remaining processes
    for proc in processes:
        proc.terminate()


def test_hivemind_dht():
    peers = [hivemind.dht.DHT(start=True)]
    for i in range(10):
        neighbors_i = [('localhost', node.port) for node in random.sample(peers, min(3, len(peers)))]
        peers.append(hivemind.DHT(*neighbors_i, start=True))

    you: hivemind.dht.DHT = random.choice(peers)
    theguyshetoldyounottoworryabout: hivemind.dht.DHT = random.choice(peers)

    expert_uids = [str(uuid.uuid4()) for _ in range(110)]
    batch_size = 10
    for batch_start in range(0, len(expert_uids), batch_size):
        you.declare_experts(expert_uids[batch_start: batch_start + batch_size], 'localhost', 1234)

    found = theguyshetoldyounottoworryabout.get_experts(random.sample(expert_uids, 5) + ['foo', 'bar'])
    assert all(res is not None for res in found[:-2]), "Could not find some existing experts"
    assert all(res is None for res in found[-2:]), "Found non-existing experts"

    that_guys_expert, that_guys_port = str(uuid.uuid4()), random.randint(1000, 9999)
    theguyshetoldyounottoworryabout.declare_experts([that_guys_expert], 'that_host', that_guys_port)
    you_notfound, you_found = you.get_experts(['foobar', that_guys_expert])
    assert isinstance(you_found, hivemind.RemoteExpert)
    assert you_found.host == 'that_host', you_found.port == that_guys_port

    # test first_k_active
    assert theguyshetoldyounottoworryabout.first_k_active(expert_uids, k=10) == expert_uids[:10]

    some_permuted_experts = random.sample(expert_uids, k=32)
    assert theguyshetoldyounottoworryabout.first_k_active(some_permuted_experts, k=32) == some_permuted_experts
    assert theguyshetoldyounottoworryabout.first_k_active(some_permuted_experts, k=1) == some_permuted_experts[:1]
    fake_and_real_experts = list(chain(*zip(
        [str(uuid.uuid4()) for _ in some_permuted_experts], some_permuted_experts)))
    assert theguyshetoldyounottoworryabout.first_k_active(fake_and_real_experts, k=9) == some_permuted_experts[:9]

    for peer in peers:
        peer.shutdown()


def test_store():
    d = LocalStorage()
    d.store("key", "val", get_dht_time() + 10)
    assert d.get("key")[0] == "val", "Wrong value"
    print("Test store passed")


def test_get_expired():
    d = LocalStorage()
    d.store("key", "val", get_dht_time() + 1)
    time.sleep(2)
    assert d.get("key") == (None, None), "Expired value must be deleted"
    print("Test get expired passed")


def test_get_empty():
    d = LocalStorage()
    assert d.get("key") == (None, None), "Expired value must be deleted"
    print("Test get expired passed")


def test_change_expiration_time():
    d = LocalStorage()
    d.store("key", "val1", get_dht_time() + 2)
    d.store("key", "val2", get_dht_time() + 200)
    time.sleep(4)
    assert d.get("key")[0] == "val2", "Value must be changed, but still kept in table"
    print("Test change expiration time passed")


def test_maxsize_cache():
    d = LocalStorage(maxsize=1)
    d.store("key1", "val1", get_dht_time() + 1)
    d.store("key2", "val2", get_dht_time() + 200)
    assert d.get("key2")[0] == "val2", "Value with bigger exp. time must be kept"
    assert d.get("key1")[0] is None, "Value with less exp time, must be deleted"
