import time
import asyncio
import multiprocessing as mp
import random
import heapq
from functools import partial
from typing import Optional
import numpy as np

import hivemind
from typing import List, Dict
from hivemind.dht.node import DHTID, Endpoint, DHTNode, LOCALHOST, KademliaProtocol
from hivemind.dht.protocol import LocalStorage


def run_protocol_listener(port: int, dhtid: DHTID, started: mp.synchronize.Event, ping: Optional[hivemind.Endpoint] = None):
    loop = asyncio.new_event_loop()
    protocol = partial(KademliaProtocol, dhtid, bucket_size=20, depth_modulo=5, wait_timeout=5)
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
        protocol = partial(KademliaProtocol, DHTID.generate(), bucket_size=20, depth_modulo=5, wait_timeout=5)
        listen = loop.create_datagram_endpoint(protocol, local_addr=('127.0.0.1', port))
        transport, protocol = loop.run_until_complete(listen)
        print(f"Self id={protocol.node_id} port={port}", flush=True)

        assert loop.run_until_complete(protocol.call_ping(('127.0.0.1', peer1_port))) == peer1_id

        key, value, expiration = DHTID.generate(), [123, {'ololo': 'pyshpysh'}], time.monotonic() + 1e3
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


def run_node(node_id, port, peers, ready: mp.Event):
    if asyncio.get_event_loop().is_running():
        asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop
    asyncio.set_event_loop(asyncio.new_event_loop())
    node = DHTNode(node_id, port, initial_peers=peers)
    await_forever = hivemind.run_forever(asyncio.get_event_loop().run_forever)
    ready.set()
    await_forever.result()  # process will exit only if event loop broke down


def test_beam_search_dht():
    # create dht with 50 nodes + your 51-st node
    dht: Dict[Endpoint, DHTID] = {}
    processes: List[mp.Process] = []

    for i in range(50):
        node_id = DHTID.generate()
        port = hivemind.find_open_port()
        peers = random.sample(dht.keys(), min(len(dht), 5))
        ready = mp.Event()
        proc = mp.Process(target=run_node, args=(node_id, port, peers, ready), daemon=True)
        proc.start()
        ready.wait()
        processes.append(proc)
        dht[(LOCALHOST, port)] = node_id

    loop = asyncio.get_event_loop()
    me = hivemind.dht.node.DHTNode(initial_peers=random.sample(peers, 5))

    # test 1: find self
    nearest = loop.run_until_complete(me.beam_search(query_id=me.node_id, k_nearest=1))
    assert len(nearest) == 1 and nearest[me.node_id] == (LOCALHOST, me.port)

    # test 2: find others
    for i in range(10):
        ref_endpoint, query_id = random.choice(list(dht.items()))
        nearest = loop.run_until_complete(me.beam_search(query_id=query_id, k_nearest=1))
        assert len(nearest) == 1 and next(iter(nearest.items())) == (query_id, ref_endpoint)

    # test 3: find neighbors to random nodes
    accuracy_numerator = accuracy_denominator = 0
    all_node_ids = list(dht.values())
    for i in range(50):
        query_id = DHTID.generate()
        k_nearest = random.randint(1, 20)
        exclude_self = random.random() > 0.5
        nearest = loop.run_until_complete(
            me.beam_search(query_id=query_id, k_nearest=k_nearest, exclude_self=exclude_self))
        nearest_nodes = list(nearest)  # keys from ordered dict

        assert len(nearest_nodes) == k_nearest, "beam search must return exactly k_nearest results"
        assert me.node_id not in nearest_nodes or not exclude_self, "if exclude, results should not contain own node id"
        assert np.all(np.diff(query_id.xor_distance(nearest_nodes)) >= 0), "results must be sorted by distance"

        ref_nearest = heapq.nsmallest(k_nearest + 1, all_node_ids, key=query_id.xor_distance)
        if exclude_self and me.node_id in ref_nearest:
            ref_nearest.remove(me.node_id)
        if len(ref_nearest) > k_nearest:
            ref_nearest.pop()

        assert nearest_nodes[0] == ref_nearest[0]
        accuracy_numerator += len(set.intersection(set(nearest_nodes), set(ref_nearest)))
        accuracy_denominator += k_nearest

    accuracy = accuracy_numerator / accuracy_denominator
    assert accuracy > 0.95, f"Beam search accuracy only {accuracy} ({accuracy_numerator} out of {accuracy_denominator})"
    print("Accuracy:", accuracy)  # should be 98-99%

    # test 4: find all nodes
    nearest = loop.run_until_complete(
        me.beam_search(query_id=DHTID.generate(), k_nearest=len(dht) + 100))
    assert len(nearest) == len(dht) + 1
    assert len(set.difference(set(nearest.keys()), set(all_node_ids) | {me.node_id})) == 0

    # test 5: node without peers
    me = hivemind.dht.node.DHTNode()
    nearest = loop.run_until_complete(me.beam_search(DHTID.generate()))
    assert len(nearest) == 1 and nearest[me.node_id] == (LOCALHOST, me.port)
    nearest = loop.run_until_complete(me.beam_search(DHTID.generate(), exclude_self=True))
    assert len(nearest) == 0


def test_store():
    d = LocalStorage()
    d.store("key", "val", time.monotonic()+10)
    assert d.get("key")[0] == "val", "Wrong value"
    print("Test store passed")


def test_get_expired():
    d = LocalStorage(keep_expired=False)
    d.store("key", "val", time.monotonic()+1)
    time.sleep(2)
    assert d.get("key") == (None, None), "Expired value must be deleted"
    print("Test get expired passed")


def test_store_maxsize():
    d = LocalStorage(maxsize=1)
    d.store("key1", "val1", time.monotonic() + 1)
    d.store("key2", "val2", time.monotonic() + 2)
    assert d.get("key1") == (None, None), "elder a value must be deleted"
    assert d.get("key2")[0] == "val2", "Newer should be stored"
    print("Test store maxsize passed")