import argparse
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


def run_benchmark_node(node_id, port, peers, ready: mp.Event, request_perod,
                       expiration_time, wait_before_read, time_to_test, statistics: mp.Queue, dht_loaded: mp.Event):
    if asyncio.get_event_loop().is_running():
        asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    node = DHTNode(node_id, port, initial_peers=peers)
    await_forever = hivemind.run_forever(asyncio.get_event_loop().run_forever)
    ready.set()
    dht_loaded.wait()
    start = time.monotonic()
    while time.monotonic() < start + time_to_test:
        query_id = DHTID.generate()
        store_value = random.randint(0, 256)

        store_time = time.monotonic()
        success_store = asyncio.run_coroutine_threadsafe(
            node.store(query_id, store_value, store_time + expiration_time), loop).result()
        store_time = time.monotonic() - store_time
        if success_store:
            time.sleep(wait_before_read)
            get_time = time.monotonic()
            get_value, get_time_expiration = asyncio.run_coroutine_threadsafe(node.get(query_id), loop).result()
            get_time = time.monotonic() - get_time
            success_get = (get_value == store_value)
            statistics.put((success_store, store_time, success_get, get_time))
        else:
            statistics.put((success_store, store_time, None, None))
    await_forever.result()  # process will exit only if event loop broke down


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=20, required=False)
    parser.add_argument('--request_perod', type=float, default=2, required=False)
    parser.add_argument('--expiration_time', type=float, default=10, required=False)
    parser.add_argument('--wait_before_read', type=float, default=1, required=False)
    parser.add_argument('--time_to_test', type=float, default=10, required=False)
    args = parser.parse_args()

    statistics = mp.Queue()
    dht: Dict[Endpoint, DHTID] = {}
    processes: List[mp.Process] = []

    num_nodes = args.num_nodes
    request_perod = args.request_perod
    expiration_time = args.expiration_time
    wait_before_read = args.wait_before_read
    time_to_test = args.time_to_test

    dht_loaded = mp.Event()
    for i in range(num_nodes):
        node_id = DHTID.generate()
        port = hivemind.find_open_port()
        peers = random.sample(dht.keys(), min(len(dht), 5))
        ready = mp.Event()
        proc = mp.Process(target=run_benchmark_node, args=(node_id, port, peers, ready, request_perod,
                                                           expiration_time, wait_before_read, time_to_test, statistics,
                                                           dht_loaded), daemon=True)
        proc.start()
        ready.wait()
        processes.append(proc)
        dht[(LOCALHOST, port)] = node_id
    dht_loaded.set()
    time.sleep(time_to_test)
    success_store = 0
    all_store = 0
    time_store = 0
    success_get = 0
    all_get = 0
    time_get = 0
    while not statistics.empty():
        success_store_i, store_time_i, success_get_i, get_time_i = statistics.get()
        all_store += 1
        time_store += store_time_i
        if success_store_i:
            success_store += 1
            all_get += 1
            success_get += 1 if success_get_i else 0
            time_get += get_time_i
    alive_nodes_count = 0
    loop = asyncio.new_event_loop()
    node = DHTNode(loop=loop)
    for addr, port in dht:
        if loop.run_until_complete(node.protocol.call_ping((addr, port))) is not None:
            alive_nodes_count += 1
    print("store success rate: ", success_store / all_store)
    print("mean store time: ", time_store / all_store)
    print("get success rate: ", success_get / all_get)
    print("mean get time: ", time_get / all_get)
    print("death rate: ", (num_nodes - alive_nodes_count) / num_nodes)
