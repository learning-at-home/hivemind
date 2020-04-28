import asyncio
import multiprocessing as mp
import time
from functools import partial
from typing import Optional

import hivemind.dht.protocol
from hivemind.dht.node import DHTID
from hivemind.dht.node import KademliaProtocol


def test_kademlia_protocol():
    try:
        # create the first peer
        peer1_port, peer1_id, peer1_started = (
            hivemind.find_open_port(),
            DHTID.generate(),
            mp.Event(),
        )
        peer1_proc = mp.Process(
            target=dht_peer_func,
            args=(peer1_port, peer1_id, peer1_started),
            daemon=True,
        )
        peer1_proc.start(), peer1_started.wait()

        # create another peer that connects to the first peer
        peer2_port, peer2_id, peer2_started = (
            hivemind.find_open_port(),
            DHTID.generate(),
            mp.Event(),
        )
        peer2_proc = mp.Process(
            target=dht_peer_func,
            args=(peer2_port, peer2_id, peer2_started),
            kwargs={"ping": ("127.0.0.1", peer1_port)},
            daemon=True,
        )
        peer2_proc.start(), peer2_started.wait()

        port = hivemind.find_open_port()
        loop = asyncio.new_event_loop()
        protocol = partial(KademliaProtocol, DHTID.generate(), 20, 5, 300, 5)
        listen = loop.create_datagram_endpoint(protocol, local_addr=("127.0.0.1", port))
        transport, protocol = loop.run_until_complete(listen)
        print(f"Self id={protocol.node_id} port={port}", flush=True)

        assert (
            loop.run_until_complete(protocol.call_ping(("127.0.0.1", peer1_port)))
            == peer1_id
        )

        key, value, expiration = (
            DHTID.generate(),
            [123, {"ololo": "pyshpysh"}],
            time.monotonic() + 1e3,
        )
        assert loop.run_until_complete(
            protocol.call_store(("127.0.0.1", peer1_port), key, value, expiration)
        )

        # peer 1 must know about peer 2
        ((recv_id, recv_endpoint),) = nodes_found = loop.run_until_complete(
            protocol.call_find_node(("127.0.0.1", peer1_port), key)
        )
        assert recv_id == peer2_id and recv_endpoint == (
            "127.0.0.1",
            peer2_port,
        ), f"expected id={peer2_id}, port={('127.0.0.1', peer2_port)} but got {recv_id}, {recv_endpoint}"

        # peer 2 must know about peer 1
        ((recv_id, recv_endpoint),) = loop.run_until_complete(
            protocol.call_find_node(("127.0.0.1", peer2_port), key)
        )
        assert recv_id == peer1_id and recv_endpoint == (
            "127.0.0.1",
            peer1_port,
        ), f"expected id={peer1_id}, port={('127.0.0.1', peer1_port)} but got {recv_id}, {recv_endpoint}"

        recv_value, recv_expiration, recv_peers = loop.run_until_complete(
            protocol.call_find_value(("127.0.0.1", peer1_port), key)
        )
        assert recv_value == value and recv_expiration == expiration, (
            "call_find_value expected "
            f"{value} (expires by {expiration}) but got {recv_value} (expires by {recv_expiration})"
        )
        assert (
            recv_peers == nodes_found
        ), "call_find_value must return the same peers as call_find_node"
        print("Kademlia test finished sucessfully!")

    finally:
        peer1_proc.terminate()
        peer2_proc.terminate()


def dht_peer_func(
    port: int,
    dhtid: DHTID,
    started: mp.synchronize.Event,
    ping: Optional[hivemind.Endpoint] = None,
):
    loop = asyncio.new_event_loop()
    protocol = partial(KademliaProtocol, dhtid, 20, 5, 300, 5)
    listen = loop.create_datagram_endpoint(protocol, local_addr=("127.0.0.1", port))
    transport, protocol = loop.run_until_complete(listen)
    print(f"Started peer id={protocol.node_id} port={port}", flush=True)

    if ping is not None:
        loop.run_until_complete(protocol.call_ping(ping))
    started.set()
    loop.run_forever()
    print(f"Finished peer id={protocol.node_id} port={port}", flush=True)
