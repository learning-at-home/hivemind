import asyncio
import multiprocessing as mp
import random
import signal
from typing import Sequence, Tuple, List

import pytest
from multiaddr import Multiaddr

import hivemind
from hivemind import P2P, PeerID, get_dht_time, get_logger
from hivemind.dht import DHTID
from hivemind.dht.protocol import DHTProtocol
from hivemind.dht.storage import DictionaryDHTValue


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

    maddr_conn.send((p2p.peer_id, visible_maddrs))

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


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dht_protocol():
    peer1_node_id, peer1_proc, peer1_id, peer1_maddrs = launch_protocol_listener()
    peer2_node_id, peer2_proc, peer2_id, _ = launch_protocol_listener(initial_peers=peer1_maddrs)

    for client_mode in [True, False]:  # note: order matters, this test assumes that first run uses client mode
        peer_id = DHTID.generate()
        p2p = await P2P.create(initial_peers=peer1_maddrs)
        protocol = await DHTProtocol.create(
            p2p, peer_id, bucket_size=20, depth_modulo=5, wait_timeout=5, num_replicas=3, client_mode=client_mode
        )
        logger.info(f"Self id={protocol.node_id}")

        assert (await protocol.call_ping(peer1_id)) == peer1_node_id

        key, value, expiration = DHTID.generate(), [random.random(), {"ololo": "pyshpysh"}], get_dht_time() + 1e3
        store_ok = await protocol.call_store(peer1_id, [key], [hivemind.MSGPackSerializer.dumps(value)], expiration)
        assert all(store_ok), "DHT rejected a trivial store"

        # peer 1 must know about peer 2
        (recv_value_bytes, recv_expiration), nodes_found = (await protocol.call_find(peer1_id, [key]))[key]
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
        empty_item, nodes_found_2 = (await protocol.call_find(peer2_id, [dummy_key]))[dummy_key]
        assert empty_item is None, "Non-existent keys shouldn't have values"
        (recv_id, recv_peer_id) = next(iter(nodes_found_2.items()))
        assert (
            recv_id == peer1_node_id and recv_peer_id == peer1_id
        ), f"expected id={peer1_node_id}, peer={peer1_id} but got {recv_id}, {recv_peer_id}"

        # cause a non-response by querying a nonexistent peer
        assert (await protocol.call_find(PeerID.from_base58("fakeid"), [key])) is None

        # store/get a dictionary with sub-keys
        nested_key, subkey1, subkey2 = DHTID.generate(), "foo", "bar"
        value1, value2 = [random.random(), {"ololo": "pyshpysh"}], "abacaba"
        assert await protocol.call_store(
            peer1_id,
            keys=[nested_key],
            values=[hivemind.MSGPackSerializer.dumps(value1)],
            expiration_time=[expiration],
            subkeys=[subkey1],
        )
        assert await protocol.call_store(
            peer1_id,
            keys=[nested_key],
            values=[hivemind.MSGPackSerializer.dumps(value2)],
            expiration_time=[expiration + 5],
            subkeys=[subkey2],
        )
        (recv_dict, recv_expiration), nodes_found = (await protocol.call_find(peer1_id, [nested_key]))[nested_key]
        assert isinstance(recv_dict, DictionaryDHTValue)
        assert len(recv_dict.data) == 2 and recv_expiration == expiration + 5
        assert recv_dict.data[subkey1] == (protocol.serializer.dumps(value1), expiration)
        assert recv_dict.data[subkey2] == (protocol.serializer.dumps(value2), expiration + 5)

        if not client_mode:
            await p2p.shutdown()

    peer1_proc.terminate()
    peer2_proc.terminate()


@pytest.mark.forked
@pytest.mark.forked
def test_empty_table():
    """Test RPC methods with empty routing table"""
    peer_id, peer_proc, peer_peer_id, peer_maddrs = launch_protocol_listener()

    p2p = await P2P.create(initial_peers=peer_maddrs)
    protocol = await DHTProtocol.create(
        p2p, DHTID.generate(), bucket_size=20, depth_modulo=5, wait_timeout=5, num_replicas=3, client_mode=True
    )

    key, value, expiration = DHTID.generate(), [random.random(), {"ololo": "pyshpysh"}], get_dht_time() + 1e3

    empty_item, nodes_found = (await protocol.call_find(peer_peer_id, [key]))[key]
    assert empty_item is None and len(nodes_found) == 0
    assert all(await protocol.call_store(peer_peer_id, [key], [hivemind.MSGPackSerializer.dumps(value)], expiration))

    (recv_value_bytes, recv_expiration), nodes_found = (await protocol.call_find(peer_peer_id, [key]))[key]
    recv_value = hivemind.MSGPackSerializer.loads(recv_value_bytes)
    assert len(nodes_found) == 0
    assert recv_value == value and recv_expiration == expiration

    assert (await protocol.call_ping(peer_peer_id)) == peer_id
    assert (await protocol.call_ping(PeerID.from_base58("fakeid"))) is None
    peer_proc.terminate()
