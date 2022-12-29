import time
from functools import partial

import pytest

import hivemind


async def ping_to_client(dht, node, peer_id: str):
    return await node.protocol.call_ping(hivemind.PeerID.from_base58(str(peer_id)))


@pytest.mark.forked
@pytest.mark.parametrize(
    "use_auto_relay,use_relay",
    [
        (True, True),
        (False, False),
    ],
)
def test_autorelay(use_auto_relay: bool, use_relay: bool):
    dht_first_peer = hivemind.DHT(
        start=True,
        use_auto_relay=use_auto_relay,
        use_relay=use_relay,
        force_reachability="public",
    )
    dht_first_peer_id = dht_first_peer.peer_id
    initial_peers = dht_first_peer.get_visible_maddrs()
    assert dht_first_peer_id is not None

    dht_third_peer = hivemind.DHT(
        initial_peers=initial_peers,
        host_maddrs=[],
        start=True,
        no_listen=True,
        use_relay=use_relay,
        client_mode=False,
        use_auto_relay=use_auto_relay,
    )
    time.sleep(5)
    dht_second_peer = hivemind.DHT(
        initial_peers=initial_peers,
        start=True,
        client_mode=False,
        no_listen=False,
        use_relay=use_relay,
        use_auto_relay=use_auto_relay,
    )

    assert dht_first_peer.is_alive() and dht_second_peer.is_alive() and dht_third_peer.is_alive()

    time_start = time.perf_counter()
    while time.perf_counter() - time_start < 30:
        reached_ip = dht_second_peer.run_coroutine(partial(ping_to_client, peer_id=dht_third_peer.peer_id))
        if reached_ip:
            assert use_relay
            break
        time.sleep(2)
    else:
        assert not use_relay

    for peer in dht_first_peer, dht_second_peer, dht_third_peer:
        peer.shutdown()
