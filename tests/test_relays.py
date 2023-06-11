import time
from functools import partial

import pytest

import hivemind


async def ping_to_client(dht, node, peer_id: hivemind.p2p.PeerID):
    return await node.protocol.call_ping(peer_id)


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
    dht_second_peer = hivemind.DHT(
        initial_peers=initial_peers,
        start=True,
        client_mode=False,
        no_listen=False,
        use_relay=use_relay,
        use_auto_relay=use_auto_relay,
    )

    assert dht_first_peer.is_alive() and dht_second_peer.is_alive() and dht_third_peer.is_alive()

    reached_ip = dht_second_peer.run_coroutine(partial(ping_to_client, peer_id=dht_third_peer.peer_id))
    if reached_ip:
        assert use_relay

    for peer in dht_first_peer, dht_second_peer, dht_third_peer:
        peer.shutdown()
