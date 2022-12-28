import time
from functools import partial

import pytest

import hivemind


async def ping_to_client(dht, node, peer_id: str):
    return await node.protocol.call_ping(hivemind.PeerID.from_base58(str(peer_id)))


@pytest.mark.parametrize(
    "use_auto_relay,use_relay",
    [
        (True, True),
        (False, False),
    ],
)
def test_client_pinging(use_auto_relay: bool, use_relay: bool):
    dht_first_peer = hivemind.DHT(
        host_maddrs=["/ip4/0.0.0.0/tcp/31330"],
        start=True,
        use_auto_relay=True,
        use_relay=True,
        force_reachability="public",
    )
    dht_first_peer_id = dht_first_peer.peer_id
    initial_peers = dht_first_peer.get_visible_maddrs()
    assert dht_first_peer_id is not None

    dht_second_peer = hivemind.DHT(
        initial_peers=initial_peers,
        host_maddrs=["/ip4/0.0.0.0/tcp/0"],
        start=True,
        client_mode=False,
        no_listen=True,
        use_relay=use_relay,
        use_auto_relay=use_auto_relay,
    )

    dht_third_peer = hivemind.DHT(
        initial_peers=initial_peers,
        host_maddrs=["/ip4/0.0.0.0/tcp/0"],
        start=True,
        client_mode=False,
        no_listen=True,
        use_relay=use_relay,
        use_auto_relay=use_auto_relay,
    )

    time.sleep(4*60)
    assert dht_first_peer.is_alive() and dht_second_peer.is_alive() and dht_third_peer.is_alive()

    reached_ip = dht_second_peer.run_coroutine(partial(ping_to_client, peer_id=dht_third_peer.peer_id))


    if use_auto_relay and use_relay:
        assert reached_ip is not None
    else:
        assert reached_ip is None
