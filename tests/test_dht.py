import asyncio
import concurrent.futures
import random
import time

import pytest
from multiaddr import Multiaddr

import hivemind

from test_utils.dht_swarms import launch_dht_instances
from test_utils.networking import get_free_port


@pytest.mark.asyncio
async def test_startup_error():
    with pytest.raises(hivemind.p2p.P2PDaemonError, match=r"(?i)Failed to connect to bootstrap peers"):
        hivemind.DHT(
            initial_peers=[f"/ip4/127.0.0.1/tcp/{get_free_port()}/p2p/QmdaK4LUeQaKhqSFPRu9N7MvXUEWDxWwtCvPrS444tCgd1"],
            start=True,
        )

    dht = hivemind.DHT(start=True, await_ready=False)
    with pytest.raises(concurrent.futures.TimeoutError):
        dht.wait_until_ready(timeout=0.01)
    dht.shutdown()


@pytest.mark.forked
def test_get_store(n_peers=10):
    peers = launch_dht_instances(n_peers)

    node1, node2 = random.sample(peers, 2)
    assert node1.store("key1", "value1", expiration_time=hivemind.get_dht_time() + 30)
    assert node1.get("key1").value == "value1"
    assert node2.get("key1").value == "value1"
    assert node2.get("key2") is None

    future = node1.get("foo", return_future=True)
    assert future.result() is None

    future = node1.get("foo", return_future=True)
    future.cancel()

    assert node2.store("key1", 123, expiration_time=hivemind.get_dht_time() + 31)
    assert node2.store("key2", 456, expiration_time=hivemind.get_dht_time() + 32)
    assert node1.get("key1", latest=True).value == 123
    assert node1.get("key2").value == 456

    assert node1.store("key2", subkey="subkey1", value=789, expiration_time=hivemind.get_dht_time() + 32)
    assert node2.store("key2", subkey="subkey2", value="pew", expiration_time=hivemind.get_dht_time() + 32)
    found_dict = node1.get("key2", latest=True).value
    assert isinstance(found_dict, dict) and len(found_dict) == 2
    assert found_dict["subkey1"].value == 789 and found_dict["subkey2"].value == "pew"

    for peer in peers:
        peer.shutdown()


async def dummy_dht_coro(self, node):
    return "pew"


async def dummy_dht_coro_error(self, node):
    raise ValueError("Oops, i did it again...")


async def dummy_dht_coro_stateful(self, node):
    self._x_dummy = getattr(self, "_x_dummy", 123) + 1
    return self._x_dummy


async def dummy_dht_coro_long(self, node):
    await asyncio.sleep(0.25)
    return self._x_dummy**2


async def dummy_dht_coro_for_cancel(self, node):
    self._x_dummy = -100
    await asyncio.sleep(0.5)
    self._x_dummy = 999


@pytest.mark.forked
def test_run_coroutine():
    dht = hivemind.DHT(start=True)
    assert dht.run_coroutine(dummy_dht_coro) == "pew"

    with pytest.raises(ValueError):
        res = dht.run_coroutine(dummy_dht_coro_error)

    bg_task = dht.run_coroutine(dummy_dht_coro_long, return_future=True)
    assert dht.run_coroutine(dummy_dht_coro_stateful) == 124
    assert dht.run_coroutine(dummy_dht_coro_stateful) == 125
    assert dht.run_coroutine(dummy_dht_coro_stateful) == 126
    assert not hasattr(dht, "_x_dummy")
    assert bg_task.result() == 126**2

    future = dht.run_coroutine(dummy_dht_coro_for_cancel, return_future=True)
    time.sleep(0.25)
    future.cancel()
    assert dht.run_coroutine(dummy_dht_coro_stateful) == -99

    dht.shutdown()


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dht_get_visible_maddrs():
    # test 1: IPv4 localhost multiaddr is visible by default

    dht = hivemind.DHT(start=True)

    assert any(str(maddr).startswith("/ip4/127.0.0.1") for maddr in dht.get_visible_maddrs())
    dht.shutdown()

    # test 2: announce_maddrs are the single visible multiaddrs if defined

    dummy_endpoint = Multiaddr("/ip4/123.45.67.89/tcp/31337")
    p2p = await hivemind.p2p.P2P.create(announce_maddrs=[dummy_endpoint])
    dht = hivemind.DHT(start=True, p2p=p2p)

    assert dht.get_visible_maddrs() == [dummy_endpoint.encapsulate(f"/p2p/{p2p.peer_id}")]
    dht.shutdown()
