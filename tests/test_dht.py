import asyncio
import random
import time

import pytest

import hivemind
from hivemind import LOCALHOST, strip_port


@pytest.mark.forked
def test_get_store():
    peers = []
    for i in range(10):
        neighbors_i = [f'{LOCALHOST}:{node.port}' for node in random.sample(peers, min(3, len(peers)))]
        peers.append(hivemind.DHT(initial_peers=neighbors_i, start=True))

    node1, node2 = random.sample(peers, 2)
    assert node1.store('key1', 'value1', expiration_time=hivemind.get_dht_time() + 30)
    assert node1.get('key1').value == 'value1'
    assert node2.get('key1').value == 'value1'
    assert node2.get('key2') is None

    future = node1.get('foo', return_future=True)
    assert future.result() is None

    future = node1.get('foo', return_future=True)
    future.cancel()

    assert node2.store('key1', 123, expiration_time=hivemind.get_dht_time() + 31)
    assert node2.store('key2', 456, expiration_time=hivemind.get_dht_time() + 32)
    assert node1.get('key1', latest=True).value == 123
    assert node1.get('key2').value == 456

    assert node1.store('key2', subkey='subkey1', value=789, expiration_time=hivemind.get_dht_time() + 32)
    assert node2.store('key2', subkey='subkey2', value='pew', expiration_time=hivemind.get_dht_time() + 32)
    found_dict = node1.get('key2', latest=True).value
    assert isinstance(found_dict, dict) and len(found_dict) == 2
    assert found_dict['subkey1'].value == 789 and found_dict['subkey2'].value == 'pew'

    for peer in peers:
        peer.shutdown()


async def dummy_dht_coro(self, node):
    return 'pew'


async def dummy_dht_coro_error(self, node):
    raise ValueError("Oops, i did it again...")


async def dummy_dht_coro_stateful(self, node):
    self._x_dummy = getattr(self, '_x_dummy', 123) + 1
    return self._x_dummy


async def dummy_dht_coro_long(self, node):
    await asyncio.sleep(0.25)
    return self._x_dummy ** 2


async def dummy_dht_coro_for_cancel(self, node):
    self._x_dummy = -100
    await asyncio.sleep(0.5)
    self._x_dummy = 999


@pytest.mark.forked
def test_run_coroutine():
    dht = hivemind.DHT(start=True)
    assert dht.run_coroutine(dummy_dht_coro) == 'pew'

    with pytest.raises(ValueError):
        res = dht.run_coroutine(dummy_dht_coro_error)

    bg_task = dht.run_coroutine(dummy_dht_coro_long, return_future=True)
    assert dht.run_coroutine(dummy_dht_coro_stateful) == 124
    assert dht.run_coroutine(dummy_dht_coro_stateful) == 125
    assert dht.run_coroutine(dummy_dht_coro_stateful) == 126
    assert not hasattr(dht, '_x_dummy')
    assert bg_task.result() == 126 ** 2

    future = dht.run_coroutine(dummy_dht_coro_for_cancel, return_future=True)
    time.sleep(0.25)
    future.cancel()
    assert dht.run_coroutine(dummy_dht_coro_stateful) == -99


@pytest.mark.forked
def test_dht_get_address(addr=LOCALHOST, dummy_endpoint='123.45.67.89:*'):
    node1 = hivemind.DHT(start=True, listen_on=f"0.0.0.0:*")
    node2 = hivemind.DHT(start=True, listen_on=f"0.0.0.0:*", initial_peers=[f"{addr}:{node1.port}"])
    node3 = hivemind.DHT(start=True, listen_on=f"0.0.0.0:*", initial_peers=[f"{addr}:{node2.port}"])
    assert addr in node3.get_visible_address(num_peers=2)

    node4 = hivemind.DHT(start=True, listen_on=f"0.0.0.0:*")
    with pytest.raises(ValueError):
        node4.get_visible_address()
    assert node4.get_visible_address(peers=[f'{addr}:{node1.port}']).endswith(addr)

    node5 = hivemind.DHT(start=True, listen_on=f"0.0.0.0:*", endpoint=f"{dummy_endpoint}")
    assert node5.get_visible_address() == strip_port(dummy_endpoint)
