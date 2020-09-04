import time

from hivemind import DHTID, get_dht_time
from hivemind.dht.protocol import LocalStorage


def test_store():
    d = LocalStorage()
    d.store(DHTID.generate("key"), b"val", get_dht_time() + 0.5)
    assert d.get(DHTID.generate("key"))[0] == b"val", "Wrong value"
    print("Test store passed")


def test_get_expired():
    d = LocalStorage()
    d.store(DHTID.generate("key"), b"val", get_dht_time() + 0.1)
    time.sleep(0.5)
    assert d.get(DHTID.generate("key")) == (None, None), "Expired value must be deleted"
    print("Test get expired passed")


def test_get_empty():
    d = LocalStorage()
    assert d.get(DHTID.generate(source="key")) == (None, None), "LocalStorage returned non-existent value"
    print("Test get expired passed")


def test_change_expiration_time():
    d = LocalStorage()
    d.store(DHTID.generate("key"), b"val1", get_dht_time() + 1)
    assert d.get(DHTID.generate("key"))[0] == b"val1", "Wrong value"
    d.store(DHTID.generate("key"), b"val2", get_dht_time() + 200)
    time.sleep(1)
    assert d.get(DHTID.generate("key"))[0] == b"val2", "Value must be changed, but still kept in table"
    print("Test change expiration time passed")


def test_maxsize_cache():
    d = LocalStorage(maxsize=1)
    d.store(DHTID.generate("key1"), b"val1", get_dht_time() + 1)
    d.store(DHTID.generate("key2"), b"val2", get_dht_time() + 200)
    assert d.get(DHTID.generate("key2"))[0] == b"val2", "Value with bigger exp. time must be kept"
    assert d.get(DHTID.generate("key1"))[0] is None, "Value with less exp time, must be deleted"

def test_localstorage_top():
    d = LocalStorage(maxsize=3)
    d.store(DHTID.generate("key1"), b"val1", get_dht_time() + 1)
    d.store(DHTID.generate("key2"), b"val2", get_dht_time() + 2)
    d.store(DHTID.generate("key3"), b"val3", get_dht_time() + 4)
    assert d.top()[:2] == (DHTID.generate("key1"), b"val1")

    d.store(DHTID.generate("key1"), b"val1_new", get_dht_time() + 3)
    assert d.top()[:2] == (DHTID.generate("key2"), b"val2")

    del d[DHTID.generate('key2')]
    assert d.top()[:2] == (DHTID.generate("key1"), b"val1_new")
    d.store(DHTID.generate("key2"), b"val2_new", get_dht_time() + 5)
    d.store(DHTID.generate("key4"), b"val3", get_dht_time() + 6)
    assert d.top()[:2] == (DHTID.generate("key3"), b"val3")

