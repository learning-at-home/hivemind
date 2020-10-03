import time

from hivemind import DHTID, get_dht_time
from hivemind.dht.protocol import ExpirableStorage


def test_store():
    d = ExpirableStorage()
    d.store(DHTID.generate("key"), b"val", get_dht_time() + 0.5)
    assert d.get(DHTID.generate("key"))[0] == b"val", "Wrong value"
    print("Test store passed")


def test_get_expired():
    d = ExpirableStorage()
    d.store(DHTID.generate("key"), b"val", get_dht_time() + 0.1)
    time.sleep(0.5)
    assert d.get(DHTID.generate("key")) == (None, None), "Expired value must be deleted"
    print("Test get expired passed")


def test_get_empty():
    d = ExpirableStorage()
    assert d.get(DHTID.generate(source="key")) == (None, None), "ExpirableStorage returned non-existent value"
    print("Test get expired passed")


def test_change_expiration_time():
    d = ExpirableStorage()
    d.store(DHTID.generate("key"), b"val1", get_dht_time() + 1)
    assert d.get(DHTID.generate("key"))[0] == b"val1", "Wrong value"
    d.store(DHTID.generate("key"), b"val2", get_dht_time() + 200)
    time.sleep(1)
    assert d.get(DHTID.generate("key"))[0] == b"val2", "Value must be changed, but still kept in table"
    print("Test change expiration time passed")


def test_maxsize_cache():
    d = ExpirableStorage(maxsize=1)
    d.store(DHTID.generate("key1"), b"val1", get_dht_time() + 1)
    d.store(DHTID.generate("key2"), b"val2", get_dht_time() + 200)
    assert d.get(DHTID.generate("key2"))[0] == b"val2", "Value with bigger exp. time must be kept"
    assert d.get(DHTID.generate("key1"))[0] is None, "Value with less exp time, must be deleted"


def test_localstorage_top():
    d = ExpirableStorage(maxsize=3)
    d.store(DHTID.generate("key1"), b"val1", get_dht_time() + 1)
    d.store(DHTID.generate("key2"), b"val2", get_dht_time() + 2)
    d.store(DHTID.generate("key3"), b"val3", get_dht_time() + 4)
    assert d.top()[:2] == (DHTID.generate("key1"), b"val1")

    d.store(DHTID.generate("key1"), b"val1_new", get_dht_time() + 3)
    assert d.top()[:2] == (DHTID.generate("key2"), b"val2")

    del d[DHTID.generate('key2')]
    assert d.top()[:2] == (DHTID.generate("key1"), b"val1_new")
    d.store(DHTID.generate("key2"), b"val2_new", get_dht_time() + 5)
    d.store(DHTID.generate("key4"), b"val4", get_dht_time() + 6)  # key4 will push out key1 due to maxsize

    assert d.top()[:2] == (DHTID.generate("key3"), b"val3")


def test_localstorage_nested():
    time = get_dht_time()
    d1 = ExpirableStorage()
    d2 = ExpirableStorage()
    d2.store(DHTID.generate('subkey1'), b'value1', time + 2)
    d2.store(DHTID.generate('subkey2'), b'value2', time + 3)
    d2.store(DHTID.generate('subkey3'), b'value3', time + 1)

    assert d2.latest_expiration_time == time + 3
    assert d1.store(DHTID.generate('foo'), d2, d2.latest_expiration_time)
    assert d1.store(DHTID.generate('bar'), b'456', time + 2)
    assert d1.get(DHTID.generate('foo')) == (d2, d2.latest_expiration_time)
    assert d1.get(DHTID.generate('foo'))[0].get(DHTID.generate('subkey1')) == (b'value1', time + 2)
    assert d1.store(DHTID.generate('foo'), b'nothing', time + 1) is False  # previous value has better expiration
    assert d1.get(DHTID.generate('foo'))[0].get(DHTID.generate('subkey2')) == (b'value2', time + 3)
    assert d1.store(DHTID.generate('foo'), b'nothing', time + 4) is True   # new value has better expiraiton
    assert d1.get(DHTID.generate('foo')) == (b'nothing', time + 4)         # value should be replaced


def test_localstorage_freeze():
    d = ExpirableStorage(maxsize=2)

    with d.freeze():
        d.store(DHTID.generate("key1"), b"val1", get_dht_time() + 0.01)
        assert DHTID.generate("key1") in d
        time.sleep(0.03)
        assert DHTID.generate("key1") in d
    assert DHTID.generate("key1") not in d

    with d.freeze():
        d.store(DHTID.generate("key1"), b"val1", get_dht_time() + 1)
        d.store(DHTID.generate("key2"), b"val2", get_dht_time() + 2)
        d.store(DHTID.generate("key3"), b"val3", get_dht_time() + 3)  # key3 will push key1 out due to maxsize
        assert DHTID.generate("key1") in d
    assert DHTID.generate("key1") not in d
