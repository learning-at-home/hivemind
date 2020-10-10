import time

from hivemind.dht.routing import get_dht_time
from hivemind.dht.storage import DHTLocalStorage, DHTID, DictionaryDHTValue
from hivemind.utils.serializer import MSGPackSerializer


def test_store():
    d = DHTLocalStorage()
    d.store(DHTID.generate("key"), b"val", get_dht_time() + 0.5)
    assert d.get(DHTID.generate("key"))[0] == b"val", "Wrong value"
    print("Test store passed")


def test_get_expired():
    d = DHTLocalStorage()
    d.store(DHTID.generate("key"), b"val", get_dht_time() + 0.1)
    time.sleep(0.5)
    assert d.get(DHTID.generate("key")) == (None, None), "Expired value must be deleted"
    print("Test get expired passed")


def test_get_empty():
    d = DHTLocalStorage()
    assert d.get(DHTID.generate(source="key")) == (None, None), "DHTLocalStorage returned non-existent value"
    print("Test get expired passed")


def test_change_expiration_time():
    d = DHTLocalStorage()
    d.store(DHTID.generate("key"), b"val1", get_dht_time() + 1)
    assert d.get(DHTID.generate("key"))[0] == b"val1", "Wrong value"
    d.store(DHTID.generate("key"), b"val2", get_dht_time() + 200)
    time.sleep(1)
    assert d.get(DHTID.generate("key"))[0] == b"val2", "Value must be changed, but still kept in table"
    print("Test change expiration time passed")


def test_maxsize_cache():
    d = DHTLocalStorage(maxsize=1)
    d.store(DHTID.generate("key1"), b"val1", get_dht_time() + 1)
    d.store(DHTID.generate("key2"), b"val2", get_dht_time() + 200)
    assert d.get(DHTID.generate("key2"))[0] == b"val2", "Value with bigger exp. time must be kept"
    assert d.get(DHTID.generate("key1"))[0] is None, "Value with less exp time, must be deleted"


def test_localstorage_top():
    d = DHTLocalStorage(maxsize=3)
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
    d1 = DHTLocalStorage()
    d2 = DictionaryDHTValue()
    d2.store('subkey1', b'value1', time + 2)
    d2.store('subkey2', b'value2', time + 3)
    d2.store('subkey3', b'value3', time + 1)

    assert d2.latest_expiration_time == time + 3
    for subkey, subvalue, subexpiration in d2.items():
        assert d1.store_subkey(DHTID.generate('foo'), subkey, subvalue, subexpiration)
    assert d1.store(DHTID.generate('bar'), b'456', time + 2)
    assert d1.get(DHTID.generate('foo'))[0].data == d2.data
    assert d1.get(DHTID.generate('foo'))[1] == d2.latest_expiration_time
    assert d1.get(DHTID.generate('foo'))[0].get('subkey1') == (b'value1', time + 2)
    assert len(d1.get(DHTID.generate('foo'))[0]) == 3
    assert d1.store_subkey(DHTID.generate('foo'), 'subkey4', b'value4', time + 4)
    assert len(d1.get(DHTID.generate('foo'))[0]) == 4

    assert d1.store_subkey(DHTID.generate('bar'), 'subkeyA', b'valueA', time + 1) is False  # prev has better expiration
    assert d1.store_subkey(DHTID.generate('bar'), 'subkeyA', b'valueA', time + 3)  # new value has better expiration
    assert d1.store_subkey(DHTID.generate('bar'), 'subkeyB', b'valueB', time + 4)  # new value has better expiration
    assert d1.store_subkey(DHTID.generate('bar'), 'subkeyA', b'valueA+', time + 5)  # overwrite subkeyA under key bar
    assert all(subkey in d1.get(DHTID.generate('bar'))[0] for subkey in ('subkeyA', 'subkeyB'))
    assert len(d1.get(DHTID.generate('bar'))[0]) == 2 and d1.get(DHTID.generate('bar'))[1] == time + 5

    assert d1.store(DHTID.generate('foo'), b'nothing', time + 3.5) is False  # previous value has better expiration
    assert d1.get(DHTID.generate('foo'))[0].get('subkey2') == (b'value2', time + 3)
    assert d1.store(DHTID.generate('foo'), b'nothing', time + 5) is True  # new value has better expiraiton
    assert d1.get(DHTID.generate('foo')) == (b'nothing', time + 5)  # value should be replaced


def test_localstorage_freeze():
    d = DHTLocalStorage(maxsize=2)

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


def test_localstorage_serialize():
    d1 = DictionaryDHTValue()
    d2 = DictionaryDHTValue()

    now = get_dht_time()
    d1.store('key1', b'ololo', now + 1)
    d2.store('key2', b'pysh', now + 1)
    d2.store('key3', b'pyshpysh', now + 2)

    data = MSGPackSerializer.dumps([d1, d2, 123321])
    assert isinstance(data, bytes)
    new_d1, new_d2, new_value = MSGPackSerializer.loads(data)
    assert isinstance(new_d1, DictionaryDHTValue) and isinstance(new_d2, DictionaryDHTValue) and new_value == 123321
    assert 'key1' in new_d1 and len(new_d1) == 1
    assert 'key1' not in new_d2 and len(new_d2) == 2 and new_d2.get('key3') == (b'pyshpysh', now + 2)
