from hivemind.dht.protocol import LocalStorage
from time import sleep, monotonic


def test_store():
    d = LocalStorage()
    d.store("key", "val", monotonic()+10)
    assert d.get("key")[0] == "val", "Wrong value"
    print("Test store passed")


def test_get_expired():
    d = LocalStorage(keep_expired=False)
    d.store("key", "val", monotonic()+1)
    sleep(2)
    assert d.get("key") == (None, None), "Expired value must be deleted"
    print("Test get expired passed")


def test_store_maxsize():
    d = LocalStorage(maxsize=1)
    d.store("key1", "val1", monotonic() + 1)
    d.store("key2", "val2", monotonic() + 2)
    assert d.get("key1") == (None, None), "elder a value must be deleted"
    assert d.get("key2")[0] == "val2", "Newer should be stored"
    print("Test store maxsize passed")


test_get_expired()
test_store()
test_store_maxsize()