from __future__ import annotations
import heapq
from contextlib import contextmanager
from typing import Generic, Optional, Dict, Tuple, List, Iterator, TypeVar, Union
from hivemind.dht.routing import DHTID, DHTExpiration, get_dht_time, BinaryDHTValue, Subkey

KeyType = TypeVar('KeyType')
ValueType = TypeVar('ValueType')


class ExpirableStorage(Generic[KeyType, ValueType]):
    """ A dictionary that maintains up to :maxsize: key-value-expiration tuples until their expiration_time """

    def __init__(self, maxsize: Optional[int] = None):
        self.maxsize = maxsize or float("inf")
        self.data: Dict[KeyType, Tuple[ValueType, DHTExpiration]] = dict()
        self.expiration_heap: List[Tuple[DHTExpiration, KeyType]] = []
        self.key_to_heap: Dict[KeyType, Tuple[DHTExpiration, KeyType]] = dict()
        self.frozen = False  # if True, do not remove outdated elements

    def _remove_outdated(self):
        while not self.frozen and self.expiration_heap and (self.expiration_heap[0][0] < get_dht_time()
                                                            or len(self.expiration_heap) > self.maxsize):
            heap_entry = heapq.heappop(self.expiration_heap)
            key = heap_entry[1]
            if self.key_to_heap.get(key) == heap_entry:
                del self.data[key], self.key_to_heap[key]

    def store(self, key: KeyType, value: ValueType, expiration_time: DHTExpiration) -> bool:
        """
        Store a (key, value) pair locally at least until expiration_time. See class docstring for details.
        :returns: True if new value was stored, False it was rejected (current value is newer)
        """
        if expiration_time < get_dht_time() and not self.frozen:
            return False
        self.key_to_heap[key] = (expiration_time, key)
        heapq.heappush(self.expiration_heap, (expiration_time, key))
        if key in self.data:
            if self.data[key][1] < expiration_time:
                self.data[key] = (value, expiration_time)
                return True
            return False
        self.data[key] = (value, expiration_time)
        self._remove_outdated()
        return True

    def get(self, key: KeyType) -> (Optional[ValueType], Optional[DHTExpiration]):
        """ Get a value corresponding to a key if that (key, value) pair was previously stored here. """
        self._remove_outdated()
        if key in self.data:
            return self.data[key]
        return None, None

    def items(self) -> Iterator[Tuple[KeyType, ValueType, DHTExpiration]]:
        """ Iterate over (key, value, expiration_time) tuples stored in this storage """
        self._remove_outdated()
        return ((key, value, expiration_time) for key, (value, expiration_time) in self.data.items())

    def top(self) -> Optional[Tuple[KeyType, ValueType, DHTExpiration]]:
        """ Return the entry with earliest expiration or None if there isn't any """
        self._remove_outdated()
        if self.data:
            top_entry, top_key = self.expiration_heap[0], self.expiration_heap[0][1]
            while self.key_to_heap.get(top_key) != top_entry:
                heapq.heappop(self.expiration_heap)  # skip leftover "ghost" entries until first real entry
                top_entry, top_key = self.expiration_heap[0], self.expiration_heap[0][1]
            value, expiration = self.data[top_key]
            return top_key, value, expiration

    def __contains__(self, key: KeyType):
        self._remove_outdated()
        return key in self.data

    def __len__(self):
        self._remove_outdated()
        return len(self.data)

    def __delitem__(self, key: KeyType):
        if key in self.key_to_heap:
            del self.data[key], self.key_to_heap[key]
        # note: key may still be in self.expiration_heap, but it will not be used and eventually ._remove_outdated()

    def __bool__(self):
        return bool(self.data)

    @contextmanager
    def freeze(self):
        """ Temporarily cease to ._remove_outdated() elements inside this context to ensure consistency """
        prev_frozen, self.frozen = self.frozen, True
        try:
            yield self
        finally:
            self.frozen = prev_frozen


class DictionaryDHTValue(ExpirableStorage[Subkey, BinaryDHTValue]):
    """ a dictionary-like DHT value type that maps sub-keys to values with individual expirations """
    latest_expiration_time = float('-inf')

    def store(self, key: KeyType, value: ValueType, expiration_time: DHTExpiration) -> bool:
        self.latest_expiration_time = max(self.latest_expiration_time, expiration_time)
        return super().store(key, value, expiration_time)


class DHTLocalStorage(ExpirableStorage[DHTID, Union[BinaryDHTValue, DictionaryDHTValue]]):
    """ A dictionary-like storage that can store binary values and/or nested dictionaries until expiration """
    def store_subkey(self, key: DHTID, subkey: Subkey, value: BinaryDHTValue, expiration_time: DHTExpiration) -> bool:
        """
        Save a (sub-key, value) into a dictionary associated with a given key.
         1) if self[key] is empty, create a new dictionary and add sub-key there
         2) if self[key] is a dictionary (DictionaryDHTValue), store {sub-key: value, expiration} to that storage
         3) if self[key] is a normal value with smaller expiration time, overwrite it with a dictionary and add sub-key
        :returns: True if new entry was stored, False it was rejected (current value is newer)
        """
        previous_value, previous_expiration_time = self.get(key)
        if isinstance(previous_value, DictionaryDHTValue):  # already a dictionary, just add new subkey
            if expiration_time > previous_value.latest_expiration_time:
                self.store(key, previous_value, expiration_time)  # refresh expiration time
            return previous_value.store(subkey, value, expiration_time)
        elif expiration_time > (previous_expiration_time or float('-inf')):  # create new dictionary, add subkey
            new_storage = DictionaryDHTValue()
            new_storage.store(subkey, value, expiration_time)
            return self.store(key, new_storage, new_storage.latest_expiration_time)
        else:
            return False


class CacheRefreshQueue(ExpirableStorage[DHTID, DHTExpiration]):
    """ a queue of keys scheduled for refresh in future, used in DHTNode """
