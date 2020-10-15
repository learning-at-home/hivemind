from __future__ import annotations
import heapq
from contextlib import contextmanager
from typing import Generic, Optional, Dict, Tuple, List, Iterator, TypeVar, Union, NamedTuple

from hivemind.dht.routing import DHTID, DHTExpiration, get_dht_time, BinaryDHTValue, Subkey
from hivemind.utils.serializer import MSGPackSerializer

KeyType = TypeVar('KeyType')
ValueType = TypeVar('ValueType')
ROOT = 0


class ValueWithExpiration(NamedTuple, Generic[ValueType]):
    value: ValueType
    expiration_time: DHTExpiration


class HeapEntry(NamedTuple, Generic[KeyType]):
    expiration_time: DHTExpiration
    key: KeyType


class TimedStorage(Generic[KeyType, ValueType]):
    """ A dictionary that maintains up to :maxsize: key-value-expiration tuples until their expiration_time """
    frozen = False  # can be set to True. If true, do not remove outdated elements

    def __init__(self, maxsize: Optional[int] = None):
        self.maxsize = maxsize or float("inf")
        self.data: Dict[KeyType, ValueWithExpiration[ValueType]] = dict()
        self.expiration_heap: List[HeapEntry[KeyType]] = []
        self.key_to_heap: Dict[KeyType, HeapEntry[KeyType]] = dict()

    def _remove_outdated(self):
        while not self.frozen and self.expiration_heap and (self.expiration_heap[ROOT].expiration_time < get_dht_time()
                                                            or len(self.expiration_heap) > self.maxsize):
            heap_entry = heapq.heappop(self.expiration_heap)
            if self.key_to_heap.get(heap_entry.key) == heap_entry:
                del self.data[heap_entry.key], self.key_to_heap[heap_entry.key]

    def store(self, key: KeyType, value: ValueType, expiration_time: DHTExpiration) -> bool:
        """
        Store a (key, value) pair locally at least until expiration_time. See class docstring for details.
        :returns: True if new value was stored, False it was rejected (current value is newer)
        """
        if expiration_time < get_dht_time() and not self.frozen:
            return False
        self.key_to_heap[key] = HeapEntry(expiration_time, key)
        heapq.heappush(self.expiration_heap, self.key_to_heap[key])
        if key in self.data:
            if self.data[key].expiration_time < expiration_time:
                self.data[key] = ValueWithExpiration(value, expiration_time)
                return True
            return False
        self.data[key] = ValueWithExpiration(value, expiration_time)
        self._remove_outdated()
        return True

    def get(self, key: KeyType) -> Optional[ValueWithExpiration[ValueType]]:
        """ Get a value corresponding to a key if that (key, value) pair was previously stored under this key. """
        self._remove_outdated()
        if key in self.data:
            return self.data[key]
        return None

    def items(self) -> Iterator[Tuple[KeyType, ValueWithExpiration[ValueType]]]:
        """ Iterate over (key, value, expiration_time) tuples stored in this storage """
        self._remove_outdated()
        return ((key, value_and_expiration) for key, value_and_expiration in self.data.items())

    def top(self) -> Tuple[Optional[KeyType], Optional[ValueWithExpiration[ValueType]]]:
        """ Return the entry with earliest expiration or None if there isn't any """
        self._remove_outdated()
        if self.data:
            # skip leftover "ghost" entries until first real entry
            while self.key_to_heap.get(self.expiration_heap[ROOT].key) != self.expiration_heap[ROOT]:
                heapq.heappop(self.expiration_heap)
            top_key = self.expiration_heap[ROOT].key
            return top_key, self.data[top_key]
        return None, None

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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"

    @contextmanager
    def freeze(self):
        """ Temporarily cease to ._remove_outdated() elements inside this context to ensure consistency """
        prev_frozen, self.frozen = self.frozen, True
        try:
            yield self
        finally:
            self.frozen = prev_frozen


@MSGPackSerializer.ext_serializable(0x50)
class DictionaryDHTValue(TimedStorage[Subkey, BinaryDHTValue]):
    """ a dictionary-like DHT value type that maps sub-keys to values with individual expirations """
    latest_expiration_time = float('-inf')

    def store(self, key: KeyType, value: ValueType, expiration_time: DHTExpiration) -> bool:
        self.latest_expiration_time = max(self.latest_expiration_time, expiration_time)
        return super().store(key, value, expiration_time)

    def packb(self) -> bytes:
        """ custom behavior for MSGPackSerializer.dumps """
        packed_items = [[key, value, expiration_time] for key, (value, expiration_time) in self.items()]
        return MSGPackSerializer.dumps([self.maxsize, self.latest_expiration_time, packed_items])

    @classmethod
    def unpackb(cls, raw: bytes) -> DictionaryDHTValue:
        maxsize, latest_expiration_time, items = MSGPackSerializer.loads(raw)
        with DictionaryDHTValue(maxsize).freeze() as new_dict:
            for key, value, expiration_time in items:
                new_dict.store(key, value, expiration_time)
            new_dict.latest_expiration_time = latest_expiration_time
            return new_dict


class DHTLocalStorage(TimedStorage[DHTID, Union[BinaryDHTValue, DictionaryDHTValue]]):
    """ A dictionary-like storage that can store binary values and/or nested dictionaries until expiration """
    def store(self, key: DHTID, value: BinaryDHTValue, expiration_time: DHTExpiration,
              subkey: Optional[Subkey] = None) -> bool:
        """
        Store a (key, value) pair locally at least until expiration_time. See class docstring for details.
        If subkey is not None, adds a subkey-value pair to a dictionary associated with :key: (see store_subkey below)
        :returns: True if new value was stored, False it was rejected (current value is newer)
        """
        if subkey is not None:  # add one sub-key
            return self.store_subkey(key, subkey, value, expiration_time)
        else:  # store regular key
            return super().store(key, value, expiration_time)

    def store_subkey(self, key: DHTID, subkey: Subkey, value: BinaryDHTValue, expiration_time: DHTExpiration) -> bool:
        """
        Save a (sub-key, value) into a dictionary associated with a given key.
         1) if self[key] is empty, create a new dictionary and add sub-key there
         2) if self[key] is a dictionary (DictionaryDHTValue), store {sub-key: value, expiration} to that storage
         3) if self[key] is a normal value with smaller expiration time, overwrite it with a dictionary and add sub-key
        :returns: True if new entry was stored, False it was rejected (current value is newer)
        """
        previous_value, previous_expiration_time = self.get(key) or (b'', -float('inf'))
        if isinstance(previous_value, BinaryDHTValue) and expiration_time > previous_expiration_time:
            new_storage = DictionaryDHTValue()
            new_storage.store(subkey, value, expiration_time)
            return super().store(key, new_storage, new_storage.latest_expiration_time)
        elif isinstance(previous_value, DictionaryDHTValue):
            if expiration_time > previous_value.latest_expiration_time:
                super().store(key, previous_value, expiration_time)  # refresh expiration time
            return previous_value.store(subkey, value, expiration_time)
        else:
            return False


class CacheRefreshQueue(TimedStorage[DHTID, DHTExpiration]):
    """ a queue of keys scheduled for refresh in future, used in DHTNode """
    frozen = True


class NegativeCache(TimedStorage[DHTID, None]):
    """ a timed storage that stores keys banned from *something* """
