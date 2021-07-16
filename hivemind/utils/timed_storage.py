""" A dictionary-like storage that stores items until a specified expiration time or up to a limited size """
from __future__ import annotations

import heapq
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generic, Iterator, List, Optional, Tuple, TypeVar

KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")
get_dht_time = time.time  # a global (weakly synchronized) time
MAX_DHT_TIME_DISCREPANCY_SECONDS = 3  # max allowed difference between get_dht_time for two DHT nodes
DHTExpiration = float
ROOT = 0


@dataclass(init=True, repr=True, frozen=True)
class ValueWithExpiration(Generic[ValueType]):
    value: ValueType
    expiration_time: DHTExpiration

    def __iter__(self):
        return iter((self.value, self.expiration_time))

    def __getitem__(self, item):
        if item == 0:
            return self.value
        elif item == 1:
            return self.expiration_time
        else:
            return getattr(self, item)

    def __eq__(self, item):
        if isinstance(item, ValueWithExpiration):
            return self.value == item.value and self.expiration_time == item.expiration_time
        elif isinstance(item, tuple):
            return tuple.__eq__((self.value, self.expiration_time), item)
        else:
            return False


@dataclass(init=True, repr=True, order=True, frozen=True)
class HeapEntry(Generic[KeyType]):
    expiration_time: DHTExpiration
    key: KeyType


class TimedStorage(Generic[KeyType, ValueType]):
    """A dictionary that maintains up to :maxsize: key-value-expiration tuples until their expiration_time"""

    frozen = False  # can be set to True. If true, do not remove outdated elements

    def __init__(self, maxsize: Optional[int] = None):
        self.maxsize = maxsize or float("inf")
        self.data: Dict[KeyType, ValueWithExpiration[ValueType]] = dict()
        self.expiration_heap: List[HeapEntry[KeyType]] = []
        self.key_to_heap: Dict[KeyType, HeapEntry[KeyType]] = dict()

    def _remove_outdated(self):
        while (
            not self.frozen
            and self.expiration_heap
            and (self.expiration_heap[ROOT].expiration_time < get_dht_time() or len(self.data) > self.maxsize)
        ):
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
        """Get a value corresponding to a key if that (key, value) pair was previously stored under this key."""
        self._remove_outdated()
        if key in self.data:
            return self.data[key]
        return None

    def items(self) -> Iterator[Tuple[KeyType, ValueWithExpiration[ValueType]]]:
        """Iterate over (key, value, expiration_time) tuples stored in this storage"""
        self._remove_outdated()
        return ((key, value_and_expiration) for key, value_and_expiration in self.data.items())

    def top(self) -> Tuple[Optional[KeyType], Optional[ValueWithExpiration[ValueType]]]:
        """Return the entry with earliest expiration or None if there isn't any"""
        self._remove_outdated()
        if self.data:
            # skip leftover "ghost" entries until first real entry
            while self.key_to_heap.get(self.expiration_heap[ROOT].key) != self.expiration_heap[ROOT]:
                heapq.heappop(self.expiration_heap)
            top_key = self.expiration_heap[ROOT].key
            return top_key, self.data[top_key]
        return None, None

    def clear(self):
        self.data.clear()
        self.key_to_heap.clear()
        self.expiration_heap.clear()

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
        """Temporarily cease to ._remove_outdated() elements inside this context to ensure consistency"""
        prev_frozen, self.frozen = self.frozen, True
        try:
            yield self
        finally:
            self.frozen = prev_frozen
