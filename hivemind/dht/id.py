from __future__ import annotations
import hashlib
import os
import random
from typing import Optional


class DHTNodeID(int):
    HASH_FUNC = hashlib.sha1
    HASH_NBYTES = 20  # SHA1 produces a 20-byte (aka 160bit) number
    RANGE = MIN, MAX = 0, 2 ** (HASH_NBYTES * 8)  # inclusive min, exclusive max

    def __new__(cls, value: int):
        assert cls.MIN <= value < cls.MAX, f"DHTNodeID must be in [{cls.MIN}, {cls.MAX}) but got {value}"
        return super().__new__(cls, value)

    def to_bytes(self, length=HASH_NBYTES, byteorder='big', *, signed=False) -> bytes:
        return super().to_bytes(length, byteorder, signed=signed)

    @classmethod
    def generate(cls, seed: Optional[int] = None, nbits: int = 255):
        """
        Generates random uid based on SHA1
        """
        randbytes = (seed or random.getrandbits(nbits)).to_bytes(nbits, byteorder='big')
        raw_uid = hashlib.sha1(randbytes).digest()
        return cls(int(raw_uid.hex(), 16))

    def xor_distance(self, other: DHTNodeID) -> int:
        """ Return a number which binary representation equals bitwise xor between the two DHTNodeIDs """
        return int(self) ^ int(other)

    @classmethod
    def longest_common_prefix_length(cls, *ids: DHTNodeID) -> int:
        ids_bits = [bin(uid)[2:].rjust(8 * cls.HASH_NBYTES, '0') for uid in ids]
        return len(os.path.commonprefix(ids_bits))

    def __repr__(self):
        return f"DHTNodeID({int(self)})"
