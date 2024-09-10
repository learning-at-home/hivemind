from __future__ import annotations
import base64
import threading
from abc import ABC, abstractmethod
from cryptography import exceptions
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


class PrivateKey(ABC):
    @abstractmethod
    def sign(self, data: bytes) -> bytes:
        ...

    @abstractmethod
    def get_public_key(self) -> PublicKey:
        ...


class PublicKey(ABC):
    @abstractmethod
    def verify(self, data: bytes, signature: bytes) -> bool:
        ...

    @abstractmethod
    def to_bytes(self) -> bytes:
        ...

    @classmethod
    @abstractmethod
    def from_bytes(cls, key: bytes) -> PublicKey:
        ...

class RSAPrivateKey(PrivateKey):
    def __init__(self):
        self._private_key = ed25519.Ed25519PrivateKey.generate()

    _process_wide_key = None
    _process_wide_key_lock = threading.RLock()

    @classmethod
    def process_wide(cls) -> RSAPrivateKey:
        if cls._process_wide_key is None:
            with cls._process_wide_key_lock:
                if cls._process_wide_key is None:
                    cls._process_wide_key = cls()
        return cls._process_wide_key

    def sign(self, data: bytes) -> bytes:
        signature = self._private_key.sign(data)
        return base64.b64encode(signature)

    def get_public_key(self) -> RSAPublicKey:  # Fix: return RSAPublicKey, not RSAPrivateKey
        return RSAPublicKey(self._private_key.public_key())  # Return RSAPublicKey

    def to_bytes(self) -> bytes:
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_private_key"] = self.to_bytes()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(self._private_key)



class RSAPublicKey(PublicKey):
    def __init__(self, public_key: ed25519.Ed25519PublicKey):
        self._public_key = public_key

    def verify(self, data: bytes, signature: bytes) -> bool:
        try:
            signature = base64.b64decode(signature)
            self._public_key.verify(signature, data)
            return True
        except (ValueError, exceptions.InvalidSignature):
            return False

    def to_bytes(self) -> bytes:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

    @classmethod
    def from_bytes(cls, key: bytes) -> RSAPrivateKey:
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(key)
        return cls(public_key)
