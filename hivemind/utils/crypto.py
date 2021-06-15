from __future__ import annotations

import base64
import threading
from abc import ABC, abstractmethod

from cryptography import exceptions
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


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
    def from_bytes(cls, key: bytes) -> bytes:
        ...


_RSA_PADDING = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH)
_RSA_HASH_ALGORITHM = hashes.SHA256()


class RSAPrivateKey(PrivateKey):
    def __init__(self):
        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

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
        signature = self._private_key.sign(data, _RSA_PADDING, _RSA_HASH_ALGORITHM)
        return base64.b64encode(signature)

    def get_public_key(self) -> RSAPublicKey:
        return RSAPublicKey(self._private_key.public_key())

    def __getstate__(self):
        state = self.__dict__.copy()
        # Serializes the private key to make the class instances picklable
        state["_private_key"] = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._private_key = serialization.load_ssh_private_key(self._private_key, password=None)


class RSAPublicKey(PublicKey):
    def __init__(self, public_key: rsa.RSAPublicKey):
        self._public_key = public_key

    def verify(self, data: bytes, signature: bytes) -> bool:
        try:
            signature = base64.b64decode(signature)

            # Returns None if the signature is correct, raises an exception otherwise
            self._public_key.verify(signature, data, _RSA_PADDING, _RSA_HASH_ALGORITHM)

            return True
        except (ValueError, exceptions.InvalidSignature):
            return False

    def to_bytes(self) -> bytes:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH, format=serialization.PublicFormat.OpenSSH
        )

    @classmethod
    def from_bytes(cls, key: bytes) -> RSAPublicKey:
        key = serialization.load_ssh_public_key(key)
        if not isinstance(key, rsa.RSAPublicKey):
            raise ValueError(f"Expected an RSA public key, got {key}")
        return cls(key)
