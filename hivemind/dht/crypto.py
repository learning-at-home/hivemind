import re
from dataclasses import astuple, dataclass
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from hivemind.utils.serializer import MSGPackSerializer


@dataclass(init=True, repr=True, frozen=True)
class ProtectedRecord:
    key: bytes
    subkey: Optional[bytes]
    value: bytes
    expiration_time: float


class RecordValidatorBase:
    def validate(self, record: ProtectedRecord) -> None:
        """
        If validation is successful, returns None.
        If it is not, raises ValueError with the reason.

        This method must not raise exceptions that are not inherited from ValueError for any input.
        """

        raise NotImplementedError()

    def sign(self, record: ProtectedRecord) -> bytes:
        raise NotImplementedError()


class RSASignatureValidator(RecordValidatorBase):
    """
    Introduces a notion of *protected records* whose key/subkey contains substring
    "[owner:ssh-rsa ...]" (the format can be changed) with an RSA public key of the owner.

    If this validator is used, changes to such records always must be signed with
    the corresponding private key (so only the owner can change them).
    """

    def __init__(self, marker_format: bytes=b'[owner:key]'):
        self._marker_format = marker_format
        self._marker_re = re.compile(re.escape(marker_format).replace(b'key', rb'(.+?)'))

        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        serialized_public_key = self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.OpenSSH, format=serialization.PublicFormat.OpenSSH)
        self._our_marker = marker_format.replace(b'key', serialized_public_key)

        self._padding = padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                    salt_length=padding.PSS.MAX_LENGTH)
        self._hash_algorithm = hashes.SHA256()

    @property
    def ownership_marker(self) -> bytes:
        return self._our_marker

    def validate(self, record: ProtectedRecord, signature: Optional[bytes]) -> None:
        self._marker_re.findall(record.key)

        public_keys = self._marker_re.findall(record.key)
        if record.subkey is not None:
            public_keys += self._marker_re.findall(record.subkey)
        if not public_keys:
            return

        if signature is None:
            raise ValueError('Signature is required but not present')
        if len(set(public_keys)) > 1:
            raise ValueError("Key and subkey can't contain different public keys")
        public_key = serialization.load_ssh_public_key(public_keys[0])

        try:
            public_key.verify(signature, self._serialize_record(record),
                              self._padding, self._hash_algorithm)
            # verify() returns None iff the signature is correct
        except InvalidSignature:
            raise ValueError('Invalid signature')

    def sign(self, record: ProtectedRecord) -> Optional[bytes]:
        if self._our_marker not in record.key and self._our_marker not in record.subkey:
            return None

        return self._private_key.sign(self._serialize_record(record),
                                      self._padding, self._hash_algorithm)

    def _serialize_record(self, record: ProtectedRecord) -> bytes:
        return MSGPackSerializer.dumps(astuple(record))
