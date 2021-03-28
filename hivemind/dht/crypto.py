import base64
import dataclasses
import re
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from hivemind.utils.serializer import MSGPackSerializer


@dataclasses.dataclass(init=True, repr=True, frozen=True)
class DHTRecord:
    key: bytes
    subkey: bytes
    value: bytes
    expiration_time: float


class RecordValidatorBase:
    """
    Record validators are a generic mechanism for checking the DHT records including:
      - Enforcing a data schema (e.g. checking content types)
      - Enforcing security requirements (e.g. allowing only the owner to update the record)
    """

    def validate(self, record: DHTRecord) -> None:
        """
        This method should:
          - Return None if validation is successful
          - Raise ValueError with the reason if it is not

        validate() is called when another DHT peer:
          - Asks us to store the record
          - Returns the record by our request
        """

        raise NotImplementedError()

    def sign_value(self, record: DHTRecord) -> bytes:
        """
        Should return `record.value` extended with the record's signature.

        Note: there's no need to overwrite this method if a validator doesn't use a signature.

        sign_value() is called after the application asks the DHT to store the record.
        """

        return record.value

    def strip_value(self, record: DHTRecord) -> bytes:
        """
        Should return `record.value` stripped of the record's signature.
        strip_value() is only called if validate() was successful.

        Note: there's no need to overwrite this method if a validator doesn't use a signature.

        strip_value() is called before the DHT returns the record by the application's request.
        """

        return record.value


class RSASignatureValidator(RecordValidatorBase):
    """
    Introduces a notion of *protected records* whose key/subkey contains substring
    "[owner:ssh-rsa ...]" (the format can be changed) with an RSA public key of the owner.

    If this validator is used, changes to such records always must be signed with
    the corresponding private key (so only the owner can change them).
    """

    def __init__(self,
                 marker_format: bytes=b'[owner:_key_]',
                 signature_format: bytes=b'[signature:_value_]'):
        self._marker_re = re.compile(re.escape(marker_format).replace(b'_key_', rb'(.+?)'))

        self._signature_format = signature_format
        self._signature_re = re.compile(re.escape(signature_format).replace(b'_value_', rb'(.+?)'))

        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        serialized_public_key = self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.OpenSSH, format=serialization.PublicFormat.OpenSSH)
        self._our_marker = marker_format.replace(b'_key_', serialized_public_key)

        self._padding = padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                    salt_length=padding.PSS.MAX_LENGTH)
        self._hash_algorithm = hashes.SHA256()

    @property
    def ownership_marker(self) -> bytes:
        return self._our_marker

    def validate(self, record: DHTRecord) -> None:
        public_keys = self._marker_re.findall(record.key)
        if record.subkey is not None:
            public_keys += self._marker_re.findall(record.subkey)
        if not public_keys:
            return None  # Success (the record is not protected by a public key)

        if len(set(public_keys)) > 1:
            raise ValueError("Key and subkey can't contain different public keys")
        public_key = serialization.load_ssh_public_key(public_keys[0])

        signatures = self._signature_re.findall(record.value)
        if len(signatures) != 1:
            raise ValueError('Protected record should have exactly one signature')
        signature = base64.b64decode(signatures[0])

        stripped_record = dataclasses.replace(record, value=self.strip_value(record))
        try:
            # verify() returns None iff the signature is correct
            public_key.verify(signature, self._serialize_record(stripped_record),
                              self._padding, self._hash_algorithm)
            return None  # Success
        except InvalidSignature:
            raise ValueError('Invalid signature')

    def sign_value(self, record: DHTRecord) -> bytes:
        if self._our_marker not in record.key and self._our_marker not in record.subkey:
            return record.value

        signature = self._private_key.sign(self._serialize_record(record),
                                           self._padding, self._hash_algorithm)
        signature = base64.b64encode(signature)
        return record.value + self._signature_format.replace(b'_value_', signature)

    def strip_value(self, record: DHTRecord) -> bytes:
        return self._signature_re.sub(b'', record.value)

    def _serialize_record(self, record: DHTRecord) -> bytes:
        return MSGPackSerializer.dumps(dataclasses.astuple(record))
