import base64
import dataclasses
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from hivemind.dht.validation import DHTRecord, RecordValidatorBase
from hivemind.utils import MSGPackSerializer, get_logger


logger = get_logger(__name__)


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
        self._ownership_marker = marker_format.replace(b'_key_', serialized_public_key)

        self._padding = padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                    salt_length=padding.PSS.MAX_LENGTH)
        self._hash_algorithm = hashes.SHA256()

    @property
    def ownership_marker(self) -> bytes:
        return self._ownership_marker

    def validate(self, record: DHTRecord) -> bool:
        public_keys = self._marker_re.findall(record.key)
        if record.subkey is not None:
            public_keys += self._marker_re.findall(record.subkey)
        if not public_keys:
            return True  # The record is not protected with a public key

        if len(set(public_keys)) > 1:
            logger.debug(f"Key and subkey can't contain different public keys in {record}")
            return False
        public_key = serialization.load_ssh_public_key(public_keys[0])

        signatures = self._signature_re.findall(record.value)
        if len(signatures) != 1:
            logger.debug(f"Record should have exactly one signature in {record}")
            return False
        signature = base64.b64decode(signatures[0])

        stripped_record = dataclasses.replace(record, value=self.strip_value(record))
        try:
            # verify() returns None iff the signature is correct
            public_key.verify(signature, self._serialize_record(stripped_record),
                              self._padding, self._hash_algorithm)
            return True
        except InvalidSignature:
            logger.debug(f'Signature is invalid in {record}')
            return False

    def sign_value(self, record: DHTRecord) -> bytes:
        if self._ownership_marker not in record.key and self._ownership_marker not in record.subkey:
            return record.value

        signature = self._private_key.sign(self._serialize_record(record),
                                           self._padding, self._hash_algorithm)
        signature = base64.b64encode(signature)
        return record.value + self._signature_format.replace(b'_value_', signature)

    def strip_value(self, record: DHTRecord) -> bytes:
        return self._signature_re.sub(b'', record.value)

    def _serialize_record(self, record: DHTRecord) -> bytes:
        return MSGPackSerializer.dumps(dataclasses.astuple(record))

    def __getstate__(self):
        state = self.__dict__.copy()
        # Serializes the private key to make the class instances picklable
        state['_private_key'] = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption())
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._private_key = serialization.load_ssh_private_key(self._private_key, password=None)
