import dataclasses
import re
from typing import Optional

from hivemind.dht.validation import DHTRecord, RecordValidatorBase
from hivemind.utils import MSGPackSerializer, get_logger
from hivemind.utils.crypto import RSAPrivateKey, RSAPublicKey

logger = get_logger(__name__)


class RSASignatureValidator(RecordValidatorBase):
    """
    Introduces a notion of *protected records* whose key/subkey contains substring
    "[owner:ssh-rsa ...]" with an RSA public key of the owner.

    If this validator is used, changes to such records always must be signed with
    the corresponding private key (so only the owner can change them).
    """

    PUBLIC_KEY_FORMAT = b"[owner:_key_]"
    SIGNATURE_FORMAT = b"[signature:_value_]"

    PUBLIC_KEY_REGEX = re.escape(PUBLIC_KEY_FORMAT).replace(b"_key_", rb"(.+?)")
    _PUBLIC_KEY_RE = re.compile(PUBLIC_KEY_REGEX)
    _SIGNATURE_RE = re.compile(re.escape(SIGNATURE_FORMAT).replace(b"_value_", rb"(.+?)"))

    _cached_private_key = None

    def __init__(self, private_key: Optional[RSAPrivateKey] = None):
        if private_key is None:
            private_key = RSAPrivateKey.process_wide()
        self._private_key = private_key

        serialized_public_key = private_key.get_public_key().to_bytes()
        self._local_public_key = self.PUBLIC_KEY_FORMAT.replace(b"_key_", serialized_public_key)

    @property
    def local_public_key(self) -> bytes:
        return self._local_public_key

    def validate(self, record: DHTRecord) -> bool:
        public_keys = self._PUBLIC_KEY_RE.findall(record.key)
        if record.subkey is not None:
            public_keys += self._PUBLIC_KEY_RE.findall(record.subkey)
        if not public_keys:
            return True  # The record is not protected with a public key

        if len(set(public_keys)) > 1:
            logger.debug(f"Key and subkey can't contain different public keys in {record}")
            return False
        public_key = RSAPublicKey.from_bytes(public_keys[0])

        signatures = self._SIGNATURE_RE.findall(record.value)
        if len(signatures) != 1:
            logger.debug(f"Record should have exactly one signature in {record}")
            return False
        signature = signatures[0]

        stripped_record = dataclasses.replace(record, value=self.strip_value(record))
        if not public_key.verify(self._serialize_record(stripped_record), signature):
            logger.debug(f"Signature is invalid in {record}")
            return False
        return True

    def sign_value(self, record: DHTRecord) -> bytes:
        if self._local_public_key not in record.key and self._local_public_key not in record.subkey:
            return record.value

        signature = self._private_key.sign(self._serialize_record(record))
        return record.value + self.SIGNATURE_FORMAT.replace(b"_value_", signature)

    def strip_value(self, record: DHTRecord) -> bytes:
        return self._SIGNATURE_RE.sub(b"", record.value)

    def _serialize_record(self, record: DHTRecord) -> bytes:
        return MSGPackSerializer.dumps(dataclasses.astuple(record))

    @property
    def priority(self) -> int:
        # On validation, this validator must be executed before validators
        # that deserialize the record
        return 10

    def merge_with(self, other: RecordValidatorBase) -> bool:
        if not isinstance(other, RSASignatureValidator):
            return False

        # Ignore another RSASignatureValidator instance (it doesn't make sense to have several
        # instances of this class) and report successful merge
        return True
