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

    PUBLIC_KEY_FORMAT = b'[owner:_key_]'
    SIGNATURE_FORMAT = b'[signature:_value_]'

    PUBLIC_KEY_REGEX = re.escape(PUBLIC_KEY_FORMAT).replace(b'_key_', rb'(.+?)')
    _public_key_re = re.compile(PUBLIC_KEY_REGEX)
    _signature_re = re.compile(re.escape(SIGNATURE_FORMAT).replace(b'_value_', rb'(.+?)'))

    _cached_private_key = None

    def __init__(self, *, ignore_cached_key=False):
        if self._cached_private_key is None or ignore_cached_key:
            # Since generating a private key takes ~100 ms, we cache it for future validator
            # instances in the same process (unless ignore_cached_key=True)
            self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            if not ignore_cached_key:
                RSASignatureValidator._cached_private_key = self._private_key
        else:
            self._private_key = RSASignatureValidator._cached_private_key

        serialized_public_key = self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.OpenSSH, format=serialization.PublicFormat.OpenSSH)
        self._local_public_key = self.PUBLIC_KEY_FORMAT.replace(b'_key_', serialized_public_key)

        self._init_signature_params()

    def _init_signature_params(self) -> None:
        self._padding = padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                    salt_length=padding.PSS.MAX_LENGTH)
        self._hash_algorithm = hashes.SHA256()

    @property
    def local_public_key(self) -> bytes:
        return self._local_public_key

    def validate(self, record: DHTRecord) -> bool:
        public_keys = self._public_key_re.findall(record.key)
        if record.subkey is not None:
            public_keys += self._public_key_re.findall(record.subkey)
        if not public_keys:
            return True  # The record is not protected with a public key

        if len(set(public_keys)) > 1:
            logger.warning(f"Key and subkey can't contain different public keys in {record}")
            return False
        public_key = serialization.load_ssh_public_key(public_keys[0])

        signatures = self._signature_re.findall(record.value)
        if len(signatures) != 1:
            logger.warning(f"Record should have exactly one signature in {record}")
            return False
        signature = base64.b64decode(signatures[0])

        stripped_record = dataclasses.replace(record, value=self.strip_value(record))
        try:
            # verify() returns None iff the signature is correct
            public_key.verify(signature, self._serialize_record(stripped_record),
                              self._padding, self._hash_algorithm)
            return True
        except InvalidSignature:
            logger.warning(f'Signature is invalid in {record}')
            return False

    def sign_value(self, record: DHTRecord) -> bytes:
        if self._local_public_key not in record.key and self._local_public_key not in record.subkey:
            return record.value

        signature = self._private_key.sign(self._serialize_record(record),
                                           self._padding, self._hash_algorithm)
        signature = base64.b64encode(signature)
        return record.value + self.SIGNATURE_FORMAT.replace(b'_value_', signature)

    def strip_value(self, record: DHTRecord) -> bytes:
        return self._signature_re.sub(b'', record.value)

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
        self._init_signature_params()
