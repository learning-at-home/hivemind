import dataclasses
import time

import pytest

from hivemind.dht.crypto import ProtectedRecord, RSASignatureValidator


def test_rsa_validator_on_correct_signature():
    sender_validator = RSASignatureValidator()
    plain_record = ProtectedRecord(key=b'key', subkey=b'subkey', value=b'value',
                                   expiration_time=time.time())
    protected_record = dataclasses.replace(
        plain_record, key=plain_record.key + sender_validator.ownership_marker)
    signature = sender_validator.sign(protected_record)

    receiver_validator = RSASignatureValidator()
    receiver_validator.validate(plain_record, b'')
    receiver_validator.validate(protected_record, signature)


def test_rsa_validator_on_fake_signatures():
    sender_validator = RSASignatureValidator()
    original_record = ProtectedRecord(key=b'key' + sender_validator.ownership_marker,
                                      subkey=b'subkey', value=b'true_value',
                                      expiration_time=time.time())
    fake_record = dataclasses.replace(original_record, value=b'fake_value')
    fake_signatures = [b'', b'arbitrary_bytes', sender_validator.sign(fake_record)]

    receiver_validator = RSASignatureValidator()
    for signature in fake_signatures:
        with pytest.raises(ValueError):
            receiver_validator.validate(original_record, signature)
