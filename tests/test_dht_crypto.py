import dataclasses

import pytest

from hivemind.dht import get_dht_time
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.validation import DHTRecord


def test_rsa_signature_validator():
    receiver_validator = RSASignatureValidator()
    sender_validator = RSASignatureValidator()
    mallory_validator = RSASignatureValidator()

    plain_record = DHTRecord(key=b'key', subkey=b'subkey', value=b'value',
                             expiration_time=get_dht_time() + 10)
    protected_records = [
        dataclasses.replace(plain_record,
                            key=plain_record.key + sender_validator.ownership_marker),
        dataclasses.replace(plain_record,
                            subkey=plain_record.subkey + sender_validator.ownership_marker),
    ]

    # test 1: Non-protected record (no signature added)
    assert sender_validator.sign_value(plain_record) == plain_record.value
    assert receiver_validator.validate(plain_record)

    # test 2: Correct signatures
    signed_records = [dataclasses.replace(record, value=sender_validator.sign_value(record))
                      for record in protected_records]
    for record in signed_records:
        assert receiver_validator.validate(record)
        assert receiver_validator.strip_value(record) == b'value'

    # test 3: Invalid signatures
    signed_records = protected_records  # Without signature
    signed_records += [dataclasses.replace(record,
                                           value=record.value + b'[signature:INVALID_BYTES]')
                       for record in protected_records]  # With invalid signature
    signed_records += [dataclasses.replace(record, value=mallory_validator.sign_value(record))
                       for record in protected_records]  # With someone else's signature
    for record in signed_records:
        assert not receiver_validator.validate(record)
