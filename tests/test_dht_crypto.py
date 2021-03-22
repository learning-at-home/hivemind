import dataclasses
import time

import pytest

from hivemind.dht.crypto import ProtectedRecord, RSASignatureValidator


def test_rsa_signature_validator():
    sender_validator = RSASignatureValidator()
    plain_record = ProtectedRecord(key=b'key', subkey=b'subkey', value=b'value',
                                   expiration_time=time.time())
    protected_records = [
        dataclasses.replace(plain_record,
                            key=plain_record.key + sender_validator.ownership_marker),
        dataclasses.replace(plain_record,
                            subkey=plain_record.subkey + sender_validator.ownership_marker),
    ]
    signatures = [sender_validator.sign(record) for record in protected_records]

    receiver_validator = RSASignatureValidator()

    # test 1: Non-protected record
    receiver_validator.validate(plain_record, b'')

    # test 2: Correct signatures
    for record, signature in zip(protected_records, signatures):
        receiver_validator.validate(record, signature)

    # test 3: Invalid signatures
    for record in protected_records:
        record = dataclasses.replace(record, value=b'Mallory_changed_this_value')

        for signature in signatures + [b'', b'arbitrary_bytes']:
            with pytest.raises(ValueError):
                receiver_validator.validate(record, signature)
