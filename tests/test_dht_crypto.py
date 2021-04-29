import dataclasses

import pytest

import hivemind
from hivemind.dht import get_dht_time
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.node import LOCALHOST
from hivemind.dht.validation import DHTRecord


def test_rsa_signature_validator():
    receiver_validator = RSASignatureValidator()
    sender_validator = RSASignatureValidator(ignore_cached_key=True)
    mallory_validator = RSASignatureValidator(ignore_cached_key=True)

    plain_record = DHTRecord(key=b'key', subkey=b'subkey', value=b'value',
                             expiration_time=get_dht_time() + 10)
    protected_records = [
        dataclasses.replace(plain_record,
                            key=plain_record.key + sender_validator.local_public_key),
        dataclasses.replace(plain_record,
                            subkey=plain_record.subkey + sender_validator.local_public_key),
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


def test_cached_key():
    first_validator = RSASignatureValidator()
    second_validator = RSASignatureValidator()
    assert first_validator.local_public_key == second_validator.local_public_key

    third_validator = RSASignatureValidator(ignore_cached_key=True)
    assert first_validator.local_public_key != third_validator.local_public_key


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dhtnode_signatures():
    alice = await hivemind.DHTNode.create(record_validator=RSASignatureValidator())
    bob = await hivemind.DHTNode.create(
        record_validator=RSASignatureValidator(ignore_cached_key=True),
        initial_peers=[f"{LOCALHOST}:{alice.port}"])
    mallory = await hivemind.DHTNode.create(
        record_validator=RSASignatureValidator(ignore_cached_key=True),
        initial_peers=[f"{LOCALHOST}:{alice.port}"])

    key = b'key'
    subkey = b'protected_subkey' + bob.protocol.record_validator.local_public_key

    assert await bob.store(key, b'true_value', hivemind.get_dht_time() + 10, subkey=subkey)
    assert (await alice.get(key, latest=True)).value[subkey].value == b'true_value'

    store_ok = await mallory.store(key, b'fake_value', hivemind.get_dht_time() + 10, subkey=subkey)
    assert not store_ok
    assert (await alice.get(key, latest=True)).value[subkey].value == b'true_value'

    assert await bob.store(key, b'updated_true_value', hivemind.get_dht_time() + 10, subkey=subkey)
    assert (await alice.get(key, latest=True)).value[subkey].value == b'updated_true_value'

    await bob.shutdown()  # Bob has shut down, now Mallory is the single peer of Alice

    store_ok = await mallory.store(key, b'updated_fake_value',
                                   hivemind.get_dht_time() + 10, subkey=subkey)
    assert not store_ok
    assert (await alice.get(key, latest=True)).value[subkey].value == b'updated_true_value'
