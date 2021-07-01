import dataclasses
import pickle
import multiprocessing as mp

import pytest

import hivemind
from hivemind.dht import get_dht_time
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.validation import DHTRecord
from hivemind.utils.crypto import RSAPrivateKey


def test_rsa_signature_validator():
    receiver_validator = RSASignatureValidator()
    sender_validator = RSASignatureValidator(RSAPrivateKey())
    mallory_validator = RSASignatureValidator(RSAPrivateKey())

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

    third_validator = RSASignatureValidator(RSAPrivateKey())
    assert first_validator.local_public_key != third_validator.local_public_key


def test_validator_instance_is_picklable():
    # Needs to be picklable because the validator instance may be sent between processes

    original_validator = RSASignatureValidator()
    unpickled_validator = pickle.loads(pickle.dumps(original_validator))

    # To check that the private key was pickled and unpickled correctly, we sign a record
    # with the original public key using the unpickled validator and then validate the signature

    record = DHTRecord(key=b'key', subkey=b'subkey' + original_validator.local_public_key,
                       value=b'value', expiration_time=get_dht_time() + 10)
    signed_record = dataclasses.replace(record, value=unpickled_validator.sign_value(record))

    assert b'[signature:' in signed_record.value
    assert original_validator.validate(signed_record)
    assert unpickled_validator.validate(signed_record)


def get_signed_record(conn: mp.connection.Connection) -> DHTRecord:
    validator = conn.recv()
    record = conn.recv()

    record = dataclasses.replace(record, value=validator.sign_value(record))

    conn.send(record)


def test_signing_in_different_process():
    parent_conn, child_conn = mp.Pipe()
    process = mp.Process(target=get_signed_record, args=[child_conn])
    process.start()

    validator = RSASignatureValidator()
    parent_conn.send(validator)

    record = DHTRecord(key=b'key', subkey=b'subkey' + validator.local_public_key,
                       value=b'value', expiration_time=get_dht_time() + 10)
    parent_conn.send(record)

    signed_record = parent_conn.recv()
    assert b'[signature:' in signed_record.value
    assert validator.validate(signed_record)


@pytest.mark.forked
@pytest.mark.asyncio
async def test_dhtnode_signatures():
    alice = await hivemind.DHTNode.create(record_validator=RSASignatureValidator())
    initial_peers = await alice.identify_maddrs()
    bob = await hivemind.DHTNode.create(
        record_validator=RSASignatureValidator(RSAPrivateKey()), initial_peers=initial_peers)
    mallory = await hivemind.DHTNode.create(
        record_validator=RSASignatureValidator(RSAPrivateKey()), initial_peers=initial_peers)

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
