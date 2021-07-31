import dataclasses
from typing import Dict

import pytest
from pydantic import BaseModel, StrictInt

import hivemind
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.protocol import DHTProtocol
from hivemind.dht.routing import DHTID
from hivemind.dht.schema import BytesWithPublicKey, SchemaValidator
from hivemind.dht.validation import CompositeValidator, DHTRecord


class SchemaA(BaseModel):
    field_a: bytes


class SchemaB(BaseModel):
    field_b: Dict[BytesWithPublicKey, StrictInt]


@pytest.fixture
def validators_for_app():
    # Each application may add its own validator set
    return {
        "A": [RSASignatureValidator(), SchemaValidator(SchemaA, allow_extra_keys=False)],
        "B": [SchemaValidator(SchemaB, allow_extra_keys=False), RSASignatureValidator()],
    }


def test_composite_validator(validators_for_app):
    validator = CompositeValidator(validators_for_app["A"])
    assert [type(item) for item in validator._validators] == [SchemaValidator, RSASignatureValidator]

    validator.extend(validators_for_app["B"])
    assert [type(item) for item in validator._validators] == [SchemaValidator, RSASignatureValidator]
    assert len(validator._validators[0]._schemas) == 2

    local_public_key = validators_for_app["A"][0].local_public_key
    record = DHTRecord(
        key=DHTID.generate(source="field_b").to_bytes(),
        subkey=DHTProtocol.serializer.dumps(local_public_key),
        value=DHTProtocol.serializer.dumps(777),
        expiration_time=hivemind.get_dht_time() + 10,
    )

    signed_record = dataclasses.replace(record, value=validator.sign_value(record))
    # Expect only one signature since two RSASignatureValidatos have been merged
    assert signed_record.value.count(b"[signature:") == 1
    # Expect successful validation since the second SchemaValidator has been merged to the first
    assert validator.validate(signed_record)
    assert validator.strip_value(signed_record) == record.value

    record = DHTRecord(
        key=DHTID.generate(source="unknown_key").to_bytes(),
        subkey=DHTProtocol.IS_REGULAR_VALUE,
        value=DHTProtocol.serializer.dumps(777),
        expiration_time=hivemind.get_dht_time() + 10,
    )

    signed_record = dataclasses.replace(record, value=validator.sign_value(record))
    assert signed_record.value.count(b"[signature:") == 0
    # Expect failed validation since `unknown_key` is not a part of any schema
    assert not validator.validate(signed_record)


@pytest.mark.forked
def test_dht_add_validators(validators_for_app):
    # One app may create a DHT with its validators
    dht = hivemind.DHT(start=False, record_validators=validators_for_app["A"])

    # While the DHT process is not started, you can't send a command to append new validators
    with pytest.raises(RuntimeError):
        dht.add_validators(validators_for_app["B"])
    dht.run_in_background(await_ready=True)

    # After starting the process, other apps may add new validators to the existing DHT
    dht.add_validators(validators_for_app["B"])

    assert dht.store("field_a", b"bytes_value", hivemind.get_dht_time() + 10)
    assert dht.get("field_a", latest=True).value == b"bytes_value"

    assert not dht.store("field_a", 666, hivemind.get_dht_time() + 10)
    assert dht.get("field_a", latest=True).value == b"bytes_value"

    local_public_key = validators_for_app["A"][0].local_public_key
    assert dht.store("field_b", 777, hivemind.get_dht_time() + 10, subkey=local_public_key)
    dictionary = dht.get("field_b", latest=True).value
    assert len(dictionary) == 1 and dictionary[local_public_key].value == 777

    assert not dht.store("unknown_key", 666, hivemind.get_dht_time() + 10)
    assert dht.get("unknown_key", latest=True) is None
