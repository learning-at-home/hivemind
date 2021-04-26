import dataclasses
from functools import partial
from typing import List, Tuple

import pytest
from pydantic import BaseModel

import hivemind
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.protocol import DHTProtocol
from hivemind.dht.routing import DHTID
from hivemind.dht.schema import SchemaValidator
from hivemind.dht.validation import DHTRecord, CompositeValidator, RecordValidatorBase


class LoggingValidatorWrapper(RecordValidatorBase):
    def __init__(self, wrapped: RecordValidatorBase, name: str, log: List[Tuple[str, str]]):
        self._wrapped = wrapped
        self._log = log
        self._name = name

    def validate(self, record: DHTRecord) -> bool:
        self._log.append((self._name, 'validate'))
        print(self._name, 'validate', record.value)
        return self._wrapped.validate(record)

    def sign_value(self, record: DHTRecord) -> bytes:
        self._log.append((self._name, 'sign_value'))
        print(self._name, 'sign_value')
        return self._wrapped.sign_value(record)

    def strip_value(self, record: DHTRecord) -> bytes:
        self._log.append((self._name, 'strip_value'))
        print(self._name, 'strip_value')
        return self._wrapped.strip_value(record)


class SchemaA(BaseModel):
    field_a: bytes


class SchemaB(BaseModel):
    field_b: bytes


@pytest.fixture
def logging_validators():
    log = []
    log_as = partial(LoggingValidatorWrapper, log=log)

    # Each application may add its own validator set
    validator_sets = [
        {
            '10-schema-app-A': log_as(SchemaValidator(SchemaA), 'schema-val-A'),
            '20-signature': log_as(RSASignatureValidator(), 'signature-val-A'),
        },
        {
            '10-schema-app-B': log_as(SchemaValidator(SchemaB), 'schema-val-B'),
            '20-signature': log_as(RSASignatureValidator(), 'signature-val-B'),
        },
    ]

    return log, validator_sets


def test_composite_validator(logging_validators):
    log, validator_sets = logging_validators
    validator = CompositeValidator(validator_sets[0])
    validator.set_if_not_present(validator_sets[1])

    record = DHTRecord(key=DHTID.generate(source=b'field_a').to_bytes(),
                       subkey=DHTProtocol.IS_REGULAR_VALUE,
                       value=DHTProtocol.serializer.dumps(b'value'),
                       expiration_time=hivemind.get_dht_time() + 10)

    log.clear()
    signed_record = dataclasses.replace(record, value=validator.sign_value(record))
    assert log == [
        ('schema-val-A', 'sign_value'),
        ('schema-val-B', 'sign_value'),
        ('signature-val-A', 'sign_value'),
    ]

    log.clear()
    assert validator.validate(signed_record)
    assert log == [
        ('signature-val-A', 'validate'),
        ('signature-val-A', 'strip_value'),
        ('schema-val-B', 'validate'),
        ('schema-val-B', 'strip_value'),
        ('schema-val-A', 'validate'),
    ]

    log.clear()
    assert validator.strip_value(signed_record) == record.value
    assert log == [
        ('signature-val-A', 'strip_value'),
        ('schema-val-B', 'strip_value'),
        ('schema-val-A', 'strip_value'),
    ]


async def dummy_dht_coro(self, node, validators):
    node.protocol.record_validator.set_if_not_present(validators)


@pytest.mark.forked
def test_dht_set_validators_if_not_present(logging_validators):
    log, validator_sets = logging_validators
    # One app may create a DHT with its validators
    dht = hivemind.DHT(start=True, record_validator=CompositeValidator(validator_sets[0]))
    # Other apps may use the existing DHT and add their own validators
    dht.set_validators_if_not_present(validator_sets[1])

    assert dht.store(b'field_a', b'bytes_value', hivemind.get_dht_time() + 10)
    assert dht.get(b'field_a', latest=True).value == b'bytes_value'

    # This record does not pass the SchemaValidator
    assert not dht.store(b'field_a', 666, hivemind.get_dht_time() + 10)
    assert dht.get(b'field_a', latest=True).value == b'bytes_value'
