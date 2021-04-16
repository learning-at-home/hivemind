import pytest

from hivemind.dht import get_dht_time
from hivemind.dht.node import DHTNode, LOCALHOST
from hivemind.dht.schema import SchemaValidator
from hivemind.dht.validation import DHTRecord


@pytest.mark.forked
@pytest.mark.asyncio
async def test_schema_validator():
    # See https://docs.python-cerberus.org/en/stable/validation-rules.html#type
    validator = SchemaValidator({
        b'experiment_name': {'type': 'binary'},
        b'logs': {'type': 'dict',
                  'keysrules': {'type': 'binary'},
                  'valuesrules': {'type': 'binary'}},
        b'n_batches': {'type': 'dict',
                       'keysrules': {'type': 'string', 'regex': r'\[owner:.+\]'},
                       'valuesrules': {'type': 'integer', 'min': 0}},
    })

    alice = await DHTNode.create(record_validator=validator)
    bob = await DHTNode.create(
        record_validator=validator, initial_peers=[f"{LOCALHOST}:{alice.port}"])

    assert await bob.store(b'experiment_name', b'foo_bar', get_dht_time() + 10)
    assert not await bob.store(b'experiment_name', 666, get_dht_time() + 10)
    assert not await bob.store(b'experiment_name', b'foo_bar', get_dht_time() + 10,
                               subkey=b'subkey')
