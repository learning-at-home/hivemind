import re

import pydantic
import pytest
from pydantic import conint
from typing import Dict

from hivemind.dht import get_dht_time
from hivemind.dht.node import DHTNode, LOCALHOST
from hivemind.dht.schema import SchemaValidator, conbytes


@pytest.fixture
async def dht_nodes_with_schema():
    class Schema(pydantic.BaseModel):
        experiment_name: bytes
        n_batches: Dict[bytes, conint(ge=0, strict=True)]
        signed_data: Dict[conbytes(regex=rb'.*\[owner:.+\]'), bytes]

    validator = SchemaValidator(Schema)

    alice = await DHTNode.create(record_validator=validator)
    bob = await DHTNode.create(
        record_validator=validator, initial_peers=[f"{LOCALHOST}:{alice.port}"])
    return alice, bob


@pytest.mark.forked
@pytest.mark.asyncio
async def test_keys_outside_schema(dht_nodes_with_schema):
    alice, bob = dht_nodes_with_schema

    assert await bob.store(b'unknown_key', b'foo_bar', get_dht_time() + 10)

    for peer in [alice, bob]:
        assert (await peer.get(b'unknown_key', latest=True)).value == b'foo_bar'


@pytest.mark.forked
@pytest.mark.asyncio
async def test_expecting_regular_value(dht_nodes_with_schema):
    alice, bob = dht_nodes_with_schema

    # Regular value (bytes) expected
    assert await bob.store(b'experiment_name', b'foo_bar', get_dht_time() + 10)
    assert not await bob.store(b'experiment_name', 666, get_dht_time() + 10)
    assert not await bob.store(b'experiment_name', b'foo_bar', get_dht_time() + 10,
                               subkey=b'subkey')

    # Validate despite https://pydantic-docs.helpmanual.io/usage/models/#data-conversion
    assert not await bob.store(b'experiment_name', [], get_dht_time() + 10)
    assert not await bob.store(b'experiment_name', [1, 2, 3], get_dht_time() + 10)

    for peer in [alice, bob]:
        assert (await peer.get(b'experiment_name', latest=True)).value == b'foo_bar'


@pytest.mark.forked
@pytest.mark.asyncio
async def test_expecting_dictionary(dht_nodes_with_schema):
    alice, bob = dht_nodes_with_schema

    # Dictionary (bytes -> non-negative int) expected
    assert await bob.store(b'n_batches', 777, get_dht_time() + 10, subkey=b'uid1')
    assert await bob.store(b'n_batches', 778, get_dht_time() + 10, subkey=b'uid2')
    assert not await bob.store(b'n_batches', -666, get_dht_time() + 10, subkey=b'uid3')
    assert not await bob.store(b'n_batches', 666, get_dht_time() + 10)
    assert not await bob.store(b'n_batches', b'not_integer', get_dht_time() + 10, subkey=b'uid1')
    assert not await bob.store(b'n_batches', 666, get_dht_time() + 10, subkey=666)

    # Validate despite https://pydantic-docs.helpmanual.io/usage/models/#data-conversion
    assert not await bob.store(b'n_batches', 779.5, get_dht_time() + 10, subkey=b'uid3')
    assert not await bob.store(b'n_batches', 779.0, get_dht_time() + 10, subkey=b'uid3')
    assert not await bob.store(b'n_batches', [], get_dht_time() + 10)
    assert not await bob.store(b'n_batches', [(b'uid3', 779)], get_dht_time() + 10)

    # Validate despite https://github.com/samuelcolvin/pydantic/issues/1268
    assert not await bob.store(b'n_batches', '', get_dht_time() + 10)

    for peer in [alice, bob]:
        dictionary = (await peer.get(b'n_batches', latest=True)).value
        assert (len(dictionary) == 2 and
                dictionary[b'uid1'].value == 777 and
                dictionary[b'uid2'].value == 778)


@pytest.mark.forked
@pytest.mark.asyncio
async def test_expecting_public_keys(dht_nodes_with_schema):
    alice, bob = dht_nodes_with_schema

    # Subkeys expected to contain a public key
    # (so hivemind.dht.crypto.RSASignatureValidator would require a signature)
    assert await bob.store(b'signed_data', b'foo_bar', get_dht_time() + 10,
                           subkey=b'uid[owner:public-key]')
    assert not await bob.store(b'signed_data', b'foo_bar', get_dht_time() + 10,
                               subkey=b'uid-without-public-key')

    for peer in [alice, bob]:
        dictionary = (await peer.get(b'signed_data', latest=True)).value
        assert (len(dictionary) == 1 and
                dictionary[b'uid[owner:public-key]'].value == b'foo_bar')
