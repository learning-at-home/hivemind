import pytest

from hivemind.dht import get_dht_time
from hivemind.dht.node import DHTNode, LOCALHOST
from hivemind.dht.schema import SchemaValidator


@pytest.mark.forked
@pytest.mark.asyncio
async def test_schema_validator():
    validator = SchemaValidator({
        b'experiment_name': {'type': 'binary'},
        b'n_batches': {'type': 'dict',
                       'keysrules': {'type': 'binary'},
                       'valuesrules': {'type': 'integer', 'min': 0}},
        b'signed_data': {'type': 'dict',
                         'keysrules': {'type': 'binary', 'regex': rb'^.*\[owner:.+\].*$'}},
    })

    alice = await DHTNode.create(record_validator=validator)
    bob = await DHTNode.create(
        record_validator=validator, initial_peers=[f"{LOCALHOST}:{alice.port}"])

    # test 1: Unknown keys are allowed
    assert await bob.store(b'unknown_key', b'foo_bar', get_dht_time() + 10)
    for peer in [alice, bob]:
        assert (await peer.get(b'unknown_key', latest=True)).value == b'foo_bar'

    # test 2: Regular value (bytes) expected
    assert await bob.store(b'experiment_name', b'foo_bar', get_dht_time() + 10)
    assert not await bob.store(b'experiment_name', 666, get_dht_time() + 10)
    assert not await bob.store(b'experiment_name', b'foo_bar', get_dht_time() + 10,
                               subkey=b'subkey')
    for peer in [alice, bob]:
        assert (await peer.get(b'experiment_name', latest=True)).value == b'foo_bar'

    # test 3: Dictionary (bytes -> int) expected
    assert await bob.store(b'n_batches', 777, get_dht_time() + 10, subkey=b'uid1')
    assert await bob.store(b'n_batches', 778, get_dht_time() + 10, subkey=b'uid2')
    assert not await bob.store(b'n_batches', -666, get_dht_time() + 10, subkey=b'uid3')
    assert not await bob.store(b'n_batches', 666, get_dht_time() + 10)
    assert not await bob.store(b'n_batches', b'not_integer', get_dht_time() + 10, subkey=b'uid1')
    assert not await bob.store(b'n_batches', 666, get_dht_time() + 10, subkey=666)
    for peer in [alice, bob]:
        dictionary = (await peer.get(b'n_batches', latest=True)).value
        assert (len(dictionary) == 2 and
                dictionary[b'uid1'].value == 777 and
                dictionary[b'uid2'].value == 778)

    # test 4: Subkeys expected to contain a public key
    # (so hivemind.dht.crypto.RSASignatureValidator would require a signature)
    assert await bob.store(b'signed_data', b'foo_bar', get_dht_time() + 10,
                           subkey=b'uid[owner:public-key]')
    assert not await bob.store(b'signed_data', b'foo_bar', get_dht_time() + 10,
                               subkey=b'uid-without-public-key')
    for peer in [alice, bob]:
        dictionary = (await peer.get(b'signed_data', latest=True)).value
        assert (len(dictionary) == 1 and
                dictionary[b'uid[owner:public-key]'].value == b'foo_bar')
