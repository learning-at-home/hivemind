import asyncio
import random
import time

import psutil
import numpy as np
import pytest

import hivemind
import hivemind.server.expert_uid
from hivemind import LOCALHOST, declare_experts, get_experts
from hivemind.client.beam_search import MoEBeamSearcher
from hivemind.server.expert_uid import UidEndpoint, is_valid_uid, is_valid_prefix, split_uid
from test_utils.fixtures import cleanup_children


@pytest.mark.forked
def test_store_get_experts():
    peers = [hivemind.DHT(start=True)]
    for i in range(10):
        neighbors_i = [f'{LOCALHOST}:{node.port}' for node in random.sample(peers, min(3, len(peers)))]
        peers.append(hivemind.DHT(initial_peers=neighbors_i, start=True))

    first_peer = random.choice(peers)
    other_peer = random.choice(peers)

    expert_uids = [f"my_expert.{i}" for i in range(50)]
    batch_size = 10
    for batch_start in range(0, len(expert_uids), batch_size):
        hivemind.declare_experts(first_peer, expert_uids[batch_start: batch_start + batch_size], 'localhost:1234')

    found = get_experts(other_peer, random.sample(expert_uids, 5) + ['foo', 'bar'])
    assert all(res is not None for res in found[:-2]), "Could not find some existing experts"
    assert all(res is None for res in found[-2:]), "Found non-existing experts"

    other_expert, other_port = "my_other_expert.1337", random.randint(1000, 9999)
    hivemind.declare_experts(other_peer, [other_expert], f'that_host:{other_port}')
    first_notfound, first_found = get_experts(first_peer, ['foobar', other_expert])
    assert isinstance(first_found, hivemind.RemoteExpert)
    assert first_found.endpoint == f'that_host:{other_port}'

    # test graceful shutdown
    first_peer.shutdown()
    other_peer.shutdown()
    time.sleep(1.0)
    assert sum(peer.is_alive() for peer in peers) == len(peers) - 2
    remaining_peer1 = random.choice([peer for peer in peers if peer.is_alive()])
    remaining_peer2 = random.choice([peer for peer in peers if peer.is_alive()])
    assert all(hivemind.declare_experts(remaining_peer1, ['new_expert.1'], 'dummy'))
    assert hivemind.get_experts(remaining_peer2, ['new_expert.1'])[0].endpoint == 'dummy'




@pytest.mark.forked
def test_beam_search(dht_size=20, total_experts=128, batch_size=32, initial_peers=3, beam_size=4, parallel_rpc=4,
                     grid_dims=(32, 32, 32)):
    dht = []
    for i in range(dht_size):
        neighbors_i = [f'{LOCALHOST}:{node.port}' for node in random.sample(dht, min(initial_peers, len(dht)))]
        dht.append(hivemind.DHT(start=True, initial_peers=neighbors_i, parallel_rpc=parallel_rpc))

    real_experts = sorted({
        'expert.' + '.'.join([str(random.randint(0, dim - 1)) for dim in grid_dims])
        for _ in range(total_experts)
    })
    for batch_start in range(0, len(real_experts), batch_size):
        declare_experts(random.choice(dht), real_experts[batch_start: batch_start + batch_size], wait=True,
                        endpoint=f"host{batch_start // batch_size}:{random.randint(0, 65536)}")

    neighbors_i = [f'{LOCALHOST}:{node.port}' for node in random.sample(dht, min(initial_peers, len(dht)))]
    you = hivemind.DHT(start=True, initial_peers=neighbors_i, parallel_rpc=parallel_rpc)
    beam_search = MoEBeamSearcher(you, 'expert.', grid_dims)

    for i in range(10):
        topk_experts = beam_search.find_best_experts([np.random.randn(dim) for dim in grid_dims], beam_size)
        assert all(isinstance(e, hivemind.RemoteExpert) for e in topk_experts)
        assert len(topk_experts) == beam_size

    for i in range(10):
        batch_experts = beam_search.batch_find_best_experts([np.random.randn(batch_size, dim) for dim in grid_dims],
                                                            beam_size=beam_size)
        assert isinstance(batch_experts, list) and len(batch_experts) == batch_size
        assert all(isinstance(e, hivemind.RemoteExpert) for experts in batch_experts for e in experts)
        assert all(len(experts) == beam_size for experts in batch_experts)


@pytest.mark.forked
def test_dht_single_node():
    node = hivemind.DHT(start=True)
    beam_search = MoEBeamSearcher(node, 'expert.', grid_size=(10,))

    assert all(declare_experts(node, ['expert.1', 'expert.2', 'expert.3'], f"{hivemind.LOCALHOST}:1337").values())
    assert len(declare_experts(node, ["ffn.1", "ffn.2"], endpoint="that_place")) == 4
    assert len(declare_experts(node, ['e.1.2.3', 'e.1.2.5', 'e.2.0'], f"{hivemind.LOCALHOST}:42")) == 7

    for expert in get_experts(node, ['expert.3', 'expert.2']):
        assert expert.endpoint == f"{hivemind.LOCALHOST}:1337"

    assert all(declare_experts(node, ['expert.5', 'expert.2'], f"{hivemind.LOCALHOST}:1337").values())
    found_experts = beam_search.find_best_experts([(0., 1., 2., 3., 4., 5., 6., 7., 8.)], beam_size=2)
    assert len(found_experts) == 2 and [expert.uid for expert in found_experts] == ['expert.5', 'expert.3']

    successors = beam_search.get_active_successors(['e.1.2.', 'e.2.', 'e.4.5.'])
    assert len(successors['e.1.2.']) == 2
    assert successors['e.1.2.'][3] == UidEndpoint('e.1.2.3', f'{LOCALHOST}:42')
    assert successors['e.1.2.'][5] == UidEndpoint('e.1.2.5', f'{LOCALHOST}:42')
    assert len(successors['e.2.']) == 1 and successors['e.2.'][0] == UidEndpoint('e.2.0', f'{LOCALHOST}:42')
    assert successors['e.4.5.'] == {}

    initial_beam = beam_search.get_initial_beam((3, 2, 1, 0, -1, -2, -3), beam_size=3)
    assert len(initial_beam) == 3
    assert initial_beam[0][:2] == (2.0, 'expert.1.')
    assert initial_beam[1][:2] == (1.0, 'expert.2.')
    assert initial_beam[2][:2] == (0.0, 'expert.3.')

    with pytest.raises(AssertionError):
        beam_search = MoEBeamSearcher(node, 'expert.1.ffn', (2, 2))

    with pytest.raises(AssertionError):
        beam_search.get_active_successors(['e.1.2.', 'e.2', 'e.4.5.'])


def test_uid_patterns():
    valid_experts = ["expert.1", "expert.0", "expert.0.0.1", "expert.1337", "ffn.12.34.56.78.90",
                     "transformer.3.2.1.0", "transformer_encoder.2", "transformer::encoder.2", "T®@nsf0rmE®🤗.321",
                     "🤗.321", "0.1.2", "00.1.2", "7070.3.2.1.0", "block2.1.23", "LAYER.1.0.1"]
    valid_prefixes = ["expert.", "e.1.", "e.2.", "e.1.2.3.", "ololo.123.456.789.10."]
    valid_prefixes.extend([f"{uid}." for uid in valid_experts])
    valid_prefixes.extend([split_uid(uid)[0] for uid in valid_experts])
    for uid in valid_experts:
        assert is_valid_uid(uid), f"UID {uid} is valid, but was perceived as invalid"
    for pfx in valid_prefixes:
        assert is_valid_prefix(pfx), f"Prefix {pfx} is valid, but was perceived as invalid"

    invalid = ["", ".", "expert.-1", "xxx.a", "expert.1x", "expert_ffn.1.abc1", "some.123.01", "expert.123.01",
               "e1", "e..1", "e", "e.1.2.3..4", "ffn.1..1", ".123", ".1.2.3.", ".expert", "transformer.encoder.2",
               "T®@nsf0rmE®.🤗.321", "layer::123", "expert.0.1.2.suffix", "0.1.2.suffix", "expert.1 something",
               "expert.1\n", "expert.1\n2", "expert.1 ", "expert.1\nexpert.2", "'expert.1'", '"expert.1"']
    invalid_experts = invalid + valid_prefixes + ["0", "123456"]
    invalid_prefixes = invalid + valid_experts + ["expert", ".🤗", ".expert"]
    for uid in invalid_experts:
        assert not is_valid_uid(uid), f"UID {uid} is not valid, but was perceived as valid"
    for pfx in invalid_prefixes:
        assert not is_valid_prefix(pfx), f"Prefix {pfx} is not valid, but was perceived as valid"


@pytest.mark.forked
@pytest.mark.asyncio
async def test_negative_caching():
    peers = []
    for i in range(10):
        neighbors_i = [f'{LOCALHOST}:{node.port}' for node in random.sample(peers, min(3, len(peers)))]
        peers.append(hivemind.DHT(initial_peers=neighbors_i, cache_locally=False, start=True))

    writer_peer = random.choice(peers)
    assert all(hivemind.declare_experts(writer_peer, ['ffn.1.2.3', 'ffn.3.4.5'], 'myaddr:1234').values())

    neighbors_i = [f'{LOCALHOST}:{node.port}' for node in random.sample(peers, min(3, len(peers)))]
    neg_caching_peer = hivemind.DHT(initial_peers=neighbors_i, cache_locally=False, start=True)
    beam_search = MoEBeamSearcher(neg_caching_peer, uid_prefix='ffn.', grid_size=(10, 10, 10), negative_caching=True)
    # get prefixes by the peer with negative caching. Cache "no data" entries for ffn.0.*, ffn.2.*, ffn.4.*, ffn.5.*
    assert len(beam_search.get_initial_beam(scores=[.1, .2, .3, .4, .5, .6], beam_size=3)) == 2

    node = await hivemind.DHTNode.create(initial_peers=neighbors_i)
    fetched = await asyncio.gather(*(node.get(f'ffn.{i}.') for i in range(10)))
    for i in range(6):
        assert fetched[i] is not None, f"node should have cached ffn.{i}."
    for i in range(6, len(fetched)):
        assert fetched[i] is None, f"node shouldn't have cached ffn.{i}."

    await node.shutdown()
    neg_caching_peer.shutdown()
    for peer in peers:
        peer.shutdown()
