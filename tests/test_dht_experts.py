import random
import uuid
import numpy as np

import hivemind
from hivemind import LOCALHOST


def test_store_get_experts():
    peers = [hivemind.DHT(start=True)]
    for i in range(10):
        neighbors_i = [f'{LOCALHOST}:{node.port}' for node in random.sample(peers, min(3, len(peers)))]
        peers.append(hivemind.DHT(initial_peers=neighbors_i, start=True))

    you: hivemind.dht.DHT = random.choice(peers)
    theguyshetoldyounottoworryabout: hivemind.dht.DHT = random.choice(peers)

    expert_uids = [str(uuid.uuid4()) for _ in range(110)]
    batch_size = 10
    for batch_start in range(0, len(expert_uids), batch_size):
        you.declare_experts(expert_uids[batch_start: batch_start + batch_size], 'localhost', 1234)

    found = theguyshetoldyounottoworryabout.get_experts(random.sample(expert_uids, 5) + ['foo', 'bar'])
    assert all(res is not None for res in found[:-2]), "Could not find some existing experts"
    assert all(res is None for res in found[-2:]), "Found non-existing experts"

    that_guys_expert, that_guys_port = str(uuid.uuid4()), random.randint(1000, 9999)
    theguyshetoldyounottoworryabout.declare_experts([that_guys_expert], f'that_host:{that_guys_port}')
    you_notfound, you_found = you.get_experts(['foobar', that_guys_expert])
    assert isinstance(you_found, hivemind.RemoteExpert)
    assert you_found.endpoint == f'that_host:{that_guys_port}'

    for peer in peers:
        peer.shutdown()


def test_beam_search(dht_size=20, total_experts=128, batch_size=32, initial_peers=3, beam_size=4, parallel_rpc=256,
                     grid_dims=(32, 32, 32)):
    dht = []
    for i in range(dht_size):
        neighbors_i = [f'{LOCALHOST}:{node.port}' for node in random.sample(dht, min(initial_peers, len(dht)))]
        dht.append(hivemind.DHT(start=True, expiration=999999, initial_peers=neighbors_i, parallel_rpc=parallel_rpc))

    real_experts = sorted({
        'expert.' + '.'.join([str(random.randint(0, dim - 1)) for dim in grid_dims])
        for _ in range(total_experts)
    })
    for batch_start in range(0, len(real_experts), batch_size):
        random.choice(dht).declare_experts(
            real_experts[batch_start: batch_start + batch_size], wait=True,
            endpoint=f"host{batch_start // batch_size}:{random.randint(0, 65536)}")

    neighbors_i = [f'{LOCALHOST}:{node.port}' for node in random.sample(dht, min(initial_peers, len(dht)))]
    you = hivemind.DHT(start=True, expiration=999999, initial_peers=neighbors_i, parallel_rpc=parallel_rpc)

    for i in range(50):
        topk_experts = you.find_best_experts('expert', [np.random.randn(dim) for dim in grid_dims], beam_size=beam_size)
        assert all(isinstance(e, hivemind.RemoteExpert) for e in topk_experts)
        assert len(topk_experts) == beam_size

    for i in range(10):
        batch_experts = you.batch_find_best_experts('expert', [np.random.randn(batch_size, dim) for dim in grid_dims],
                                                    beam_size=beam_size)
        assert isinstance(batch_experts, list) and len(batch_experts) == batch_size
        assert all(isinstance(e, hivemind.RemoteExpert) for experts in batch_experts for e in experts)
        assert all(len(experts) == beam_size for experts in batch_experts)


def test_dht_single_node():
    node = hivemind.DHT(start=True)
    assert node.first_k_active(['e.3', 'e.2'], k=3) == {}
    assert node.get_experts(['e.3', 'e.2']) == [None, None]

    assert all(node.declare_experts(['e.1', 'e.2', 'e.3'], f"{hivemind.LOCALHOST}:1337"))
    for expert in node.get_experts(['e.3', 'e.2']):
        assert expert.endpoint == f"{hivemind.LOCALHOST}:1337"
    active_found = node.first_k_active(['e.0', 'e.1', 'e.3', 'e.5', 'e.2'], k=2)
    assert list(active_found.keys()) == ['e.1', 'e.3']
    assert all(expert.uid.startswith(prefix) for prefix, expert in active_found.items())

    assert all(node.declare_experts(['e.1', 'e.2', 'e.3'], f"{hivemind.LOCALHOST}:1337"))
    assert node.find_best_experts('e', [(0., 1., 2., 3., 4., 5., 6., 7., 8.)], beam_size=4)
