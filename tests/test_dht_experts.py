import random
import uuid
from itertools import chain

import hivemind
from hivemind import LOCALHOST


def test_hivemind_dht():
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

    # test first_k_active
    assert list(theguyshetoldyounottoworryabout.first_k_active(expert_uids, k=10)) == expert_uids[:10]

    some_permuted_experts = random.sample(expert_uids, k=32)
    assert list(theguyshetoldyounottoworryabout.first_k_active(some_permuted_experts, k=32)) == some_permuted_experts
    assert list(theguyshetoldyounottoworryabout.first_k_active(some_permuted_experts, k=1)) == some_permuted_experts[:1]
    fake_and_real_experts = list(chain(*zip(
        [str(uuid.uuid4()) for _ in some_permuted_experts], some_permuted_experts)))
    assert list(theguyshetoldyounottoworryabout.first_k_active(fake_and_real_experts, k=9)) == some_permuted_experts[:9]

    for peer in peers:
        peer.shutdown()


def test_first_k_active():
    node = hivemind.DHT(start=True)
    assert all(node.declare_experts(['e.1.2.3', 'e.1.2.4', 'e.3.4.5'], endpoint=f"{hivemind.LOCALHOST}:1337"))
    assert all(node.declare_experts(['e.2.1.1'], endpoint=f"{hivemind.LOCALHOST}:1338"))

    results = node.first_k_active(['e.0', 'e.1', 'e.2', 'e.3'], k=2)
    assert len(results) == 2 and next(iter(results.keys())) == 'e.1'
    assert results['e.1'].uid in ('e.1.2.3', 'e.1.2.4') and results['e.1'].endpoint == f"{hivemind.LOCALHOST}:1337"
    assert results['e.2'].uid == 'e.2.1.1' and results['e.2'].endpoint == f"{hivemind.LOCALHOST}:1338"

    results = node.first_k_active(['e', 'e.1', 'e.1.2', 'e.1.2.3'], k=10)
    assert len(results) == 4
    assert 'e' in results
    for k in ('e.1', 'e.1.2', 'e.1.2.3'):
        assert results[k].uid in ('e.1.2.3', 'e.1.2.4') and results[k].endpoint == f"{hivemind.LOCALHOST}:1337"


def test_dht_single_node():
    node = hivemind.DHT(start=True)
    assert node.first_k_active(['e3', 'e2'], k=3) == {}
    assert node.get_experts(['e3', 'e2']) == [None, None]

    assert all(node.declare_experts(['e1', 'e2', 'e3'], f"{hivemind.LOCALHOST}:1337"))
    for expert in node.get_experts(['e3', 'e2']):
        assert expert.endpoint == f"{hivemind.LOCALHOST}:1337"
    active_found = node.first_k_active(['e0', 'e1', 'e3', 'e5', 'e2'], k=2)
    assert list(active_found.keys()) == ['e1', 'e3']
    assert all(expert.uid.startswith(prefix) for prefix, expert in active_found.items())

    assert all(node.declare_experts(['e1', 'e2', 'e3'], f"{hivemind.LOCALHOST}:1337"))
