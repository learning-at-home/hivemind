import operator
from random import random

from hivemind.dht.id import DHTNodeID
from hivemind.utils.serializer import PickleSerializer


def test_ids():
    # basic functionality tests
    for i in range(100):
        id1, id2 = DHTNodeID.generate(), DHTNodeID.generate()
        assert DHTNodeID.MIN <= id1 < DHTNodeID.MAX and DHTNodeID.MIN <= id2 <= DHTNodeID.MAX
        assert DHTNodeID.xor_distance(id1, id1) == DHTNodeID.xor_distance(id2, id2) == 0
        assert DHTNodeID.xor_distance(id1, id2) > 0 or (id1 == id2)
        assert len(PickleSerializer.dumps(id1)) - len(PickleSerializer.dumps(int(id1))) < 40

    # test depth (aka longest common prefix)
    for i in range(100):
        ids = [random.randint(0, 4096) for i in range(random.randint(1, 256))]
        ours = DHTNodeID.longest_common_prefix_length(*map(DHTNodeID, ids))

        ids_bitstr = [
            "".join(bin(bite)[2:].rjust(8, '0') for bite in uid.to_bytes(20, 'big'))
            for uid in ids
        ]
        reference = len(shared_prefix(*ids_bitstr))
        assert reference == ours, f"ours {ours} != reference {reference}, ids: {ids}"




def shared_prefix(*strings: str):
    for i in range(min(map(len, strings))):
        if len(set(map(operator.itemgetter(i), strings))) != 1:
            return strings[0][:i]
    return min(strings, key=len)
