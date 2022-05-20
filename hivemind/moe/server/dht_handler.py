import threading
from functools import partial
from multiaddr import Multiaddr
from typing import Union, Dict, List, Optional, Sequence, Tuple

from hivemind.dht import DHT, DHTExpiration, DHTNode, DHTValue
from hivemind.moe.client.expert import RemoteExpert, _RemoteModuleCall
from hivemind.moe.server.expert_uid import (
    FLAT_EXPERT,
    UID_DELIMITER,
    UID_PATTERN,
    Coordinate,
    ExpertPrefix,
    ExpertUID,
    is_valid_uid,
    split_uid,
)
from hivemind.p2p import PeerID, PeerInfo
from hivemind.utils import get_dht_time, LazyFutureCaller, LazyValue


class DHTHandlerThread(threading.Thread):
    def __init__(self, experts, dht: DHT, peer_id: PeerID, update_period: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.peer_id = peer_id
        self.experts = experts
        self.dht = dht
        self.update_period = update_period
        self.stop = threading.Event()

    def run(self) -> None:
        declare_experts(self.dht, self.experts.keys(), self.peer_id)
        while not self.stop.wait(self.update_period):
            declare_experts(self.dht, self.experts.keys(), self.peer_id)


def declare_experts(
    dht: DHT, uids: Sequence[ExpertUID], peer_id: PeerID, expiration: DHTExpiration = 300, wait: bool = True
) -> Dict[ExpertUID, bool]:
    """
    Make experts visible to all DHT peers; update timestamps if declared previously.

    :param uids: a list of expert ids to update
    :param endpoint: endpoint that serves these experts, usually your server endpoint (e.g. "201.111.222.333:1337")
    :param wait: if True, awaits for declaration to finish, otherwise runs in background
    :param expiration: experts will be visible for this many seconds
    :returns: if wait, returns store status for every key (True = store succeeded, False = store rejected)
    """
    assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
    for uid in uids:
        assert is_valid_uid(uid), f"{uid} is not a valid expert uid. All uids must follow {UID_PATTERN.pattern}"
    addrs = tuple(str(a.decapsulate("/p2p/" + a.get("p2p"))) for a in dht.get_visible_maddrs())
    return dht.run_coroutine(
        partial(_declare_experts, uids=list(uids), peer_id=peer_id, addrs=addrs, expiration=expiration), return_future=not wait
    )


async def _declare_experts(
    dht: DHT, node: DHTNode, uids: List[ExpertUID], peer_id: PeerID, addrs: Tuple[str], expiration: DHTExpiration
) -> Dict[ExpertUID, bool]:
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    expiration_time = get_dht_time() + expiration
    data_to_store: Dict[Tuple[ExpertPrefix, Optional[Coordinate]], DHTValue] = {}
    for uid in uids:
        data_to_store[uid, None] = (peer_id.to_base58(), addrs)
        prefix = uid if uid.count(UID_DELIMITER) > 1 else f"{uid}{UID_DELIMITER}{FLAT_EXPERT}"
        for i in range(prefix.count(UID_DELIMITER) - 1):
            prefix, last_coord = split_uid(prefix)
            data_to_store[prefix, last_coord] = [uid, (peer_id.to_base58(), addrs)]

    keys, maybe_subkeys, values = zip(*((key, subkey, value) for (key, subkey), value in data_to_store.items()))
    store_ok = await node.store_many(keys, values, expiration_time, subkeys=maybe_subkeys, num_workers=num_workers)
    return store_ok


def get_experts(
    dht: DHT, uids: List[ExpertUID], expiration_time: Optional[DHTExpiration] = None, return_future: bool = False
) -> Union[List[Optional[RemoteExpert]], LazyFutureCaller[Optional[LazyValue[RemoteExpert]], Optional[RemoteExpert]]]:
    """
    :param uids: find experts with these ids from across the DHT
    :param expiration_time: if specified, return experts that expire no sooner than this (based on get_dht_time)
    :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
    :returns: a list of [RemoteExpert if found else None]
    """
    assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
    p2p = _RemoteModuleCall.run_coroutine(dht.replicate_p2p())
    result = dht.run_coroutine(partial(_get_experts, uids=list(uids), expiration_time=expiration_time), return_future)

    def _unwrap_experts(vals: List[Optional[LazyValue[RemoteExpert]]]) -> List[Optional[RemoteExpert]]:
        return [val.get(p2p=p2p) if val is not None else None for val in vals]

    if return_future:
        return LazyFutureCaller(result, _unwrap_experts)
    return _unwrap_experts(result)


async def _get_experts(
    dht: DHT, node: DHTNode, uids: List[ExpertUID], expiration_time: Optional[DHTExpiration]
) -> List[Optional[LazyValue[RemoteExpert]]]:
    if expiration_time is None:
        expiration_time = get_dht_time()
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    found: Dict[ExpertUID, DHTValue] = await node.get_many(uids, expiration_time, num_workers=num_workers)

    experts: List[Optional[RemoteExpert]] = [None] * len(uids)
    for i, uid in enumerate(uids):
        if (elem := found[uid]) is not None and \
            isinstance(elem.value, tuple):
            peer_id, addrs = elem.value
            peer_info = PeerInfo(peer_id=PeerID.from_base58(peer_id), addrs=tuple(Multiaddr(a) for a in addrs))
            experts[i] = LazyValue(init=partial(RemoteExpert, uid=uid, server_peer_info=peer_info))
    return experts
