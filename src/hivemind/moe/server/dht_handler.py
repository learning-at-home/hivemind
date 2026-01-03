import threading
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

from hivemind.dht import DHT, DHTNode, DHTValue
from hivemind.moe.client.expert import RemoteExpert, create_remote_experts
from hivemind.moe.expert_uid import (
    FLAT_EXPERT,
    UID_DELIMITER,
    UID_PATTERN,
    Coordinate,
    ExpertInfo,
    ExpertPrefix,
    ExpertUID,
    is_valid_uid,
    split_uid,
)
from hivemind.p2p import PeerID
from hivemind.utils import MAX_DHT_TIME_DISCREPANCY_SECONDS, DHTExpiration, MPFuture, get_dht_time


class DHTHandlerThread(threading.Thread):
    def __init__(
        self, module_backends, dht: DHT, update_period: float = 30, expiration: Optional[int] = None, **kwargs
    ):
        super().__init__(**kwargs)
        if expiration is None:
            expiration = max(2 * update_period, MAX_DHT_TIME_DISCREPANCY_SECONDS)
        self.module_backends = module_backends
        self.dht = dht
        self.update_period = update_period
        self.expiration = expiration
        self.stop = threading.Event()

    def run(self) -> None:
        declare_experts(self.dht, self.module_backends.keys(), expiration_time=get_dht_time() + self.expiration)
        while not self.stop.wait(self.update_period):
            declare_experts(self.dht, self.module_backends.keys(), expiration_time=get_dht_time() + self.expiration)


def declare_experts(
    dht: DHT, uids: Sequence[ExpertUID], expiration_time: DHTExpiration, wait: bool = True
) -> Union[Dict[ExpertUID, bool], MPFuture[Dict[ExpertUID, bool]]]:
    """
    Make experts visible to all DHT peers; update timestamps if declared previously.

    :param uids: a list of expert ids to update
    :param wait: if True, awaits for declaration to finish, otherwise runs in background
    :param expiration_time: experts will be visible for this many seconds
    :returns: if wait, returns store status for every key (True = store succeeded, False = store rejected)
    """
    assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
    if not isinstance(uids, list):
        uids = list(uids)
    for uid in uids:
        assert is_valid_uid(uid), f"{uid} is not a valid expert uid. All uids must follow {UID_PATTERN.pattern}"
    return dht.run_coroutine(
        partial(_declare_experts, uids=uids, expiration_time=expiration_time), return_future=not wait
    )


async def _declare_experts(
    dht: DHT, node: DHTNode, uids: List[ExpertUID], expiration_time: DHTExpiration
) -> Dict[ExpertUID, bool]:
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    data_to_store: Dict[Tuple[ExpertPrefix, Optional[Coordinate]], DHTValue] = {}
    peer_id_base58 = dht.peer_id.to_base58()

    for uid in uids:
        data_to_store[uid, None] = peer_id_base58
        prefix = uid if uid.count(UID_DELIMITER) > 1 else f"{uid}{UID_DELIMITER}{FLAT_EXPERT}"
        for i in range(prefix.count(UID_DELIMITER) - 1):
            prefix, last_coord = split_uid(prefix)
            data_to_store[prefix, last_coord] = (uid, peer_id_base58)

    keys, maybe_subkeys, values = zip(*((key, subkey, value) for (key, subkey), value in data_to_store.items()))
    store_ok = await node.store_many(keys, values, expiration_time, subkeys=maybe_subkeys, num_workers=num_workers)
    return store_ok


def get_experts(
    dht: DHT, uids: List[ExpertUID], expiration_time: Optional[DHTExpiration] = None, return_future: bool = False
) -> Union[List[Optional[RemoteExpert]], MPFuture[List[Optional[RemoteExpert]]]]:
    """
    :param uids: find experts with these ids from across the DHT
    :param expiration_time: if specified, return experts that expire no sooner than this (based on get_dht_time)
    :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
    :returns: a list of [RemoteExpert if found else None]
    """
    assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
    result = dht.run_coroutine(partial(_get_experts, uids=list(uids), expiration_time=expiration_time), return_future)
    return create_remote_experts(result, dht, return_future)


async def _get_experts(
    dht: DHT, node: DHTNode, uids: List[ExpertUID], expiration_time: Optional[DHTExpiration]
) -> List[Optional[ExpertInfo]]:
    if expiration_time is None:
        expiration_time = get_dht_time()
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    found: Dict[ExpertUID, DHTValue] = await node.get_many(uids, expiration_time, num_workers=num_workers)

    experts: List[Optional[ExpertInfo]] = [None] * len(uids)
    for i, uid in enumerate(uids):
        server_peer_id = found[uid]
        if server_peer_id is not None and isinstance(server_peer_id.value, str):
            experts[i] = ExpertInfo(uid, PeerID.from_base58(server_peer_id.value))
    return experts
