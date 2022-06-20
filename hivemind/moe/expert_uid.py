from __future__ import annotations

import re
from typing import NamedTuple, Tuple, Union

from hivemind.p2p import PeerID

ExpertUID, ExpertPrefix, Coordinate, Score = str, str, int, float
ExpertInfo = NamedTuple("ExpertInfo", [("uid", ExpertUID), ("peer_id", PeerID)])
UID_DELIMITER = "."  # when declaring experts, DHT store all prefixes of that expert's uid, split over this prefix
FLAT_EXPERT = -1  # grid prefix reserved for storing 1d expert uids. Used to speed up find_best_experts in 1d case.
UID_PATTERN = re.compile("^(([^.])+)([.](?:[0]|([1-9]([0-9]*))))+$")  # e.g. ffn_expert.98.76.54 - prefix + some dims
PREFIX_PATTERN = re.compile("^(([^.])+)([.](?:[0]|([1-9]([0-9]*))))*[.]$")  # e.g. expert. or ffn.45. (ends with ".")
#  formally, prefixes = {uid.split(UID_DELIMITER)[:length] for length in range(1, uid.count(UID_DELIMITER) + 2)}


def is_valid_uid(maybe_uid: str) -> bool:
    """An uid must contain a string expert type, followed by one or more .-separated numeric indices"""
    return bool(UID_PATTERN.fullmatch(maybe_uid))


def is_valid_prefix(maybe_prefix: str) -> bool:
    """An uid prefix must contain a string expert type, followed by optional numeric indices and a trailing period"""
    return bool(PREFIX_PATTERN.fullmatch(maybe_prefix))


def split_uid(uid_or_prefix: Union[ExpertUID, ExpertPrefix]) -> Tuple[ExpertPrefix, Coordinate]:
    """Separate an expert UID or prefix into a new ExpertPrefix and integer for the last coordinate"""
    uid_or_prefix = uid_or_prefix.rstrip(UID_DELIMITER)
    pivot = uid_or_prefix.rindex(UID_DELIMITER) + 1
    return uid_or_prefix[:pivot], int(uid_or_prefix[pivot:])
