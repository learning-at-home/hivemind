import random
import re
from typing import List, NamedTuple, Optional, Tuple, Union

import hivemind
from hivemind.dht import DHT
from hivemind.utils import Endpoint, get_logger

logger = get_logger(__name__)

ExpertUID, ExpertPrefix, Coordinate, Score = str, str, int, float
UidEndpoint = NamedTuple("UidEndpoint", [("uid", ExpertUID), ("endpoint", Endpoint)])
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


def generate_uids_from_pattern(
    num_experts: int, expert_pattern: Optional[str], dht: Optional[DHT] = None, attempts_per_expert=10
) -> List[str]:
    """
    Sample experts from a given pattern, remove duplicates.
    :param num_experts: sample this many unique expert uids
    :param expert_pattern: a string pattern or a list of expert uids,  example: myprefix.[0:32].[0:256]\
     means "sample random experts between myprefix.0.0 and myprefix.255.255;
    :param dht: if specified, uses this DHT to check that expert uids are not yet occupied by other peers
    :param attempts_per_expert: give up if unable to generate a new expert uid after this many attempts per uid
    :note: this method is not strictly process-safe. If several servers run it concurrently, they have
     a small chance of sampling duplicate expert uids.
    """
    remaining_attempts = attempts_per_expert * num_experts
    found_uids, attempted_uids = list(), set()

    def _generate_uid():
        if expert_pattern is None:
            return f"expert{UID_DELIMITER}{attempts_per_expert * num_experts - remaining_attempts}"

        uid = []
        for block in expert_pattern.split(UID_DELIMITER):
            try:
                if "[" not in block and "]" not in block:
                    uid.append(block)
                elif block.startswith("[") and block.endswith("]") and ":" in block:
                    slice_start, slice_end = map(int, block[1:-1].split(":"))
                    uid.append(str(random.randint(slice_start, slice_end - 1)))
                else:
                    raise ValueError("Block must be either fixed or a range [from:to]")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                raise ValueError(f"Expert pattern {expert_pattern} has invalid block {block}, {e}")
        return UID_DELIMITER.join(uid)

    while remaining_attempts > 0 and len(found_uids) < num_experts:

        # 1. sample new expert uids at random
        new_uids = []
        while len(new_uids) + len(found_uids) < num_experts and remaining_attempts > 0:
            new_uid = _generate_uid()
            remaining_attempts -= 1
            if new_uid not in attempted_uids:
                attempted_uids.add(new_uid)
                new_uids.append(new_uid)

        # 2. look into DHT (if given) and remove duplicates
        if dht:
            existing_expert_uids = {
                found_expert.uid
                for found_expert in hivemind.moe.server.get_experts(dht, new_uids)
                if found_expert is not None
            }
            new_uids = [new_uid for new_uid in new_uids if new_uid not in existing_expert_uids]

        found_uids += new_uids

    if len(found_uids) != num_experts:
        logger.warning(
            f"Found only {len(found_uids)} out of {num_experts} free expert uids after "
            f"{attempts_per_expert * num_experts} attempts"
        )
    return found_uids
