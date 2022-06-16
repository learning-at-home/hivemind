import asyncio
import heapq
from collections import deque
from functools import partial
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from hivemind.dht import DHT, DHTNode
from hivemind.moe.client.expert import RemoteExpert, batch_create_remote_experts, create_remote_experts
from hivemind.moe.expert_uid import (
    FLAT_EXPERT,
    PREFIX_PATTERN,
    UID_DELIMITER,
    Coordinate,
    ExpertInfo,
    ExpertPrefix,
    ExpertUID,
    Score,
    is_valid_prefix,
    is_valid_uid,
)
from hivemind.p2p import PeerID
from hivemind.utils import DHTExpiration, MPFuture, ValueWithExpiration, get_dht_time, get_logger

logger = get_logger(__name__)


class MoEBeamSearcher:
    """
    Utility class that uses DHT to find most suitable experts for RemoteMixtureOfExperts.
    Each expert has an identifier in the form of {prefix}.{i}.{j}.{...}, e.g. "ffn_expert.98.76.54.32.10"
    An expert identifier consists of:

        * optional prefix that determines expert role, experiment name, etc.
        * one or more integers that determine that expert's position in an N-dimensional grid

    A hivemind.moe.Server can ``declare_experts(dht, expert_uids: List[str])`` to make its experts visible to everyone.
    When declaring experts, DHT will store each expert's uid and all its prefixes until :expiration: (specified at init)
    For instance, declaring "ffn_expert.98.76.54.32.10" will store the following keys in a DHT:
    ``"ffn_expert.98", "ffn_expert.98.76", "ffn_expert.98.76.54", ..., "ffn_expert.98.76.54.32.10"``

    In order to enable fast beam search, DHT maintains dictionaries of all active suffixes for every prefix
    (e.g. "ffn_expert.98": {76: ffn_expert.98.76...., 123: ffn_expert.98.123..., 225: ffn_expert.98.225....}))

    RemoteMixtureOfExperts can use these prefixes to find top-k most suitable experts with a left-to-right beam search.
    For instance, consider RemoteMixtureOfExperts with prefix "ffn_expert" and grid size [100, 100, 100, 100, 100].
    This MoE can query all experts with that prefix and arbitrary indices in 0...99 along each dimension.
    However, not every expert in such 100^5 grid can be alive at a given moment of time (the grid size is redundant).
    In order to find k best "alive" experts, MoE first ranks indices along the first dimension with its gating function.
    It can then check which of those indices correspond to "alive" experts by querying keys such as "ffn_expert.98".

    After selecting k best indices along first dimension, MoE moves to the second dimension.
    It can find top-k index pairs (e.g. "expert.98.76") that use one of k best indices from the previous step.
    This beam search explores one additional dimension per step and finds k best experts from across the DHT
    in O(k * num_dimensions * dimension_size) time depending on the chosen grid dimensions.

    :param dht: a running DHT daemon that is used for beam search AND local caching
    :param uid_prefix: search for experts whose uids start with this prefix
    :param grid_size: dimensions that form expert uid (see above)
    :param num_workers: number of concurrent DHT coroutines per beam search
    :param negative_caching: if True, whenever DHT is unable to find an expert or prefix, it will cache the "no key"
      result inside the DHT for :expiration: seconds. Caching only affects beam search and has three main effects:

      1. Faster beam search under node failures: if there are inconsistencies in DHT keys, such as a prefix pointing to
         a now-defunct expert, these inconsistencies will be overwritten by the first peer that stumbles upon them. As a
         result, beam search will not have to wait for non-existent experts until the expiration of their DHT entries;
      2. Delayed expert availability: Without negative cache, new experts are always immediately available for beam
         search after they are published to the DHT. With negative cache, there are rare cases (e.g. when adding new
         experts in place of recently defunct ones) when new experts will be initially invisible, but gradually become
         visible to more peers as those peers refresh their cache. This process takes at most :expiration: seconds;
      3. Faster beam search in very sparse grids: there is one edge case where negative cache will improve beam search
         performance; If an expert grid is very sparse, there can be empty indices in the first grid dimension (i.e.
         indices {i} such that _no_ experts that start with "{prefix}.{i}.*"). If so, the default beam search will
         be very slow due to the way it forms initial beam. Beam search with negative cache enabled will run normally.
         Though, this is a pathological case (e.g. only 90 experts in an oversized 100x100 grid) that should be avoided.
    """

    def __init__(
        self,
        dht: DHT,
        uid_prefix: ExpertPrefix,
        grid_size: Sequence[int],
        num_workers: Optional[int] = None,
        negative_caching: bool = True,
        cache_expiration: DHTExpiration = 300,
        **kwargs,
    ):
        if not uid_prefix.endswith(UID_DELIMITER):
            uid_prefix += UID_DELIMITER
            logger.info(f"Prefix must end with '{UID_DELIMITER}'. Changing to {uid_prefix}{UID_DELIMITER}")
        assert is_valid_prefix(uid_prefix), f"Prefix '{uid_prefix}' is invalid."
        self.dht = dht
        self.uid_prefix, self.grid_size = uid_prefix, grid_size
        self.total_grid_size = sum(grid_size)
        self.negative_caching, self.cache_expiration = negative_caching, cache_expiration
        self.num_workers, self.dht_kwargs = num_workers, kwargs

    def get_initial_beam(
        self, scores: Sequence[float], beam_size: int, return_future: bool = False
    ) -> List[Tuple[Score, ExpertPrefix, Dict[Coordinate, ExpertInfo]]]:
        """
        :param scores: prefer suffix coordinates that have highest scores
        :param beam_size: select this many active suffixes with highest scores
        :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
        :returns: a list of up to beam_size tuples of (prefix score, prefix itself, dict{suffix: example expert})
        """
        return self.dht.run_coroutine(
            partial(
                self._get_initial_beam,
                prefix=self.uid_prefix,
                beam_size=beam_size,
                scores=tuple(scores),
                negative_caching=self.negative_caching,
                cache_expiration=self.cache_expiration,
                num_workers=self.num_workers,
            ),
            return_future,
        )

    @staticmethod
    async def _get_initial_beam(
        dht: DHT,
        node: DHTNode,
        prefix: ExpertPrefix,
        beam_size: int,
        scores: Tuple[float, ...],
        negative_caching: bool,
        cache_expiration: DHTExpiration,
        num_workers: Optional[int] = None,
    ) -> List[Tuple[Score, ExpertPrefix, Dict[Coordinate, ExpertInfo]]]:
        num_workers = num_workers or dht.num_workers or beam_size
        beam: List[Tuple[Score, ExpertPrefix, Dict[Coordinate, ExpertInfo]]] = []
        unattempted_indices: List[Coordinate] = sorted(
            range(len(scores)), key=scores.__getitem__
        )  # from worst to best
        pending_tasks: Deque[Tuple[Coordinate, ExpertPrefix, asyncio.Task]] = deque()

        while len(beam) < beam_size and (unattempted_indices or pending_tasks):
            # dispatch additional tasks
            while unattempted_indices and len(pending_tasks) < num_workers:
                next_index = unattempted_indices.pop()  # note: this is best unattempted index because of sort order
                next_best_prefix = f"{prefix}{next_index}{UID_DELIMITER}"
                pending_tasks.append((next_index, next_best_prefix, asyncio.create_task(node.get(next_best_prefix))))

            # await the next best prefix to be fetched
            pending_best_index, pending_best_prefix, pending_task = pending_tasks.popleft()
            try:
                maybe_prefix_data = await pending_task
                if maybe_prefix_data is not None and isinstance(maybe_prefix_data.value, dict):
                    successors = MoEBeamSearcher._select_valid_entries(maybe_prefix_data)
                    if successors:
                        beam.append((scores[pending_best_index], pending_best_prefix, successors))
                elif maybe_prefix_data is None and negative_caching:
                    logger.debug(f"DHT negative caching: storing a 'no prefix' entry for {pending_best_prefix}")
                    asyncio.create_task(
                        node.store(
                            pending_best_prefix,
                            subkey=-1,
                            value=None,
                            expiration_time=get_dht_time() + cache_expiration,
                        )
                    )

            except asyncio.CancelledError:
                for _, pending_task in pending_tasks:
                    pending_task.cancel()
                raise
        return beam

    def get_active_successors(
        self, prefixes: List[ExpertPrefix], grid_size: Optional[int] = None, return_future: bool = False
    ) -> Dict[ExpertPrefix, Dict[Coordinate, ExpertInfo]]:
        """
        :param prefixes: a list of prefix for which to find active successor uids
        :param grid_size: if specified, only return successors if ther are in range [0, grid_size)
        :param return_future: if False (default), find and return successors. Otherwise return MPFuture and fill later.
        :returns: for every expert, return a dict{active_next_coordinate: (matching_expert_uid, matching_endpoint)}
        :note: if a prefix is not found, get_active_successors will return an empty dictionary for that prefix
        """
        assert not isinstance(prefixes, str), "Please send a list / tuple of expert prefixes."
        for prefix in prefixes:
            assert is_valid_prefix(prefix), f"prefix '{prefix}' is invalid, it must follow {PREFIX_PATTERN.pattern}"
        return self.dht.run_coroutine(
            partial(
                self._get_active_successors,
                prefixes=list(prefixes),
                grid_size=grid_size,
                negative_caching=self.negative_caching,
                cache_expiration=self.cache_expiration,
                num_workers=self.num_workers,
            ),
            return_future=return_future,
        )

    @staticmethod
    def _select_valid_entries(entry: ValueWithExpiration, grid_size: Optional[int] = None):
        if not isinstance(entry, ValueWithExpiration) or not isinstance(entry.value, dict):
            return {}
        return {
            coord: ExpertInfo(uid=match.value[0], peer_id=PeerID.from_base58(match.value[1]))
            for coord, match in entry.value.items()
            if isinstance(coord, Coordinate)
            and (grid_size is None or 0 <= coord < grid_size)
            and isinstance(match, ValueWithExpiration)
            and isinstance(match.value, tuple)
            and len(match.value) == 2
            and is_valid_uid(match.value[0])
            and isinstance(match.value[1], str)
        }

    @staticmethod
    async def _get_active_successors(
        dht: DHT,
        node: DHTNode,
        prefixes: List[ExpertPrefix],
        grid_size: Optional[int],
        negative_caching: bool,
        cache_expiration: DHTExpiration,
        num_workers: Optional[int] = None,
    ) -> Dict[ExpertPrefix, Dict[Coordinate, ExpertInfo]]:
        grid_size = grid_size or float("inf")
        num_workers = num_workers or min(len(prefixes), dht.num_workers or len(prefixes))
        dht_responses = await node.get_many(keys=prefixes, num_workers=num_workers)
        successors: Dict[ExpertPrefix, Dict[Coordinate, ExpertInfo]] = {}
        for prefix, found in dht_responses.items():
            successors[prefix] = MoEBeamSearcher._select_valid_entries(found, grid_size)
            if not successors[prefix] and negative_caching:
                logger.debug(f"DHT negative caching: storing a 'no prefix' entry for {prefix}")
                asyncio.create_task(
                    node.store(prefix, subkey=-1, value=None, expiration_time=get_dht_time() + cache_expiration)
                )
        return successors

    def find_best_experts(
        self, grid_scores: Sequence[Sequence[float]], beam_size: int, return_future: bool = False
    ) -> Union[List[RemoteExpert], MPFuture[List[RemoteExpert]]]:
        """
        Find and return :beam_size: active experts with highest scores, use both local cache and DHT

        :param grid_scores: scores predicted for each dimension in the grid
        :type grid_scores: model scores for each grid dimension, list of arrays of shape grid_size[i]
        :param beam_size: how many best experts should beam search return
         After time_budget is reached, beam search won't search for more experts and instead fall back on local cache
         Please note that any queries that fall outside the budget will still be performed in background and cached
         for subsequent iterations as long as DHTNode.cache_locally is True
        :param return_future: if set to True, returns MPFuture that can be awaited to get the actual result
        :returns: a list that contains *up to* k_best RemoteExpert instances
        """
        assert len(grid_scores) == len(self.grid_size) and beam_size > 0
        result = self.dht.run_coroutine(
            partial(
                self._find_best_experts,
                prefix=self.uid_prefix,
                beam_size=beam_size,
                grid_scores=list(grid_scores),
                negative_caching=self.negative_caching,
                cache_expiration=self.cache_expiration,
                num_workers=self.num_workers,
            ),
            return_future,
        )
        return create_remote_experts(result, self.dht, return_future)

    @classmethod
    async def _find_best_experts(
        cls,
        dht: DHT,
        node: DHTNode,
        prefix: str,
        grid_scores: List[Tuple[float]],
        beam_size: int,
        negative_caching: bool,
        cache_expiration: DHTExpiration,
        num_workers: Optional[int] = None,
    ) -> List[ExpertInfo]:
        num_workers = num_workers or min(beam_size, dht.num_workers or beam_size)

        # form initial beam from top-k active L1 prefixes, each row is (score, uid prefix, possible suffixes)
        beam: List[Tuple[Score, ExpertPrefix, Dict[Coordinate, ExpertInfo]]] = await cls._get_initial_beam(
            dht, node, prefix, beam_size, grid_scores[0], negative_caching, min(beam_size, num_workers)
        )

        best_experts_heap: List[Tuple[Score, ExpertInfo]] = []  # max-heap of expert infos ordered by scores
        unique_experts: Set[ExpertUID] = set()

        for dim_index in range(1, len(grid_scores) - 1):
            for score, expert_info in cls._iterate_matching_experts(beam, grid_scores):
                if expert_info.uid not in unique_experts:
                    push_and_maybe_pop = heapq.heappush if len(best_experts_heap) < beam_size else heapq.heappushpop
                    push_and_maybe_pop(best_experts_heap, (score, expert_info))
                    unique_experts.add(expert_info.uid)

            # form new beam using successors from the current beam
            dim_scores = grid_scores[dim_index]
            best_active_pairs: List[Tuple[Score, ExpertPrefix]] = heapq.nlargest(
                beam_size,
                (
                    (prefix_score + dim_scores[next_coord], f"{prefix}{next_coord}{UID_DELIMITER}")
                    for prefix_score, prefix, suffixes in beam
                    for next_coord in suffixes.keys()
                    if isinstance(next_coord, int) and 0 <= next_coord < len(dim_scores)
                ),
            )

            _, best_uid_prefixes = zip(*best_active_pairs)

            # search DHT for next step suffixes
            successors = await cls._get_active_successors(
                dht,
                node,
                best_uid_prefixes,
                grid_size=None,
                negative_caching=negative_caching,
                cache_expiration=cache_expiration,
                num_workers=num_workers,
            )
            beam = [(score, prefix, successors[prefix]) for score, prefix in best_active_pairs if successors[prefix]]
            if not beam:
                logger.warning(f"Beam search had to terminate prematurely because of empty beam (dim 0)")
                break

        # add best experts from the final beam
        for score, expert_info in cls._iterate_matching_experts(beam, grid_scores):
            if expert_info.uid not in unique_experts:
                push_and_maybe_pop = heapq.heappush if len(best_experts_heap) < beam_size else heapq.heappushpop
                push_and_maybe_pop(best_experts_heap, (score, expert_info))
                unique_experts.add(expert_info.uid)

        return [expert_info for _, expert_info in sorted(best_experts_heap, reverse=True)]

    @staticmethod
    def _iterate_matching_experts(
        beam: List[Tuple[Score, ExpertPrefix, Dict[Coordinate, ExpertInfo]]], grid_scores: Sequence[Sequence[float]]
    ) -> Iterator[Tuple[Score, ExpertInfo]]:
        """iterate over all exemplar experts attached to current beam"""
        for score, prefix, suffixes in beam:
            for next_coord, match in suffixes.items():
                if len(grid_scores) == 1 and next_coord == FLAT_EXPERT:
                    yield score, match
                elif isinstance(match.uid, ExpertUID) and match.uid.count(UID_DELIMITER) == len(grid_scores):
                    expert_coords = match.uid.split(UID_DELIMITER)[1:]
                    if all(
                        coord.isdigit() and 0 <= int(coord) < len(grid_scores[i])
                        for i, coord in enumerate(expert_coords)
                    ):
                        expert_score = sum(
                            scores[coord] for scores, coord in zip(grid_scores, map(int, expert_coords))
                        )
                        yield expert_score, match
                    else:
                        logger.warning(f"Found incompatible expert coordinates: {expert_coords}")
                else:
                    logger.warning(f"Found incompatible expert UID: {match.uid}")

    def batch_find_best_experts(
        self, batch_grid_scores: Sequence[Sequence[Sequence[float]]], beam_size: int, return_future: bool = False
    ) -> Union[List[List[RemoteExpert]], MPFuture[List[List[RemoteExpert]]]]:
        """
        Find and return :beam_size: active experts with highest scores, use both local cache and DHT

        :param batch_grid_scores: scores predicted for each batch example and each dimension in the grid,
        :type batch_grid_scores: list of arrays of shape (batch_size, grid_size[i])
        :param beam_size: how many best experts should beam search return
         After time_budget is reached, beam search won't search for more experts and instead fall back on local cache
         Please note that any queries that fall outside the budget will still be performed in background and cached
         for subsequent iterations as long as DHTNode.cache_locally is True
        :param return_future: if set to True, returns MPFuture that can be awaited to get the actual result
        :returns: a list that contains *up to* k_best RemoteExpert instances
        """
        result = self.dht.run_coroutine(
            partial(
                self._batch_find_best_experts,
                prefix=self.uid_prefix,
                batch_grid_scores=batch_grid_scores,
                beam_size=beam_size,
                negative_caching=self.negative_caching,
                num_workers=self.num_workers,
            ),
            return_future,
        )

        return batch_create_remote_experts(result, self.dht, return_future)

    @classmethod
    async def _batch_find_best_experts(
        cls,
        dht: DHT,
        node: DHTNode,
        prefix: str,
        batch_grid_scores: Sequence[Sequence[Tuple[float]]],
        beam_size: int,
        negative_caching: bool,
        num_workers: Optional[int],
    ) -> Sequence[Sequence[ExpertInfo]]:
        batch_grid_scores = [
            [tuple(grid_score[i]) for grid_score in batch_grid_scores] for i in range(len(batch_grid_scores[0]))
        ]
        coros = [
            cls._find_best_experts(dht, node, prefix, grid_scores, beam_size, negative_caching, num_workers)
            for grid_scores in batch_grid_scores
        ]
        return await asyncio.gather(*coros)
