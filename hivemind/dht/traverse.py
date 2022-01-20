""" Utility functions for crawling DHT nodes, used to get and store keys in a DHT """
import asyncio
import heapq
from collections import Counter
from typing import Any, Awaitable, Callable, Collection, Dict, List, Optional, Set, Tuple

from hivemind.dht.routing import DHTID

ROOT = 0  # alias for heap root


async def simple_traverse_dht(
    query_id: DHTID,
    initial_nodes: Collection[DHTID],
    beam_size: int,
    get_neighbors: Callable[[DHTID], Awaitable[Tuple[Collection[DHTID], bool]]],
    visited_nodes: Collection[DHTID] = (),
) -> Tuple[Tuple[DHTID], Set[DHTID]]:
    """
    Traverse the DHT graph using get_neighbors function, find :beam_size: nearest nodes according to DHTID.xor_distance.

    :note: This is a simplified (but working) algorithm provided for documentation purposes. Actual DHTNode uses
       `traverse_dht` - a generalization of this this algorithm that allows multiple queries and concurrent workers.

    :param query_id: search query, find k_nearest neighbors of this DHTID
    :param initial_nodes: nodes used to pre-populate beam search heap, e.g. [my_own_DHTID, ...maybe_some_peers]
    :param beam_size: beam search will not give up until it exhausts this many nearest nodes (to query_id) from the heap
        Recommended value: A beam size of k_nearest * (2-5) will yield near-perfect results.
    :param get_neighbors: A function that returns neighbors of a given node and controls beam search stopping criteria.
        async def get_neighbors(node: DHTID) -> neighbors_of_that_node: List[DHTID], should_continue: bool
        If should_continue is False, beam search will halt and return k_nearest of whatever it found by then.
    :param visited_nodes: beam search will neither call get_neighbors on these nodes, nor return them as nearest
    :returns: a list of k nearest nodes (nearest to farthest), and a set of all visited nodes (including visited_nodes)
    """
    visited_nodes = set(visited_nodes)  # note: copy visited_nodes because we will add more nodes to this collection.
    initial_nodes = [node_id for node_id in initial_nodes if node_id not in visited_nodes]
    if not initial_nodes:
        return (), visited_nodes

    unvisited_nodes = [(distance, uid) for uid, distance in zip(initial_nodes, query_id.xor_distance(initial_nodes))]
    heapq.heapify(unvisited_nodes)  # nearest-first heap of candidates, unlimited size

    nearest_nodes = [(-distance, node_id) for distance, node_id in heapq.nsmallest(beam_size, unvisited_nodes)]
    heapq.heapify(
        nearest_nodes
    )  # farthest-first heap of size beam_size, used for early-stopping and to select results
    while len(nearest_nodes) > beam_size:
        heapq.heappop(nearest_nodes)

    visited_nodes |= set(initial_nodes)
    upper_bound = -nearest_nodes[0][0]  # distance to farthest element that is still in beam
    was_interrupted = False  # will set to True if host triggered beam search to stop via get_neighbors

    while (not was_interrupted) and len(unvisited_nodes) != 0 and unvisited_nodes[0][0] <= upper_bound:
        _, node_id = heapq.heappop(unvisited_nodes)  # note: this  --^ is the smallest element in heap (see heapq)
        neighbors, was_interrupted = await get_neighbors(node_id)
        neighbors = [node_id for node_id in neighbors if node_id not in visited_nodes]
        visited_nodes.update(neighbors)

        for neighbor_id, distance in zip(neighbors, query_id.xor_distance(neighbors)):
            if distance <= upper_bound or len(nearest_nodes) < beam_size:
                heapq.heappush(unvisited_nodes, (distance, neighbor_id))

                heapq_add_or_replace = heapq.heappush if len(nearest_nodes) < beam_size else heapq.heappushpop
                heapq_add_or_replace(nearest_nodes, (-distance, neighbor_id))
                upper_bound = -nearest_nodes[0][0]  # distance to beam_size-th nearest element found so far

    return tuple(node_id for _, node_id in heapq.nlargest(beam_size, nearest_nodes)), visited_nodes


async def traverse_dht(
    queries: Collection[DHTID],
    initial_nodes: List[DHTID],
    beam_size: int,
    num_workers: int,
    queries_per_call: int,
    get_neighbors: Callable[[DHTID, Collection[DHTID]], Awaitable[Dict[DHTID, Tuple[Tuple[DHTID], bool]]]],
    found_callback: Optional[Callable[[DHTID, List[DHTID], Set[DHTID]], Awaitable[Any]]] = None,
    await_all_tasks: bool = True,
    visited_nodes: Optional[Dict[DHTID, Set[DHTID]]] = (),
) -> Tuple[Dict[DHTID, List[DHTID]], Dict[DHTID, Set[DHTID]]]:
    """
    Search the DHT for nearest neighbors to :queries: (based on DHTID.xor_distance). Use get_neighbors to request peers.
    The algorithm can reuse intermediate results from each query to speed up search for other (similar) queries.

    :param queries: a list of search queries, find beam_size neighbors for these DHTIDs
    :param initial_nodes: nodes used to pre-populate beam search heap, e.g. [my_own_DHTID, ...maybe_some_peers]
    :param beam_size: beam search will not give up until it visits this many nearest nodes (to query_id) from the heap
    :param num_workers: run up to this many concurrent get_neighbors requests, each querying one peer for neighbors.
        When selecting a peer to request neighbors from, workers try to balance concurrent exploration across queries.
        A worker will expand the nearest candidate to a query with least concurrent requests from other workers.
        If several queries have the same number of concurrent requests, prefer the one with nearest XOR distance.

    :param queries_per_call: workers can pack up to this many queries in one get_neighbors call. These queries contain
        the primary query (see num_workers above) and up to `queries_per_call - 1` nearest unfinished queries.

    :param get_neighbors: A function that requests a given peer to find nearest neighbors for multiple queries
        async def get_neighbors(peer, queries) -> {query1: ([nearest1, nearest2, ...], False), query2: ([...], True)}
        For each query in queries, return nearest neighbors (known to a given peer) and a boolean "should_stop" flag
        If should_stop is True, traverse_dht will no longer search for this query or request it from other peers.
        The search terminates iff each query is either stopped via should_stop or finds beam_size nearest nodes.

    :param found_callback: if specified, call this callback for each finished query the moment it finishes or is stopped
        More specifically, run asyncio.create_task(found_callback(query, nearest_to_query, visited_for_query))
        Using this callback allows one to process results faster before traverse_dht is finishes for all queries.
        It is guaranteed that found_callback will be called exactly once on each query in queries.

    :param await_all_tasks: if True, wait for all tasks to finish before returning, otherwise returns after finding
        nearest neighbors and finishes the remaining tasks (callbacks and queries to known-but-unvisited nodes)

    :param visited_nodes: for each query, do not call get_neighbors on these nodes, nor return them among nearest.
    :note: the source code of this function can get tricky to read. Take a look at `simple_traverse_dht` function
        for reference. That function implements a special case of traverse_dht with a single query and one worker.

    :returns: a dict of nearest nodes, and another dict of visited nodes
        nearest nodes: { query -> a list of up to beam_size nearest nodes, ordered nearest-first }
        visited nodes: { query -> a set of all nodes that received requests for a given query }
    """
    if len(queries) == 0:
        return {}, dict(visited_nodes or {})

    unfinished_queries = set(queries)  # all queries that haven't triggered finish_search yet
    candidate_nodes: Dict[DHTID, List[Tuple[int, DHTID]]] = {}  # heap: unvisited nodes, ordered nearest-to-farthest
    nearest_nodes: Dict[DHTID, List[Tuple[int, DHTID]]] = {}  # heap: top-k nearest nodes, farthest-to-nearest
    known_nodes: Dict[DHTID, Set[DHTID]] = {}  # all nodes ever added to the heap (for deduplication)
    visited_nodes: Dict[DHTID, Set[DHTID]] = dict(visited_nodes or {})  # nodes that were chosen for get_neighbors call
    pending_tasks = set()  # all active tasks (get_neighbors and found_callback)
    active_workers = Counter({q: 0 for q in queries})  # count workers that search for this query

    search_finished_event = asyncio.Event()  # used to immediately stop all workers when the search is finished
    heap_updated_event = asyncio.Event()  # if a worker has no nodes to explore, it will await other workers
    heap_updated_event.set()

    # initialize data structures
    for query in queries:
        distances = query.xor_distance(initial_nodes)
        candidate_nodes[query] = list(zip(distances, initial_nodes))
        nearest_nodes[query] = list(zip([-d for d in distances], initial_nodes))
        heapq.heapify(candidate_nodes[query])
        heapq.heapify(nearest_nodes[query])
        while len(nearest_nodes[query]) > beam_size:
            heapq.heappop(nearest_nodes[query])
        known_nodes[query] = set(initial_nodes)
        visited_nodes[query] = set(visited_nodes.get(query, ()))

    def heuristic_priority(heap_query: DHTID):
        """Workers prioritize expanding nodes that lead to under-explored queries (by other workers)"""
        if has_candidates(heap_query):
            # prefer candidates in heaps with least number of concurrent workers, break ties by distance to query
            return active_workers[heap_query], candidate_nodes[heap_query][ROOT][0]
        return float("inf"), float("inf")  # try not to explore vertices with no candidates

    def has_candidates(query: DHTID):
        """Whether this query's heap contains at least one candidate node that can be explored"""
        return candidate_nodes[query] and candidate_nodes[query][ROOT][0] <= upper_bound(query)

    def upper_bound(query: DHTID):
        """Any node that is farther from query than upper_bound(query) will not be added to heaps"""
        return -nearest_nodes[query][ROOT][0] if len(nearest_nodes[query]) >= beam_size else float("inf")

    def finish_search(query):
        """Remove query from a list of targets"""
        unfinished_queries.remove(query)
        if len(unfinished_queries) == 0:
            search_finished_event.set()
        if found_callback:
            nearest_neighbors = [peer for _, peer in heapq.nlargest(beam_size, nearest_nodes[query])]
            pending_tasks.add(asyncio.create_task(found_callback(query, nearest_neighbors, set(visited_nodes[query]))))

    async def worker():
        while unfinished_queries:
            # select the heap based on priority
            chosen_query: DHTID = min(unfinished_queries, key=heuristic_priority)

            # if there are no peers to explore...
            if not has_candidates(chosen_query):
                other_workers_pending = active_workers.most_common(1)[0][1] > 0
                if other_workers_pending:  # ... wait for other workers (if any) or add more peers
                    heap_updated_event.clear()
                    await heap_updated_event.wait()
                    continue
                else:  # ... or if there is no hope of new nodes, finish search immediately
                    for query in list(unfinished_queries):
                        finish_search(query)
                    break

            # select vertex to be explored
            chosen_distance_to_query, chosen_peer = heapq.heappop(candidate_nodes[chosen_query])
            if chosen_peer in visited_nodes[chosen_query] or chosen_distance_to_query > upper_bound(chosen_query):
                if chosen_distance_to_query > upper_bound(chosen_query) and active_workers[chosen_query] == 0:
                    finish_search(chosen_query)
                continue

            # find additional queries to pack in the same request
            possible_additional_queries = [
                query
                for query in unfinished_queries
                if query != chosen_query and chosen_peer not in visited_nodes[query]
            ]
            queries_to_call = [chosen_query] + heapq.nsmallest(
                queries_per_call - 1, possible_additional_queries, key=chosen_peer.xor_distance
            )

            # update priorities for subsequent workers
            active_workers.update(queries_to_call)
            for query_to_call in queries_to_call:
                visited_nodes[query_to_call].add(chosen_peer)

            # get nearest neighbors (over network) and update search heaps. Abort if search finishes early
            get_neighbors_task = asyncio.create_task(get_neighbors(chosen_peer, queries_to_call))
            pending_tasks.add(get_neighbors_task)
            await asyncio.wait([get_neighbors_task, search_finished_event.wait()], return_when=asyncio.FIRST_COMPLETED)
            if search_finished_event.is_set():
                break  # other worker triggered finish_search, we exit immediately
            pending_tasks.remove(get_neighbors_task)

            # add nearest neighbors to their respective heaps
            for query, (neighbors_for_query, should_stop) in get_neighbors_task.result().items():
                if should_stop and (query in unfinished_queries):
                    finish_search(query)
                if query not in unfinished_queries:
                    continue  # either we finished search or someone else did while we awaited
                for neighbor in neighbors_for_query:
                    if neighbor not in known_nodes[query]:
                        known_nodes[query].add(neighbor)
                        distance = query.xor_distance(neighbor)
                        if distance <= upper_bound(query) or len(nearest_nodes[query]) < beam_size:
                            heapq.heappush(candidate_nodes[query], (distance, neighbor))
                            if len(nearest_nodes[query]) < beam_size:
                                heapq.heappush(nearest_nodes[query], (-distance, neighbor))
                            else:
                                heapq.heappushpop(nearest_nodes[query], (-distance, neighbor))

            # we finished processing a request, update priorities for other workers
            active_workers.subtract(queries_to_call)
            heap_updated_event.set()

    workers = [asyncio.create_task(worker()) for _ in range(num_workers)]
    try:
        # spawn all workers and wait for them to terminate; workers terminate after exhausting unfinished_queries
        await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)
        assert len(unfinished_queries) == 0 and search_finished_event.is_set()

        if await_all_tasks:
            await asyncio.gather(*pending_tasks)

        nearest_neighbors_per_query = {
            query: [peer for _, peer in heapq.nlargest(beam_size, nearest_nodes[query])] for query in queries
        }
        return nearest_neighbors_per_query, visited_nodes

    except asyncio.CancelledError as e:
        for worker in workers:
            worker.cancel()
        raise e
