import asyncio
import heapq
from typing import Dict, Awaitable, Callable, Any, Tuple, List, Set, Collection, Optional
from .routing import DHTID

DHTValue = Any
Heap = List
NEG_XOR_DISTANCE = XOR_DISTANCE = int
ROOT = 0


async def traverse_dht(
        queries: List[DHTID], initial_peers: List[DHTID], beam_size: int, num_workers: int,
        get_neighbors: Callable[[DHTID, Collection[DHTID]], Awaitable[Dict[DHTID, Tuple[List[DHTID], bool]]]],
        found_callback: Optional[Callable[[DHTID, List[DHTID], Set[DHTID]], Awaitable[Any]]] = None, await_found=False
) -> Tuple[Dict[DHTID, List[DHTID]], Set[DHTID]]:
    """
    Asynchronous beam search over the DHT. Not meant to be called by the user, please use DHTNode.store/get instead.
    Traverse the DHT graph using get_neighbors function, find up to beam_size nearest nodes based on DHTID.xor_distance.

    :param queries: a list of search queries, find beam_size neighbors for these DHTIDs
    :param initial_peers: nodes used to pre-populate beam search heap, e.g. [my_own_DHTID, ...maybe_some_peers]
    :param beam_size: beam search will not give up until it exhausts this many nearest nodes (to query_id) from the heap
    :param num_workers: run this many concurrent get_neighbors. Each worker expands nearest node to one of the queries.
        Workers can run concurrent requests for the same query or for several queries in parallel depending on priority.
        A worker selects a node for get_neighbors (out of nearest nodes per query) based on two factors:

           - distance reduction = how much closer we are to a query after including this node (sum over all queries)
           - crowding penalty = with everything else equal, spread workers evenly between queries (i.e. between heaps)

    :param get_neighbors: A function that requests a given peer to find nearest neighbors for several queries
        async def get_neighbors(peer, queries) -> {query1: ([nearest1, nearest2, ...], False), query2: ([...], False)}
        For each query in queries, return nearest neighbors (known to a given peer) and a boolean "should_stop" flag
        If should_stop is True, traverse_dht will no longer search for it or request it from peers
    :note: the search terminates iff each query is either stopped via should_stop or finds beam_size nearest nodes
    :param found_callback: if specified, call this callback for each finished query the moment it finishes or is stopped
        More specifically, traverse_dht will run asyncio.create_task(found_found_callback(query, nearest_peers, visited)
        Using callbacks allows one to process early results before traverse_dht is finished for all queries
    :param await_found: if set to True, wait for all callbacks to finish before returning (e.g. if you use asyncio.run)

    :returns: a dict {query -> beam_size nearest nodes nearest-first}, and a set of all nodes queried with get_neighbors
    """
    unfinished_queries = set(queries)                           # all queries that haven't triggered finish_search
    visited_nodes: Set[DHTID] = set()                           # all nodes for which we called get_neighbors
    candidate_nodes: Dict[DHTID, List[Tuple[int, DHTID]]] = {}  # heap: unvisited nodes, ordered nearest-to-farthest
    nearest_nodes: Dict[DHTID, List[Tuple[int, DHTID]]] = {}    # heap: top-k nearest nodes, ordered fartest-to-nearest
    known_nodes: Dict[DHTID, Set[DHTID]] = {}                   # all nodes ever added to the heap (for deduplication)
    pending_callbacks = []                                      # all found_callback tasks created via finish_search

    # variables used exclusively for priority computation
    distance_from_visited: Dict[DHTID, XOR_DISTANCE] = {query: float('inf') for query in queries}
    num_active_workers: Dict[DHTID, int] = {query: 0 for query in queries}

    # initialize data structures
    for query in queries:
        distances = query.xor_distance(initial_peers)
        candidate_nodes[query] = list(zip(distances, initial_peers))
        nearest_nodes[query] = list(zip([-d for d in distances], initial_peers))
        heapq.heapify(candidate_nodes[query])
        heapq.heapify(nearest_nodes[query])
        while len(nearest_nodes[query]) > beam_size:
            heapq.heappop(nearest_nodes[query])
        known_nodes[query] = set(initial_peers)

    def get_query_priority(query: DHTID):
        """ Workers prioritize expanding nodes (out of roots of query heaps) that reduce distances to all queries """
        distance_reduction = 0
        for query in unfinished_queries:
            current_distance = distance_from_visited[query]
            distance_reduction += current_distance - min(current_distance, candidate_nodes[query][ROOT][0])
        return -distance_reduction, num_active_workers[query]  # break ties by minimum concurrent requests

    def upper_bound(query: DHTID):
        """ Any node that is farther from query than upper_bound(query) will not be added to heaps """
        if len(nearest_nodes[query]) >= beam_size:
            return -nearest_nodes[query][ROOT][0]
        else:
            return float('inf')

    def finish_search(query):
        """ Remove query from a list of targets """
        unfinished_queries.remove(query)
        if found_callback:
            nearest_neighbors = [peer for _, peer in heapq.nlargest(beam_size, nearest_nodes[query])]
            pending_callbacks.append(asyncio.create_task(found_callback(query, nearest_neighbors, set(visited_nodes))))

    async def worker():
        while unfinished_queries:
            # select vertex to be explored
            chosen_query: DHTID = min(candidate_nodes, key=get_query_priority)
            chosen_distance_to_query, chosen_peer = heapq.heappop(candidate_nodes[chosen_query])

            if chosen_peer in visited_nodes:
                continue

            if chosen_distance_to_query > upper_bound(chosen_query):
                finish_search(chosen_query)
                continue

            # update heap priorities for other workers
            num_active_workers[chosen_query] += 1
            visited_nodes.add(chosen_peer)
            for query in unfinished_queries:
                distance_from_visited[query] = min(distance_from_visited[query], chosen_distance_to_query)

            # get nearest neighbors (over network) and update search heaps
            response = await get_neighbors(chosen_peer, unfinished_queries)
            for query, (neighbors_for_query, should_stop) in response.items():
                if should_stop:
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

            # we finished processing query, update priorities again
            num_active_workers[chosen_query] -= 1

    # spawn all workers and wait for them to terminate; workers terminate after exhausting unfinished_queries
    await asyncio.wait([worker() for _ in range(num_workers)])
    if await_found:
        await(asyncio.wait(pending_callbacks))

    nearest_neighbors_per_query = {
        query: [peer for _, peer in heapq.nlargest(beam_size, nearest_nodes[query])]
        for query in queries
    }
    return nearest_neighbors_per_query, visited_nodes


