import asyncio
import heapq
from typing import Dict, Awaitable, Callable, Any, Tuple, List, Set, Collection, Optional
from .routing import DHTID

DHTValue = Any
Heap = List
DHTID_Query = DHTID_Peer = DHTID
NEG_XOR_DISTANCE = XOR_DISTANCE = int
ROOT = 0


async def traverse_dht(
        queries: List[DHTID_Query], initial_peers: List[DHTID_Peer], beam_size: int, num_workers: int,
        get_neighbors: Callable[[DHTID_Peer, Collection[DHTID_Query]],
                                Awaitable[Dict[DHTID_Query,
                                               Tuple[List[DHTID_Peer], bool]]]],
        #                                           ^-nearest peers-^  ^--should_stop_this_key
        found_callback: Optional[Callable[[DHTID_Query, List[DHTID_Peer], Set[DHTID_Peer]], Awaitable[Any]]]
        #                                 ^found-query^ ^---nearest---^  ^---visited---^

) -> Tuple[Dict[DHTID_Query, List[DHTID_Peer]], Set[DHTID_Peer]]:
    #      for each query, nearest peers;   also all visited

    unfinished_queries = set(queries)
    visited_nodes: Set[DHTID_Peer] = set()
    candidate_nodes: Dict[DHTID_Query, Heap[Tuple[XOR_DISTANCE, DHTID_Peer]]] = {}  # {q_i -> H_i}
    nearest_nodes: Dict[DHTID_Query, Heap[Tuple[NEG_XOR_DISTANCE, DHTID_Peer]]] = {}  # {q_i -> T_i}
    known_nodes: Dict[DHTID_Query, Set[DHTID_Peer]] = {}

    # variables used exclusively for priority computation
    distance_from_visited: Dict[DHTID_Query, XOR_DISTANCE] = {query: float('inf') for query in queries}
    num_active_workers: Dict[DHTID_Query, int] = {query: 0 for query in queries}

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

    def get_query_priority(query: DHTID_Query):
        """ Priority used by workers to select which heap to explore first """
        distance_reduction = 0
        for query in unfinished_queries:
            current_distance = distance_from_visited[query]
            distance_reduction += current_distance - min(current_distance, candidate_nodes[query][ROOT][0])
        return -distance_reduction, num_active_workers[query]

    def upper_bound(query: DHTID_Query):
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
            asyncio.create_task(found_callback(query, nearest_neighbors, set(visited_nodes)))

    async def worker():
        while unfinished_queries:
            # select vertex to be explored
            chosen_query: DHTID_Query = min(candidate_nodes, key=get_query_priority)
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

    # spawn all workers and wait for them to terminate; workers terminate after exhausting unifinished_queries
    await asyncio.gather([worker() for _ in range(num_workers)])

    nearest_neighbors_per_query = {
        query: [peer for _, peer in heapq.nlargest(beam_size, nearest_nodes[query])]
        for query in queries
    }
    return nearest_neighbors_per_query, visited_nodes


