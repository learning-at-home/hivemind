import heapq
from typing import Collection, Callable, Tuple, List, Awaitable, Set
from warnings import warn

from .routing import DHTID


async def traverse_dht(query_id: DHTID, initial_nodes: Collection[DHTID], k_nearest: int, beam_size: int,
                       get_neighbors: Callable[[DHTID], Awaitable[Tuple[Collection[DHTID], bool]]],
                       visited_nodes: Collection[DHTID] = ()) -> Tuple[List[DHTID], Set[DHTID]]:
    """
    Asynchronous beam search over the DHT. Not meant to be called by the user, please use DHTNode.store/get instead.
    Traverse the DHT graph using get_neighbors function, find up to k_nearest nodes according to DHTID.xor_distance.
    Approximate time complexity: O(T * log T) where T = (path_to_true_nearest + beam_size) * mean_num_neighbors

    :param query_id: search query, find k_nearest neighbors of this DHTID
    :param initial_nodes: nodes used to pre-populate beam search heap, e.g. [my_own_DHTID, *maybe_some_peers]
    :param k_nearest: find up to this many nearest neighbors. If there are less nodes in the DHT, return all nodes
    :param beam_size: beam search will not give up until it exhausts this many nearest nodes (to query_id) from the heap
        Recommended value: A beam size of k_nearest * (2-5) will yield near-perfect results.

    :param get_neighbors: A function that returns neighbors of a given node and controls beam search stopping criteria.
        async def get_neighbors(node: DHTID) -> neighbors_of_that_node: List[DHTID], should_continue: bool
        If should_continue is False, beam search will halt and return k_nearest of whatever it found by then.

    :param visited_nodes: beam search will neither call get_neighbors on these nodes, nor return them as nearest
    :returns: a list of k nearest nodes (nearest to farthest), and a set of all visited nodes (including visited_nodes)
    """
    if beam_size < k_nearest:
        warn(f"beam search: beam_size({beam_size}) is too small, beam search may fail to find k neighbors.")
    visited_nodes = set(visited_nodes)  # note: copy visited_nodes because we will add more nodes to this collection.
    initial_nodes = [node_id for node_id in initial_nodes if node_id not in visited_nodes]
    if not initial_nodes:
        return [], visited_nodes

    unvisited_nodes = [(distance, uid) for uid, distance in zip(initial_nodes, query_id.xor_distance(initial_nodes))]
    heapq.heapify(unvisited_nodes)  # nearest-first heap of candidates, unlimited size

    nearest_nodes = [(-distance, node_id) for distance, node_id in heapq.nsmallest(beam_size, unvisited_nodes)]
    heapq.heapify(nearest_nodes)  # farthest-first heap of size beam_size, used for early-stopping and to select results
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

    return [node_id for _, node_id in heapq.nlargest(k_nearest, nearest_nodes)], visited_nodes
