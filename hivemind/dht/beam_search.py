import heapq
from typing import Dict, Optional

from .protocol import KademliaProtocol
from .routing import DHTID
from ..utils import Endpoint


async def beam_search(protocol: KademliaProtocol, query_id: DHTID, initial_peers: Dict[DHTID, Endpoint], k_nearest: int,
                      beam_size: Optional[int] = None, max_hops: Optional[int] = None):
    """ you see a lengthy description of how beam search works """
    beam_size = k_nearest if beam_size is None else beam_size
    max_hops = float('inf') if max_hops is None else max_hops

    ids_to_endpoint = dict(initial_peers)  # all ids visited by this beam search
    ids_to_distance = dict(zip(initial_peers, query_id.xor_distance(initial_peers)))

    candidates = [(distance, peer) for peer, distance in ids_to_distance.items()]  # unvisited nodes, nearest-first heap
    heapq.heapify(candidates)
    top_results = [(-distance, peer) for distance, peer in heapq.nsmallest(beam_size, candidates)]  # fathest-first heap
    heapq.heapify(top_results)

    lower_bound = -heapq.nsmallest(1, top_results)[0][0]
    num_hops = 0

    while candidates:
        dist, node_id = heapq.heappop(candidates)
        if dist > lower_bound:
            break

        neighbor_ids_to_endpoint = await protocol.call_find_node(ids_to_endpoint[node_id], query_id)

        # only consider neighbors that have not been visited before
        neighbor_ids_to_endpoint = {node_id: endpoint for node_id, endpoint in neighbor_ids_to_endpoint.items()
                                    if node_id not in ids_to_endpoint}

        neighbor_ids = list(neighbor_ids_to_endpoint.keys())
        if not neighbor_ids:
            continue

        neighbor_ids_to_distance = dict(zip(neighbor_ids, query_id.xor_distance(neighbor_ids)))

        for neighbor_id, distance in neighbor_ids_to_distance.items():
            if distance < lower_bound or len(top_results) < beam_size:
                heapq.heappush(candidates, (distance, neighbor_id))
                heapq.heappush(top_results, (-distance, neighbor_id))

                if len(top_results) > beam_size:
                    heapq.heappop(top_results)

                lower_bound = -heapq.nsmallest(1, top_results)[0][0]

        ids_to_distance.update(neighbor_ids_to_distance)
        ids_to_endpoint.update(neighbor_ids_to_endpoint)

        num_hops += 1
        if num_hops >= max_hops:
            break

    return [node_id for _, node_id in heapq.nlargest(k_nearest, top_results)]


