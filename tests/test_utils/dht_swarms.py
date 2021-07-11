import asyncio
import multiprocessing as mp
import random
import signal
import threading
from typing import Dict, List, Tuple

from multiaddr import Multiaddr

from hivemind.dht.node import DHTID, DHTNode
from hivemind.p2p import PeerID


def run_node(initial_peers: List[Multiaddr], info_queue: mp.Queue):
    if asyncio.get_event_loop().is_running():
        asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop
        asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()

    node = loop.run_until_complete(DHTNode.create(initial_peers=initial_peers, ping_n_attempts=10))
    maddrs = loop.run_until_complete(node.get_visible_maddrs())

    info_queue.put((node.node_id, node.peer_id, maddrs))

    async def shutdown():
        await node.shutdown()
        loop.stop()

    loop.add_signal_handler(signal.SIGTERM, lambda: loop.create_task(shutdown()))
    loop.run_forever()


def launch_swarm_in_separate_processes(n_peers: int, n_sequential_peers: int) -> \
        Tuple[List[mp.Process], Dict[PeerID, DHTID], List[List[Multiaddr]]]:
    assert n_sequential_peers < n_peers, \
        'Parameters imply that first n_sequential_peers of n_peers will be run sequentially'

    processes = []
    dht = {}
    swarm_maddrs = []

    info_queue = mp.Queue()
    info_lock = mp.RLock()

    for _ in range(n_sequential_peers):
        initial_peers = random.choice(swarm_maddrs) if swarm_maddrs else []

        proc = mp.Process(target=run_node, args=(initial_peers, info_queue), daemon=True)
        proc.start()
        processes.append(proc)

        node_id, peer_endpoint, peer_maddrs = info_queue.get()
        dht[peer_endpoint] = node_id
        swarm_maddrs.append(peer_maddrs)

    def collect_info():
        while True:
            node_id, peer_endpoint, peer_maddrs = info_queue.get()
            with info_lock:
                dht[peer_endpoint] = node_id
                swarm_maddrs.append(peer_maddrs)

                if len(dht) == n_peers:
                    break

    collect_thread = threading.Thread(target=collect_info)
    collect_thread.start()

    for _ in range(n_peers - n_sequential_peers):
        with info_lock:
            initial_peers = random.choice(swarm_maddrs)

        proc = mp.Process(target=run_node, args=(initial_peers, info_queue), daemon=True)
        proc.start()
        processes.append(proc)

    collect_thread.join()

    return processes, dht, swarm_maddrs


async def launch_star_shaped_swarm(n_peers: int, **kwargs) -> List[DHTNode]:
    nodes = [await DHTNode.create(**kwargs)]
    initial_peers = await nodes[0].get_visible_maddrs()
    nodes += await asyncio.gather(*[DHTNode.create(initial_peers=initial_peers, **kwargs)
                                    for _ in range(n_peers - 1)])
    return nodes
