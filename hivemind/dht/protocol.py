""" RPC protocol that provides nodes a way to communicate with each other. Based on gRPC.AIO. """
from __future__ import annotations

import asyncio
from typing import Optional, List, Tuple, Dict, Any, Sequence, Union, Collection

import grpc

from hivemind.dht.routing import RoutingTable, DHTID, BinaryDHTValue, DHTExpiration, Subkey
from hivemind.dht.storage import DHTLocalStorage, DictionaryDHTValue
from hivemind.proto import dht_pb2, dht_pb2_grpc as dht_grpc
from hivemind.utils import Endpoint, get_logger, replace_port, MSGPackSerializer, ChannelCache, ValueWithExpiration

logger = get_logger(__name__)


class DHTProtocol(dht_grpc.DHTServicer):
    # fmt:off
    node_id: DHTID; port: int; bucket_size: int; num_replicas: int; wait_timeout: float; node_info: dht_pb2.NodeInfo
    channel_options: Sequence[Tuple[str, Any]]; server: grpc.aio.Server
    storage: DHTLocalStorage; cache: DHTLocalStorage; routing_table: RoutingTable; rpc_semaphore: asyncio.Semaphore
    # fmt:on

    serializer = MSGPackSerializer  # used to pack/unpack DHT Values for transfer over network
    RESERVED_SUBKEYS = IS_REGULAR_VALUE, IS_DICTIONARY = serializer.dumps(None), b''

    @classmethod
    async def create(
            cls, node_id: DHTID, bucket_size: int, depth_modulo: int, num_replicas: int, wait_timeout: float,
            parallel_rpc: Optional[int] = None, cache_size: Optional[int] = None, listen=True, listen_on='0.0.0.0:*',
            channel_options: Sequence[Tuple[str, Any]] = (), **kwargs) -> DHTProtocol:
        """
        A protocol that allows DHT nodes to request keys/neighbors from other DHT nodes.
        As a side-effect, DHTProtocol also maintains a routing table as described in
        https://pdos.csail.mit.edu/~petar/papers/maymounkov-kademlia-lncs.pdf

        See DHTNode (node.py) for a more detailed description.

        :note: the rpc_* methods defined in this class will be automatically exposed to other DHT nodes,
         for instance, def rpc_ping can be called as protocol.call_ping(endpoint, dht_id) from a remote machine
         Only the call_* methods are meant to be called publicly, e.g. from DHTNode
         Read more: https://github.com/bmuller/rpcudp/tree/master/rpcudp
        """
        self = cls(_initialized_with_create=True)
        self.node_id, self.bucket_size, self.num_replicas = node_id, bucket_size, num_replicas
        self.wait_timeout, self.channel_options = wait_timeout, channel_options
        self.storage, self.cache = DHTLocalStorage(), DHTLocalStorage(maxsize=cache_size)
        self.routing_table = RoutingTable(node_id, bucket_size, depth_modulo)
        self.rpc_semaphore = asyncio.Semaphore(parallel_rpc if parallel_rpc is not None else float('inf'))

        if listen:  # set up server to process incoming rpc requests
            grpc.aio.init_grpc_aio()
            self.server = grpc.aio.server(**kwargs)
            dht_grpc.add_DHTServicer_to_server(self, self.server)

            found_port = self.server.add_insecure_port(listen_on)
            assert found_port != 0, f"Failed to listen to {listen_on}"
            self.node_info = dht_pb2.NodeInfo(node_id=node_id.to_bytes(), rpc_port=found_port)
            self.port = found_port
            await self.server.start()
        else:  # not listening to incoming requests, client-only mode
            # note: use empty node_info so peers won't add you to their routing tables
            self.node_info, self.server, self.port = dht_pb2.NodeInfo(), None, None
            if listen_on != '0.0.0.0:*' or len(kwargs) != 0:
                logger.warning(f"DHTProtocol has no server (due to listen=False), listen_on"
                               f"and kwargs have no effect (unused kwargs: {kwargs})")
        return self

    def __init__(self, *, _initialized_with_create=False):
        """ Internal init method. Please use DHTProtocol.create coroutine to spawn new protocol instances """
        assert _initialized_with_create, " Please use DHTProtocol.create coroutine to spawn new protocol instances "
        super().__init__()

    async def shutdown(self, timeout=None):
        """ Process existing requests, close all connections and stop the server """
        if self.server:
            await self.server.stop(timeout)
        else:
            logger.warning("DHTProtocol has no server (due to listen=False), it doesn't need to be shut down")

    def _get_dht_stub(self, peer: Endpoint) -> dht_grpc.DHTStub:
        """ get a DHTStub that sends requests to a given peer """
        return ChannelCache.get_stub(peer, dht_grpc.DHTStub, aio=True, options=self.channel_options)

    async def call_ping(self, peer: Endpoint) -> Optional[DHTID]:
        """
        Get peer's node id and add him to the routing table. If peer doesn't respond, return None
        :param peer: string network address, e.g. 123.123.123.123:1337 or [2a21:6с8:b192:2105]:8888
        :note: if DHTProtocol was created with listen=True, also request peer to add you to his routing table

        :return: node's DHTID, if peer responded and decided to send his node_id
        """
        try:
            async with self.rpc_semaphore:
                peer_info = await self._get_dht_stub(peer).rpc_ping(self.node_info, timeout=self.wait_timeout)
        except grpc.aio.AioRpcError as error:
            logger.warning(f"DHTProtocol failed to ping {peer}: {error.code()}")
            peer_info = None
        responded = bool(peer_info and peer_info.node_id)
        peer_id = DHTID.from_bytes(peer_info.node_id) if responded else None
        asyncio.create_task(self.update_routing_table(peer_id, peer, responded=responded))
        return peer_id

    async def rpc_ping(self, peer_info: dht_pb2.NodeInfo, context: grpc.ServicerContext):
        """ Some node wants us to add it to our routing table. """
        if peer_info.node_id and peer_info.rpc_port:
            sender_id = DHTID.from_bytes(peer_info.node_id)
            rpc_endpoint = replace_port(context.peer(), new_port=peer_info.rpc_port)
            asyncio.create_task(self.update_routing_table(sender_id, rpc_endpoint))
        return self.node_info

    async def call_store(self, peer: Endpoint, keys: Sequence[DHTID],
                         values: Sequence[Union[BinaryDHTValue, DictionaryDHTValue]],
                         expiration_time: Union[DHTExpiration, Sequence[DHTExpiration]],
                         subkeys: Optional[Union[Subkey, Sequence[Optional[Subkey]]]] = None,
                         in_cache: Optional[Union[bool, Sequence[bool]]] = None) -> Optional[List[bool]]:
        """
        Ask a recipient to store several (key, value : expiration_time) items or update their older value

        :param peer: request this peer to store the data
        :param keys: a list of N keys digested by DHTID.generate(source=some_dict_key)
        :param values: a list of N serialized values (bytes) for each respective key
        :param expiration_time: a list of N expiration timestamps for each respective key-value pair(see get_dht_time())
        :param subkeys: a list of N optional sub-keys. If None, stores value normally. If not subkey is not None:
          1) if local storage doesn't have :key:, create a new dictionary {subkey: (value, expiration_time)}
          2) if local storage already has a dictionary under :key:, try add (subkey, value, exp_time) to that dictionary
          2) if local storage associates :key: with a normal value with smaller expiration, clear :key: and perform (1)
          3) finally, if local storage currently associates :key: with a normal value with larger expiration, do nothing
        :param in_cache: a list of booleans, True = store i-th key in cache, value = store i-th key locally
        :note: the difference between storing normally and in cache is that normal storage is guaranteed to be stored
         until expiration time (best-effort), whereas cached storage can be evicted early due to limited cache size
        :return: list of [True / False] True = stored, False = failed (found newer value or no response)
                 if peer did not respond (e.g. due to timeout or congestion), returns None
        """
        if isinstance(expiration_time, DHTExpiration):
            expiration_time = [expiration_time] * len(keys)
        if subkeys is None:
            subkeys = [None] * len(keys)

        in_cache = in_cache if in_cache is not None else [False] * len(keys)  # default value (None)
        in_cache = [in_cache] * len(keys) if isinstance(in_cache, bool) else in_cache  # single bool
        keys, subkeys, values, expiration_time, in_cache = map(list, [keys, subkeys, values, expiration_time, in_cache])
        for i in range(len(keys)):
            if subkeys[i] is None:  # add default sub-key if not specified
                subkeys[i] = self.IS_DICTIONARY if isinstance(values[i], DictionaryDHTValue) else self.IS_REGULAR_VALUE
            else:
                subkeys[i] = self.serializer.dumps(subkeys[i])
            if isinstance(values[i], DictionaryDHTValue):
                assert subkeys[i] == self.IS_DICTIONARY, "Please don't specify subkey when storing an entire dictionary"
                values[i] = self.serializer.dumps(values[i])

        assert len(keys) == len(values) == len(expiration_time) == len(in_cache), "Data is not aligned"
        store_request = dht_pb2.StoreRequest(keys=list(map(DHTID.to_bytes, keys)), subkeys=subkeys, values=values,
                                             expiration_time=expiration_time, in_cache=in_cache, peer=self.node_info)
        try:
            async with self.rpc_semaphore:
                response = await self._get_dht_stub(peer).rpc_store(store_request, timeout=self.wait_timeout)
            if response.peer and response.peer.node_id:
                peer_id = DHTID.from_bytes(response.peer.node_id)
                asyncio.create_task(self.update_routing_table(peer_id, peer, responded=True))
            return response.store_ok
        except grpc.aio.AioRpcError as error:
            logger.warning(f"DHTProtocol failed to store at {peer}: {error.code()}")
            asyncio.create_task(self.update_routing_table(self.routing_table.get(endpoint=peer), peer, responded=False))
            return None

    async def rpc_store(self, request: dht_pb2.StoreRequest, context: grpc.ServicerContext) -> dht_pb2.StoreResponse:
        """ Some node wants us to store this (key, value) pair """
        if request.peer:  # if requested, add peer to the routing table
            asyncio.create_task(self.rpc_ping(request.peer, context))
        assert len(request.keys) == len(request.values) == len(request.expiration_time) == len(request.in_cache)
        response = dht_pb2.StoreResponse(store_ok=[], peer=self.node_info)
        keys = map(DHTID.from_bytes, request.keys)
        for key_id, tag, value_bytes, expiration_time, in_cache in zip(
                keys, request.subkeys, request.values, request.expiration_time, request.in_cache):
            storage = self.cache if in_cache else self.storage
            if tag == self.IS_REGULAR_VALUE:  # store normal value without subkeys
                response.store_ok.append(storage.store(key_id, value_bytes, expiration_time))
            elif tag == self.IS_DICTIONARY:  # store an entire dictionary with several subkeys
                value_dictionary = self.serializer.loads(value_bytes)
                assert isinstance(value_dictionary, DictionaryDHTValue)
                response.store_ok.append(all(storage.store_subkey(key_id, subkey, item.value, item.expiration_time)
                                             for subkey, item in value_dictionary.items()))
            else:  # add a new entry into an existing dictionary value or create a new dictionary with one sub-key
                subkey = self.serializer.loads(tag)
                response.store_ok.append(storage.store_subkey(key_id, subkey, value_bytes, expiration_time))
        return response

    async def call_find(self, peer: Endpoint, keys: Collection[DHTID]) -> Optional[Dict[
        DHTID, Tuple[Optional[ValueWithExpiration[Union[BinaryDHTValue, DictionaryDHTValue]]], Dict[DHTID, Endpoint]]]]:
        """
        Request keys from a peer. For each key, look for its (value, expiration time) locally and
         k additional peers that are most likely to have this key (ranked by XOR distance)

        :returns: A dict key => Tuple[optional value, optional expiration time, nearest neighbors]
         value: value stored by the recipient with that key, or None if peer doesn't have this value
         expiration time: expiration time of the returned value, None if no value was found
         neighbors: a dictionary[node_id : endpoint] containing nearest neighbors from peer's routing table
         If peer didn't respond, returns None
        """
        keys = list(keys)
        find_request = dht_pb2.FindRequest(keys=list(map(DHTID.to_bytes, keys)), peer=self.node_info)
        try:
            async with self.rpc_semaphore:
                response = await self._get_dht_stub(peer).rpc_find(find_request, timeout=self.wait_timeout)
            if response.peer and response.peer.node_id:
                peer_id = DHTID.from_bytes(response.peer.node_id)
                asyncio.create_task(self.update_routing_table(peer_id, peer, responded=True))
            assert len(keys) == len(response.results), "DHTProtocol: response is not aligned with keys"

            output = {}  # unpack data depending on its type
            for key, result in zip(keys, response.results):
                nearest = dict(zip(map(DHTID.from_bytes, result.nearest_node_ids), result.nearest_endpoints))

                if result.type == dht_pb2.NOT_FOUND:
                    output[key] = None, nearest
                elif result.type == dht_pb2.FOUND_REGULAR:
                    output[key] = ValueWithExpiration(result.value, result.expiration_time), nearest
                elif result.type == dht_pb2.FOUND_DICTIONARY:
                    deserialized_dictionary = self.serializer.loads(result.value)
                    output[key] = ValueWithExpiration(deserialized_dictionary, result.expiration_time), nearest
                else:
                    logger.error(f"Unknown result type: {result.type}")

            return output
        except grpc.aio.AioRpcError as error:
            logger.warning(f"DHTProtocol failed to find at {peer}: {error.code()}")
            asyncio.create_task(self.update_routing_table(self.routing_table.get(endpoint=peer), peer, responded=False))

    async def rpc_find(self, request: dht_pb2.FindRequest, context: grpc.ServicerContext) -> dht_pb2.FindResponse:
        """
        Someone wants to find keys in the DHT. For all keys that we have locally, return value and expiration
        Also return :bucket_size: nearest neighbors from our routing table for each key (whether or not we found value)
        """
        if request.peer:  # if requested, add peer to the routing table
            asyncio.create_task(self.rpc_ping(request.peer, context))
        response = dht_pb2.FindResponse(results=[], peer=self.node_info)
        for i, key_id in enumerate(map(DHTID.from_bytes, request.keys)):
            maybe_item = self.storage.get(key_id)
            cached_item = self.cache.get(key_id)
            if cached_item is not None and (maybe_item is None
                                            or cached_item.expiration_time > maybe_item.expiration_time):
                maybe_item = cached_item

            if maybe_item is None:  # value not found
                item = dht_pb2.FindResult(type=dht_pb2.NOT_FOUND)
            elif isinstance(maybe_item.value, DictionaryDHTValue):
                item = dht_pb2.FindResult(type=dht_pb2.FOUND_DICTIONARY, value=self.serializer.dumps(maybe_item.value),
                                          expiration_time=maybe_item.expiration_time)
            else:  # found regular value
                item = dht_pb2.FindResult(type=dht_pb2.FOUND_REGULAR, value=maybe_item.value,
                                          expiration_time=maybe_item.expiration_time)

            for node_id, endpoint in self.routing_table.get_nearest_neighbors(
                    key_id, k=self.bucket_size, exclude=DHTID.from_bytes(request.peer.node_id)):
                item.nearest_node_ids.append(node_id.to_bytes())
                item.nearest_endpoints.append(endpoint)
            response.results.append(item)
        return response

    async def update_routing_table(self, node_id: Optional[DHTID], peer_endpoint: Endpoint, responded=True):
        """
        This method is called on every incoming AND outgoing request to update the routing table

        :param peer_endpoint: sender endpoint for incoming requests, recipient endpoint for outgoing requests
        :param node_id: sender node id for incoming requests, recipient node id for outgoing requests
        :param responded: for outgoing requests, this indicated whether recipient responded or not.
          For incoming requests, this should always be True
        """
        node_id = node_id if node_id is not None else self.routing_table.get(endpoint=peer_endpoint)
        if responded:  # incoming request or outgoing request with response
            if node_id not in self.routing_table:
                # we just met a new node, maybe we know some values that it *should* store
                data_to_send: List[Tuple[DHTID, BinaryDHTValue, DHTExpiration]] = []
                for key, item in list(self.storage.items()):
                    neighbors = self.routing_table.get_nearest_neighbors(key, self.num_replicas, exclude=self.node_id)
                    if neighbors:
                        nearest_distance = neighbors[0][0].xor_distance(key)
                        farthest_distance = neighbors[-1][0].xor_distance(key)
                        new_node_should_store = node_id.xor_distance(key) < farthest_distance
                        this_node_is_responsible = self.node_id.xor_distance(key) < nearest_distance
                    if not neighbors or (new_node_should_store and this_node_is_responsible):
                        data_to_send.append((key, item.value, item.expiration_time))
                if data_to_send:
                    asyncio.create_task(self.call_store(peer_endpoint, *zip(*data_to_send), in_cache=False))

            maybe_node_to_ping = self.routing_table.add_or_update_node(node_id, peer_endpoint)
            if maybe_node_to_ping is not None:
                # we couldn't add new node because the table was full. Check if existing peers are alive (Section 2.2)
                # ping one least-recently updated peer: if it won't respond, remove it from the table, else update it
                asyncio.create_task(self.call_ping(maybe_node_to_ping[1]))  # [1]-th element is that node's endpoint

        else:  # we sent outgoing request and peer did not respond
            if node_id is not None and node_id in self.routing_table:
                del self.routing_table[node_id]
