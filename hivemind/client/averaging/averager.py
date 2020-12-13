""" A background process that averages your tensors with peers """

from __future__ import annotations

import heapq
import os
import random
import ctypes
from dataclasses import asdict
from typing import Sequence, Optional, Tuple, Any, Union, Awaitable, Dict, AsyncIterator, List, Set
from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing as mp
import asyncio

import torch
import uvloop
import grpc

import hivemind
from hivemind.client.averaging.allreduce import GroupAllReduce
from hivemind.client.averaging.matchmaking import Matchmaking
from hivemind.dht import DHTID, DHTExpiration, get_dht_time
from hivemind.utils import get_logger, Endpoint, Port, MPFuture, TensorDescriptor, MSGPackSerializer
from hivemind.utils.grpc import ChannelCache, serialize_torch_tensor, deserialize_torch_tensor
from hivemind.proto import averaging_pb2, averaging_pb2_grpc, runtime_pb2


# flavour types
GroupID = bytes
StreamCallToLeader = grpc.aio.UnaryStreamCall[averaging_pb2.JoinRequest, averaging_pb2.MessageFromLeader]


class AllreduceException(Exception):
    """ A special exception that is raised when allreduce can't continue normally (e.g. disbanded/bad request/etc) """


GROUP_NBITS_INTERVAL = 3
logger = get_logger(__file__)


class DecentralizedAverager(mp.Process, averaging_pb2_grpc.DecentralizedAveragingServicer):
    """
    **Warning!** Decentralized averager is in active development, some critical functionality is still underway

    Gating function averaging service. A trainer can run this service in background to periodically average his gating
    function with other trainers. The averaging pattern is chosen so that (1) you only need to average with a small
    group of peers at a time, but (2) all trainers will converge to global average in a logarithmic number of steps.
    Why averaging is valid: see https://github.com/learning-at-home/hivemind/issues/95#issuecomment-688806705
    On global convergence: see https://github.com/learning-at-home/hivemind/issues/95#issuecomment-717719400

    :param averaged_tensors: a sequence of pytorch tensors that will be averaged in each all-reduce
    :param dht: a DHT node that will be used to find groups
    :param start: if True, starts the background process immediately
    :param initial_group_bits: TODO (also all other examples)
    :param target_group_size:
    :param averaging_expiration: attempt to find a group for this many seconds, otherwise try again
      note - this expiration time only applies to looking for group, passing tensors in allreduce may take more time
    :param allreduce_timeout: spend at most this many seconds for allreduce (after group is formed)
    :param listen: if True (default), this averager will accept incoming requests from other peers and perform allreduce
            if False, the averager will register as a freeloader and attempt to fetch vectors from other averagers
    :param listen_on: network interface, e.g. "0.0.0.0:1337" or "localhost:*" (* means pick any port) or "[::]:7654"
    :param receiver_threads: uses this many threads to await on input pipe. Default = 1 should be enough in most cases
    :param channel_options: options for grpc.aio.insecure_channel, e.g. [('grpc.enable_retries', 0)]
          see https://grpc.github.io/grpc/core/group__grpc__arg__keys.html for a list of all options
    :param kwargs: extra parameters forwarded to in grpc.aio.server
    You can perform averaging using DecentralizedOptimizer (see below) or by manually running each step as such:

    >> TODO add a working example
    # TODO option: use a special key, group_prefix.SOMETHING to state the current number of dimensions
    # each group leader should write this key (to a dict with subkeys) on a sucessful form_group; reader should pick key
    # problem: this may cause excessive load on the peers that store that one special key.
    Alternative: use group_prefix.NDIM{group_nbits[:min_group_nbits]} aka "balance the load between peers 2**min-group-nbits" peers
    Alternative: add such keys to every smaller dimension group down to {GROUP_NBITS_INTERVAL}, check for such keys in your current highest dimension.
        If found, jump to the specified dimension.
    """
    _matchmaking: Matchmaking

    def __init__(self, averaged_tensors: Sequence[torch.Tensor], dht: hivemind.dht.DHT, *, start: bool,
                 prefix: str, target_group_size: int, min_group_size: int = 1, initial_group_bits: Optional[str] = None,
                 averaging_expiration: float = 15, allreduce_timeout: float = float('inf'),
                 compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
                 listen_on: Endpoint = '0.0.0.0:*', receiver_threads: int = 1,
                 channel_options: Optional[Sequence[Tuple[str, Any]]] = None, **kwargs):
        assert '.' not in prefix, "group prefix must be a string without ."
        if target_group_size != 2 ** target_group_size.bit_length():
            logger.warning("It is recommended to set target_group_size to a power of 2.")
        if initial_group_bits is None:
            initial_group_bits = ''.join(random.choices('01', k=GROUP_NBITS_INTERVAL))
            logger.debug(f"Initializing with random {GROUP_NBITS_INTERVAL}-bit group index: {initial_group_bits}")
        assert len(initial_group_bits) >= GROUP_NBITS_INTERVAL

        super().__init__()
        self.dht = dht
        self.listen_on, self.receiver_threads, self.kwargs = listen_on, receiver_threads, kwargs
        self.channel_options = channel_options
        self.averaged_tensors = tuple(averaged_tensors)
        # TODO use mp.Lock to prevent someone from modifying tensors before we copy them! maybe.
        for tensor in self.averaged_tensors:
            assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
            tensor.share_memory_()

        self.matchmaking_kwargs = dict(prefix=prefix, initial_group_bits=initial_group_bits,
                                       target_group_size=target_group_size, min_group_size=min_group_size,
                                       averaging_expiration=averaging_expiration)
        self.allreduce_timeout, self.compression_type = allreduce_timeout, compression_type
        self._running_groups: Dict[GroupID, GroupAllReduce] = {}  # one or more assembled groups that run all-reduce

        self._pipe, self.pipe = mp.Pipe(duplex=True)  # a control pipe used to communicate with a background process
        self._port = mp.Value(ctypes.c_uint32, 0)  # assigned when averager starts, accessible via self.port
        self._averager_endpoint: Optional[Endpoint] = None
        self.ready = mp.Event()  # whether the averager process has started (and ready for incoming requests)

        if start:
            self.run_in_background(await_ready=True)

    @property
    def port(self) -> Optional[Port]:
        return self._port.value if self._port.value != 0 else None

    @property
    def endpoint(self) -> Endpoint:
        if not hasattr(self, '_averager_endpoint'):
            logger.info(f"Assuming averager endpoint to be {self._averager_endpoint}")
            self._averager_endpoint = f"{self.listen_on}:{self.port}"
        return self._averager_endpoint

    def _get_peer_stub(self, peer: Endpoint) -> averaging_pb2_grpc.DecentralizedAveragingStub:
        return ChannelCache.get_stub(peer, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.endpoint}, matchmaking={repr(self._matchmaking)})"

    def run(self):
        """ Serve DecentralizedAverager forever. This function will not return until the averager is shut down """
        if asyncio.get_event_loop().is_running():
            asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop

        uvloop.install()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # initialize asyncio synchronization primitives in this event loop
        pipe_awaiter = ThreadPoolExecutor(self.receiver_threads)

        async def _run():
            grpc.aio.init_grpc_aio()
            server = grpc.aio.server(**self.kwargs)
            averaging_pb2_grpc.add_DecentralizedAveragingServicer_to_server(self, server)
            found_port = server.add_insecure_port(self.listen_on)
            assert found_port != 0, f"Failed to listen to {self.listen_on}"
            self._port.value = found_port
            self._matchmaking = Matchmaking(self.endpoint, self.averaged_tensors, self.dht, **self.matchmaking_kwargs)
            await server.start()
            self.ready.set()

            while True:
                method, args, kwargs = await loop.run_in_executor(pipe_awaiter, self._pipe.recv)
                asyncio.create_task(getattr(self, method)(*args, **kwargs))

        loop.run_until_complete(_run())

    def run_in_background(self, await_ready=True, timeout=None):
        """
        Starts averager in a background process. if await_ready, this method will wait until background dht
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError(f"Server didn't notify .ready in {timeout} seconds")

    def shutdown(self) -> None:
        """ Shut down the averager process """
        # TODO notify peers before terminating
        if self.is_alive():
            self.terminate()
        else:
            logger.warning("DHT shutdown has no effect: the process is not alive")

    def group_allreduce(self, timeout: Optional[float] = None, return_future=False
                        ) -> Union[Sequence[torch.Tensor], Awaitable[Sequence[torch.Tensor]]]:
        """
        Set up the averager to look for a group and run all-reduce once, then return the averaged tensors

        :note: this function implemented for debugging and will be removed in future versions
        :param timeout: if averager was unable to *find* group in this many seconds, consider allreduce failed
        :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
        """
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_group_allreduce', [], dict(future=_future, timeout=timeout)))
        return future if return_future else future.result()

    async def _group_allreduce(self, *, future: MPFuture, timeout: Optional[float]):
        group_id = None
        try:
            group_allreduce = await self._matchmaking.look_for_group(timeout=timeout)
            group_id = group_allreduce.group_id
            if group_allreduce is None:
                future.set_exception(AllreduceException(f"{self} - group_allreduce failed, unable to find group"))
            else:
                assert group_allreduce.group_id not in self._running_groups, (
                    "the new group id matches with one of the pre-existing groups. This is either an error or you "
                    "are fabulously unlucky (160-bit hash collision).")
                #TODO rewrite so that group_allreduce runs the actual allreduce (and implements rpc_aggregate_part)
                self._running_groups[group_id] = group_allreduce
                future.set_result(await self._run_allreduce(group_allreduce.group_id, timeout=self.allreduce_timeout))

        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            if group_id is not None:
                _ = self._running_groups.pop(group_id, None)

    async def _run_allreduce(self, group_id: GroupID, timeout: Optional[float] = None):
        """ send allreduce requests to all peers and collect results, return the averaged tensor """
        assert group_id in self._running_groups, f"unknown group id {group_id}, current groups: {self._running_groups}"
        group_allreduce = self._running_groups[group_id]

        async def _average_one_part(peer_endpoint: Endpoint, local_part: torch.Tensor):
            serialized_tensor_part = serialize_torch_tensor(local_part, self.compression_type, allow_inplace=False)
            response = await self._get_peer_stub(peer_endpoint).rpc_aggregate_part(
                averaging_pb2.AveragingData(code=averaging_pb2.PART_FOR_AVERAGING, group_id=group_allreduce.group_id,
                                            endpoint=group_allreduce.endpoint, tensor_part=serialized_tensor_part))
            if response.code == averaging_pb2.AVERAGED_PART:
                group_allreduce.register_averaged_part(peer_endpoint, deserialize_torch_tensor(response.tensor_part))
            else:
                message_code = averaging_pb2.MessageCode.Name(response.code)
                reference_code = averaging_pb2.MessageCode.Name(response.code)
                group_allreduce.set_exception(AllreduceException(f"peer {peer_endpoint} replied with {message_code}"
                                                                 f" instead of {reference_code}, allreduce failed"))

        try:
            for peer_endpoint, tensor_part in group_allreduce.local_tensor_parts.items():
                if peer_endpoint != self.endpoint:
                    asyncio.create_task(_average_one_part(peer_endpoint, tensor_part))
            return await asyncio.wait_for(group_allreduce.averaged_tensors, timeout=timeout)
        except Exception as e:
            code = averaging_pb2.CANCELLED if isinstance(e, asyncio.CancelledError) else averaging_pb2.INTERNAL_ERROR
            logger.debug(f"{self} - notifying peers about {averaging_pb2.MessageCode.Name(code)}")
            group_allreduce.set_exception(e)

            async def send_error_to_peer(peer_endpoint: Endpoint):
                await self._get_peer_stub(peer_endpoint).rpc_aggregate_part(averaging_pb2.AveragingData(
                    group_id=group_id, endpoint=group_allreduce.endpoint, code=code))

            for peer_endpoint in group_allreduce.ordered_group_endpoints:
                asyncio.create_task(send_error_to_peer(peer_endpoint))

    async def rpc_aggregate_part(self, request: averaging_pb2.AveragingData, context: grpc.ServicerContext):
        """ a groupmate sends us a part of his tensor; we should average it with other peers and return the result """
        if request.group_id not in self._running_groups and not self._received_latest_group_id.is_set():
            await self._received_latest_group_id.wait()  # this handles a special case when leader accepted us to group
            # AND began allreduce right away, but his response with group_id was delayed and other peers got to us first
        if request.group_id not in self._running_groups:
            return averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)
        group_allreduce = self._running_groups[request.group_id]

        if request.code == averaging_pb2.PART_FOR_AVERAGING:
            try:
                tensor_part = deserialize_torch_tensor(request.tensor_part)
                averaged_part = await group_allreduce.accumulate_part(request.endpoint, tensor_part)
                serialized = serialize_torch_tensor(averaged_part, request.tensor_part.compression, allow_inplace=False)
                return averaging_pb2.AveragingData(code=averaging_pb2.AVERAGED_PART, tensor_part=serialized)
            except Exception as e:
                group_allreduce.set_exception(e)
                logger.error(f"{self} - encountered {e} when aggregating part from {request.endpoint}")
                return averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
        else:
            error_code = averaging_pb2.MessageCode.Name(request.code)
            logger.debug(f"{self} - peer {request.endpoint} sent {error_code}, allreduce cannot continue")
            group_allreduce.set_exception(AllreduceException(f"peer {request.endpoint} sent {error_code}."))
            return averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)


def compute_schema_hash(tensors: Sequence[torch.Tensor]) -> bytes:
    """ A hash that describes follower's tensor shapes, dtypes, devices, but not the actual values """
    schema_dicts = [{field_name: str(field_value)
                    for field_name, field_value in asdict(TensorDescriptor.from_tensor(tensor)).items()}
                    for tensor in tensors]
    return DHTID.generate(source=MSGPackSerializer.dumps(schema_dicts)).to_bytes()


def is_valid_join_request(request: averaging_pb2.JoinRequest) -> bool:
    assert len(request.ListFields()) == 3, "this function assumes JoinRequest has three fields, it should be updated"
    return (isinstance(request.schema_hash, bytes) and len(request.schema_hash) > 0 and
            isinstance(request.expiration, DHTExpiration) and request.expiration != float('inf') and
            isinstance(request.endpoint, Endpoint) and len(request.endpoint) > 0)
