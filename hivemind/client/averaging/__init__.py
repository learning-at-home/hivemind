""" A background process that averages your tensors with peers """

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import multiprocessing as mp
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Sequence, Optional, Tuple, Any, Union, Dict, AsyncIterator

import grpc
import torch
import numpy as np

import hivemind
from hivemind.client.averaging.allreduce import AllReduceRunner, AllreduceException, GroupID, split_into_parts
from hivemind.client.averaging.matchmaking import Matchmaking, MatchmakingException
from hivemind.proto import averaging_pb2, averaging_pb2_grpc, runtime_pb2
from hivemind.utils.grpc import ChannelCache, GRPC_KEEPALIVE_OPTIONS, \
    serialize_torch_tensor, deserialize_torch_tensor, split_for_streaming, combine_from_streaming
from hivemind.utils.asyncio import anext, achain, aiter, switch_to_uvloop
from hivemind.utils.timed_storage import get_dht_time, ValueWithExpiration, DHTExpiration
from hivemind.utils.serializer import PickleSerializer, MSGPackSerializer
from hivemind.utils import Endpoint, Port, MPFuture, get_logger

# flavour types
StreamCallToLeader = grpc.aio.UnaryStreamCall[averaging_pb2.JoinRequest, averaging_pb2.MessageFromLeader]
DataForGather = Any
logger = get_logger(__name__)
DEFAULT_CHUNK_SIZE_BYTES = 2 ** 16


class DecentralizedAverager(mp.Process, averaging_pb2_grpc.DecentralizedAveragingServicer):
    """

    Parameter averaging service. A trainer can run this service in background to periodically average his parameters
    with other trainers. The averaging pattern is chosen so that (1) you only need to average with a small
    group of peers at a time, but (2) all trainers will converge to global average in a logarithmic number of steps.

    :param averaged_tensors: a sequence of pytorch tensors that will be averaged in each all-reduce
    :param dht: a DHT node that will be used to find groups
    :param start: if True, starts the background process immediately

    :param prefix: a shared prefix for all group keys
    :param target_group_size: attempts to form groups with up to this many peers (recommended: a power of 2, e.g. 16)
    :param initial_group_bits: a string of bits ('0' and '1') that define the initial group key (bucket index)
    :param averaging_expiration: attempt to find a group for this many seconds, otherwise try again
      note - this expiration time only applies to looking for group, passing tensors in allreduce may take more time
    :param compression_type: optionally compress tensors with this compression algorithm before sending them to peers
    :param allreduce_timeout: spend at most this many seconds for allreduce (after group is formed)
    :param averaging_alpha: optional "learning rate" for averaging. If specified, local parameters will be shifted
      towards the (estimated) average by this coefficient. By default, local parameters are set equal to average.
    :param request_timeout: when looking for group, wait for a response from leader for at most this many seconds.
    :note: request_timeout must be smaller than averaging_expiration to avoid potential deadlocks.
    :param chunk_size_bytes: tensors for AllReduce will be divided into chunks of this size (to improve gRPC throughput)
    :param throughput: if specified, this value represents the network bandwidth available to averager.
          By default, the averager is assumed to have the average bandwidth of his group.
          If throughput == 0, averager will rely on its groupmates to do all the averaging.
    :param listen: if True (default), this averager will accept incoming requests from other peers and perform allreduce
            if False, the averager will register as a freeloader and attempt to fetch vectors from other averagers
    :param listen_on: network interface, e.g. "0.0.0.0:1337" or "localhost:*" (* means pick any port) or "[::]:7654"
    :param receiver_threads: uses this many threads to await on input pipe. Default = 1 should be enough in most cases
    :param channel_options: options for grpc.aio.insecure_channel, e.g. [('grpc.enable_retries', 0)]
          see https://grpc.github.io/grpc/core/group__grpc__arg__keys.html for a list of all options
    :param kwargs: extra parameters forwarded to grpc.aio.server

    Example:

    >>> averager = DecentralizedAverager(...)
    >>> with averager.get_tensors() as tensors:
    >>>     # run some code, modify tensors if necessary
    >>>     tensors[0] += 1
    >>> # do not use tensors after the lock is released
    >>> metadata = averager.step(gather=dict(my_batch_size=32))
    >>> # run averaging once (in-place), gather metadata from groupmates
    >>> with averager.get_tensors() as tensors_after_averaging:
    >>>     pass # use the averaged tensors
    """
    _matchmaking: Matchmaking
    _pending_group_assembled: asyncio.Event
    serializer = MSGPackSerializer

    def __init__(self, averaged_tensors: Sequence[torch.Tensor], dht: hivemind.dht.DHT, *, start: bool,
                 prefix: str, target_group_size: int, min_group_size: int = 2, initial_group_bits: Optional[str] = None,
                 averaging_expiration: float = 15, request_timeout: float = 3, chunk_size_bytes: int = 2 ** 16,
                 allreduce_timeout: Optional[float] = None, averaging_alpha: float = 1.0,
                 compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
                 throughput: Optional[float] = None, min_vector_size: int = 0,
                 listen: bool = True, listen_on: Endpoint = '0.0.0.0:*', receiver_threads: int = 1, daemon: bool = True,
                 channel_options: Optional[Sequence[Tuple[str, Any]]] = None, **kwargs):
        assert '.' not in prefix, "group prefix must be a string without trailing '.'"
        assert throughput is None or (throughput >= 0 and np.isfinite(np.float32(throughput))), \
            "throughput must be a non-negative float32"
        if not listen:
            raise NotImplementedError("Client-only averaging is not implemented yet.")
        if not is_power_of_two(target_group_size):
            logger.warning("It is recommended to set target_group_size to a power of 2.")
        assert initial_group_bits is None or all(bit in '01' for bit in initial_group_bits)

        super().__init__()
        self.dht = dht
        self.listen_on, self.receiver_threads, self.kwargs = listen_on, receiver_threads, kwargs
        self.channel_options = channel_options
        self.daemon = daemon

        self._averaged_tensors = tuple(averaged_tensors)
        self.lock_averaged_tensors = mp.Lock()
        self.last_updated: DHTExpiration = -float('inf')
        for tensor in self._averaged_tensors:
            assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
            tensor.share_memory_()

        self.matchmaking_kwargs = dict(
            prefix=prefix, initial_group_bits=initial_group_bits, target_group_size=target_group_size,
            min_group_size=min_group_size, averaging_expiration=averaging_expiration, request_timeout=request_timeout,
            chunk_size_bytes=chunk_size_bytes, compression_type=compression_type,
            throughput=throughput, min_vector_size=min_vector_size)
        self._averaging_alpha, self._allreduce_timeout = averaging_alpha, allreduce_timeout
        self._running_groups: Dict[GroupID, AllReduceRunner] = {}  # one or more assembled groups that run all-reduce

        self._pipe, self.pipe = mp.Pipe(duplex=True)  # a control pipe used to communicate with a background process
        self._port = mp.Value(ctypes.c_uint32, 0)  # assigned when averager starts, accessible via self.port
        self._averager_endpoint: Optional[Endpoint] = None
        self.ready = mp.Event()  # whether the averager process has started (and ready for incoming requests)

        if start:
            self.run_in_background(await_ready=True)
            hivemind.run_in_background(self._background_thread_fetch_current_state_if_asked)

    @property
    def port(self) -> Optional[Port]:
        return self._port.value if self._port.value != 0 else None

    @property
    def endpoint(self) -> Endpoint:
        assert self.port is not None, "Averager is not running yet"
        if self._averager_endpoint is None:
            self._averager_endpoint = f"{self.dht.get_visible_address()}:{self.port}"
            logger.debug(f"Assuming averager endpoint to be {self._averager_endpoint}")
        return self._averager_endpoint

    def __repr__(self):
        return f"{self.__class__.__name__}({self.endpoint})"

    def run(self):
        """ Serve DecentralizedAverager forever. This function will not return until the averager is shut down """
        loop = switch_to_uvloop()
        # initialize asyncio synchronization primitives in this event loop
        pipe_awaiter = ThreadPoolExecutor(self.receiver_threads)

        async def _run():
            grpc.aio.init_grpc_aio()
            server = grpc.aio.server(**self.kwargs, options=GRPC_KEEPALIVE_OPTIONS)
            averaging_pb2_grpc.add_DecentralizedAveragingServicer_to_server(self, server)
            found_port = server.add_insecure_port(self.listen_on)
            assert found_port != 0, f"Failed to listen to {self.listen_on}"
            self._port.value = found_port
            self._matchmaking = Matchmaking(self.endpoint, self._averaged_tensors, self.dht, **self.matchmaking_kwargs,
                                            return_deltas=True)  # note: we need deltas to make allreduce lock-free
            self._pending_group_assembled = asyncio.Event()
            self._pending_group_assembled.set()
            await server.start()
            self.ready.set()
            asyncio.create_task(self._declare_for_download_periodically())

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

    def step(self, gather: Optional[DataForGather] = None, allow_retries: bool = True, timeout: Optional[float] = None,
             wait=True) -> Union[Optional[Dict[Endpoint, DataForGather]], MPFuture]:
        """
        Set up the averager to look for a group and run one round of averaging, return True on success, False on failure

        :param allow_retries: if averager fails to run one round of allreduce, this option will allow it to try again
          within the specified timeout
        :param gather: optionally send this informaton to all peers in the next group and gather it from every groupmate
          (this operation is known as all-gather). The gathered data will be available as the output of this function.
        :param timeout: if averager was unable to *find* a group in this many seconds, consider allreduce failedK
        :param wait: if True (default), return when finished. Otherwise return MPFuture and run in background.
        :returns: on success, update averaged_tensors and return group info; on failure, return None
        """
        future, _future = MPFuture.make_pair()
        gather_binary = self.serializer.dumps(gather)  # serialize here to avoid loading modules in the averager process
        self.pipe.send(('_step', [], dict(future=_future, gather_binary=gather_binary,
                                          allow_retries=allow_retries, timeout=timeout)))
        return future.result() if wait else future

    async def _step(self, *, future: MPFuture, gather_binary: bytes, allow_retries: bool, timeout: Optional[float]):
        loop = asyncio.get_event_loop()
        start_time = get_dht_time()
        group_id = None

        while not future.done():
            try:
                self._pending_group_assembled.clear()
                allreduce_group = await self._matchmaking.look_for_group(timeout=timeout, data_for_gather=gather_binary)
                if allreduce_group is None:
                    raise AllreduceException("Averaging step failed: could not find a group.")

                group_id = allreduce_group.group_id
                self._running_groups[group_id] = allreduce_group
                self._pending_group_assembled.set()
                await asyncio.wait_for(allreduce_group.run(), self._allreduce_timeout)
                await loop.run_in_executor(None, self.update_tensors, allreduce_group)

                # averaging is finished, exit the loop
                gathered_items = map(self.serializer.loads, allreduce_group.gathered)
                gathered_data_by_peer = dict(zip(allreduce_group.ordered_group_endpoints, gathered_items))
                future.set_result(gathered_data_by_peer)

            except (AllreduceException, MatchmakingException):
                time_elapsed = get_dht_time() - start_time
                if not allow_retries or (timeout is not None and timeout < time_elapsed):
                    future.set_result(None)

            except Exception as e:
                future.set_exception(e)
                raise
            finally:
                _ = self._running_groups.pop(group_id, None)
                self._pending_group_assembled.set()

    def update_tensors(self, allreduce_group: AllReduceRunner):
        """
        a private (extendable) method that applies changes from a finished allreduce to local tensors
        """
        assert allreduce_group.return_deltas and allreduce_group.future.done()
        averaging_deltas = allreduce_group.future.result()

        with torch.no_grad(), self.get_tensors() as local_tensors:
            assert len(local_tensors) == len(self._averaged_tensors)
            for tensor, update in zip(local_tensors, averaging_deltas):
                tensor.add_(update, alpha=self._averaging_alpha)
        self.last_updated = get_dht_time()

    @contextlib.contextmanager
    def get_tensors(self) -> Sequence[torch.Tensor]:
        """
        A contextmanager that gives user access to averaged tensors.
        It is guaranteed that the averager will not modify tensors while this context is active.
        Please do not modify the yielded tensors in-place after the context is released.
        """
        with self.lock_averaged_tensors:
            yield self._averaged_tensors
        self.last_updated = get_dht_time()

    async def rpc_join_group(self, request: averaging_pb2.JoinRequest, context: grpc.ServicerContext
                             ) -> AsyncIterator[averaging_pb2.MessageFromLeader]:
        """ accept or reject a join request from another averager; if accepted, run him through allreduce steps """
        async for response in self._matchmaking.rpc_join_group(request, context):
            yield response

    async def rpc_aggregate_part(self, stream: AsyncIterator[averaging_pb2.AveragingData], context: grpc.ServicerContext
                                 ) -> AsyncIterator[averaging_pb2.AveragingData]:
        """ a groupmate sends us a part of his tensor; we should average it with other peers and return the result """
        request = await anext(stream)
        if request.group_id not in self._running_groups:
            # this handles a special case when leader accepted us to group AND began allreduce right away,
            # but his response with group_id was delayed and other peers got to us first
            await self._pending_group_assembled.wait()

        group = self._running_groups.get(request.group_id)
        if group is None:
            yield averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)
            return

        async for message in group.rpc_aggregate_part(achain(aiter(request), stream), context):
            yield message

    async def _declare_for_download_periodically(self):
        download_key = f'{self._matchmaking.group_key_manager.prefix}.all_averagers'
        while True:
            asyncio.create_task(asyncio.wait_for(self.dht.store(
                download_key, subkey=self.endpoint, value=self.last_updated,
                expiration_time=get_dht_time() + self._matchmaking.averaging_expiration, return_future=True),
                timeout=self._matchmaking.averaging_expiration))
            await asyncio.sleep(self._matchmaking.averaging_expiration)

    async def rpc_download_state(self, request: averaging_pb2.DownloadRequest, context: grpc.ServicerContext
                                 ) -> AsyncIterator[averaging_pb2.DownloadData]:
        """
        Get the up-to-date trainer state from a peer.
        The state consists of two parts: (metadata, tensors)

         - metadata is a small pickle-serialized entry meant to store scalars and hyperparameters
         - tensors is a sequence of pytorch tensors that represent model parameters or optimizer statistics
        """
        chunk_size_bytes = self.matchmaking_kwargs.get('chunk_size_bytes', DEFAULT_CHUNK_SIZE_BYTES)
        metadata, tensors = await self._get_current_state_from_host_process()

        for tensor in tensors:
            for part in split_for_streaming(serialize_torch_tensor(tensor), chunk_size_bytes):
                if metadata is not None:
                    yield averaging_pb2.DownloadData(tensor_part=part, metadata=metadata)
                    metadata = None
                else:
                    yield averaging_pb2.DownloadData(tensor_part=part)

    def get_current_state(self) -> Tuple[Any, Sequence[torch.Tensor]]:
        """
        Get current state and send it to a peer. executed in the host process. Meant to be overriden.
        :returns: a tuple of (serializable_small_metadata, sequence of torch tensors)
        """
        with self.get_tensors() as tensors:
            return dict(group_key=self.get_group_bits()), tensors

    async def _get_current_state_from_host_process(self):
        """ Executed in the averager process inside rpc_download_state """
        future, _future = MPFuture.make_pair()
        self._pipe.send(('_TRIGGER_GET_CURRENT_STATE', _future))
        return await future

    def _background_thread_fetch_current_state_if_asked(self):
        """ Executed in the host process as a background thread. """
        while True:
            trigger, future = self.pipe.recv()
            assert trigger == '_TRIGGER_GET_CURRENT_STATE'
            try:
                state_metadata, state_tensors = self.get_current_state()
                # note: serialize here to avoid initializing cuda in the guest process
                state_metadata = PickleSerializer.dumps(state_metadata)
                state_tensors = tuple(tensor.cpu().detach().requires_grad_(tensor.requires_grad)
                                      for tensor in state_tensors)
                future.set_result((state_metadata, state_tensors))
            except BaseException as e:
                future.set_exception(e)
                logger.warning(e)
                continue

    def load_state_from_peers(self, wait=True) -> Optional[Any]:
        """ Try to download the latest optimizer state one of the existing peer """
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_load_state_from_peers', [], dict(future=_future)))
        return future.result() if wait else future

    async def _load_state_from_peers(self, future: MPFuture):
        key_manager = self._matchmaking.group_key_manager
        peer_priority, _ = self.dht.get(f"{key_manager.prefix}.all_averagers", latest=True) or ({}, None)
        peer_priority = {peer: float(info.value) for peer, info in peer_priority.items()
                         if isinstance(info, ValueWithExpiration) and isinstance(info.value, (float, int))}

        if not isinstance(peer_priority, dict) or len(peer_priority) == 0:
            logger.info(f"Averager could not load state from peers: peer dict is absent or corrupted {peer_priority}.")
            future.set_result(None)
            return

        metadata = None
        for peer in sorted(peer_priority.keys(), key=peer_priority.get, reverse=True):
            if peer != self.endpoint:
                logger.info(f"Downloading parameters from peer {peer}")
                stream = None
                try:
                    leader_stub = ChannelCache.get_stub(peer, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)
                    stream = leader_stub.rpc_download_state(averaging_pb2.DownloadRequest())
                    current_tensor_parts, tensors = [], []
                    async for message in stream:
                        if message.metadata:
                            metadata = PickleSerializer.loads(message.metadata)
                        if message.tensor_part.dtype and current_tensor_parts:
                            # tensor_part.dtype indicates the start of the new tensor, so we should wrap up this one
                            tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))
                            current_tensor_parts = []
                        current_tensor_parts.append(message.tensor_part)
                    if current_tensor_parts:
                        tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))
                    future.set_result((metadata, tensors))
                    self.last_updated = get_dht_time()
                    return
                except grpc.aio.AioRpcError as e:
                    logger.info(f"Failed to download state from {peer} - {e}")
                finally:
                    if stream is not None:
                        await stream.code()

        else:
            logger.warning("Averager could not load state from peers: found no active peers.")
            future.set_result(None)

    def get_group_bits(self, wait: bool = True):
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_get_group_bits', [], dict(future=_future)))
        return future.result() if wait else future

    async def _get_group_bits(self, future: MPFuture):
        future.set_result(self._matchmaking.group_key_manager.group_bits)

    def set_group_bits(self, group_bits: str, wait: bool = True):
        future, _future = MPFuture.make_pair()
        assert all(bit in '01' for bit in group_bits)
        self.pipe.send(('_set_group_bits', [], dict(group_bits=group_bits, future=_future)))
        return future.result() if wait else future

    async def _set_group_bits(self, group_bits: str, future: MPFuture):
        try:
            self._matchmaking.group_key_manager.group_bits = group_bits
            return future.set_result(None)
        except Exception as e:
            if not future.done():
                future.set_exception(e)


def is_power_of_two(n):
    """ Check whether n is a power of 2 """
    return (n != 0) and (n & (n - 1) == 0)
