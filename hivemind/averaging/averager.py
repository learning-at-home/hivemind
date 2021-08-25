""" A background process that averages your tensors with peers """

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import multiprocessing as mp
import os
import threading
import weakref
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from hivemind.averaging.allreduce import AllreduceException, AllReduceRunner, AveragingMode, GroupID
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import Matchmaking, MatchmakingException
from hivemind.averaging.partition import DEFAULT_PART_SIZE_BYTES
from hivemind.dht import DHT, DHTID
from hivemind.p2p import P2PContext, P2PHandlerError, PeerID, ServicerBase
from hivemind.proto import averaging_pb2, runtime_pb2
from hivemind.utils import MPFuture, TensorDescriptor, get_logger
from hivemind.utils.asyncio import achain, aiter, anext, switch_to_uvloop
from hivemind.utils.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.utils.grpc import combine_from_streaming, split_for_streaming
from hivemind.utils.serializer import MSGPackSerializer, SerializerBase
from hivemind.utils.timed_storage import DHTExpiration, ValueWithExpiration, get_dht_time

# flavour types
GatheredData = Any
logger = get_logger(__name__)


class DecentralizedAverager(mp.Process, ServicerBase):
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
    :param part_size_bytes: tensors for AllReduce are processed in parts of up to this size (after compression)
    :param bandwidth: if specified, this value represents the network bandwidth available to averager.
          By default, the averager is assumed to have the average bandwidth of his group.
          If bandwidth == 0, averager will rely on its groupmates to do all the averaging.
    :param client_mode: if False, this averager will accept incoming requests from other peers.
          if True, the averager will only join existing groups where at least one peer has client_mode=False.
          By default, this flag is copied from DHTNode inside the ``dht`` instance.
    :param auxiliary: if this flag is specified, averager.step will only assist others without sending
          local tensors for averaging
    :param allow_state_sharing: if set to True, other peers can download this peer's state. Can be overwritten
      with averager.allow_state_sharing = True / False
    :param shutdown_timeout: when calling .shutdown, wait for up to this many seconds before terminating

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

    def __init__(
        self,
        averaged_tensors: Sequence[torch.Tensor],
        dht: DHT,
        *,
        start: bool,
        prefix: str,
        target_group_size: int,
        min_group_size: int = 2,
        initial_group_bits: str = "",
        averaging_expiration: float = 15,
        request_timeout: float = 3,
        averaging_alpha: float = 1.0,
        part_size_bytes: int = DEFAULT_PART_SIZE_BYTES,
        allreduce_timeout: Optional[float] = None,
        compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
        bandwidth: Optional[float] = None,
        min_vector_size: int = 0,
        auxiliary: bool = False,
        allow_state_sharing: Optional[bool] = None,
        client_mode: Optional[bool] = None,
        daemon: bool = True,
        shutdown_timeout: float = 5,
    ):
        assert "." not in prefix, "group prefix must be a string without trailing '.'"
        assert bandwidth is None or (
            bandwidth >= 0 and np.isfinite(np.float32(bandwidth))
        ), "bandwidth must be a non-negative float32"
        if not is_power_of_two(target_group_size):
            logger.warning("It is recommended to set target_group_size to a power of 2.")
        assert all(bit in "01" for bit in initial_group_bits)
        assert not client_mode or not auxiliary, "auxiliary peers must accept incoming connections"

        super().__init__()
        self.dht = dht
        self.prefix = prefix

        if client_mode is None:
            client_mode = dht.client_mode
        self.client_mode = client_mode

        self._parent_pid = os.getpid()
        if self.client_mode:
            self.mode = AveragingMode.CLIENT
        elif auxiliary:
            self.mode = AveragingMode.AUX
        else:
            self.mode = AveragingMode.NODE
        self.daemon = daemon

        self._averaged_tensors = tuple(averaged_tensors)
        self.lock_averaged_tensors = mp.Lock()
        self.last_updated: DHTExpiration = -float("inf")
        for tensor in self._averaged_tensors:
            assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
            tensor.share_memory_()
        self.total_size = sum(map(torch.Tensor.numel, self._averaged_tensors))
        self.schema_hash = compute_schema_hash(self._averaged_tensors)
        self.shutdown_timeout = shutdown_timeout
        self.bandwidth = bandwidth

        self.matchmaking_kwargs = dict(
            servicer_type=type(self),
            prefix=prefix,
            initial_group_bits=initial_group_bits,
            target_group_size=target_group_size,
            min_group_size=min_group_size,
            averaging_expiration=averaging_expiration,
            request_timeout=request_timeout,
        )
        self.allreduce_kwargs = dict(
            compression_type=compression_type, part_size_bytes=part_size_bytes, min_vector_size=min_vector_size
        )
        self._averaging_alpha, self._allreduce_timeout = averaging_alpha, allreduce_timeout
        self._running_groups: Dict[GroupID, AllReduceRunner] = {}  # one or more assembled groups that run all-reduce

        self._inner_pipe, self._outer_pipe = mp.Pipe(duplex=True)  # a control pipe used to communicate with daemon

        self._allow_state_sharing = mp.Value(ctypes.c_bool, 0)
        if allow_state_sharing is None:
            allow_state_sharing = not client_mode and not auxiliary
        self.allow_state_sharing = allow_state_sharing

        self._ready = MPFuture()
        # note: we create a background thread weakref and with daemon=True to ensure garbage collection
        background_fetcher = threading.Thread(
            daemon=True,
            target=_background_thread_fetch_current_state,
            args=[self.serializer, self._outer_pipe, weakref.WeakMethod(self.get_current_state)],
        )
        background_fetcher.start()
        if start:
            self.run_in_background(await_ready=True)

    @property
    def allow_state_sharing(self) -> bool:
        """if set to True, other peers can download this peer's state"""
        return bool(self._allow_state_sharing.value)

    @allow_state_sharing.setter
    def allow_state_sharing(self, value: bool):
        if value and self.client_mode:
            raise ValueError("Cannot allow state sharing: averager in client mode cannot share its state.")
        else:
            self._allow_state_sharing.value = value

    @property
    def peer_id(self) -> PeerID:
        return self.dht.peer_id

    def run(self):
        """
        Run averager function in a background thread; this is needed to avoid a heisenbug with broken OMP on fork
        Turns out, using a non-main thread creates a separate OMP pool that works even if the original pool is corrupted
        Read more: https://github.com/pytorch/pytorch/issues/17199
        """
        thread = threading.Thread(target=self._run_internal, daemon=True)
        thread.start()
        thread.join()

    def _run_internal(self):
        """Serve DecentralizedAverager forever. This function will not return until the averager is shut down"""
        loop = switch_to_uvloop()
        # initialize asyncio synchronization primitives in this event loop
        with ThreadPoolExecutor(max_workers=1) as pipe_awaiter:

            async def _run():
                try:
                    self._p2p = await self.dht.replicate_p2p()
                    if not self.client_mode:
                        await self.add_p2p_handlers(self._p2p, namespace=self.prefix)
                    else:
                        logger.debug(f"The averager is running in client mode.")

                    self._matchmaking = Matchmaking(
                        self._p2p,
                        self.schema_hash,
                        self.dht,
                        client_mode=self.client_mode,
                        **self.matchmaking_kwargs,
                    )
                    if not self.client_mode:
                        asyncio.create_task(self._declare_for_download_periodically())

                    self._pending_group_assembled = asyncio.Event()
                    self._pending_group_assembled.set()
                except Exception as e:
                    # Loglevel is DEBUG since normally the exception is propagated to the caller
                    logger.debug(e, exc_info=True)
                    self._ready.set_exception(e)
                    return
                self._ready.set_result(None)

                while True:
                    try:
                        method, args, kwargs = await loop.run_in_executor(pipe_awaiter, self._inner_pipe.recv)
                    except (OSError, ConnectionError) as e:
                        logger.exception(e)
                        await asyncio.sleep(self._matchmaking.request_timeout)
                        continue
                    task = asyncio.create_task(getattr(self, method)(*args, **kwargs))
                    if method == "_shutdown":
                        await task
                        break

            loop.run_until_complete(_run())

    def run_in_background(self, await_ready: bool = True, timeout: Optional[float] = None) -> None:
        """
        Starts averager in a background process. if await_ready, this method will wait until background dht
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready:
            self.wait_until_ready(timeout)

    def wait_until_ready(self, timeout: Optional[float] = None) -> None:
        self._ready.result(timeout=timeout)

    def shutdown(self) -> None:
        """Shut down the averager process"""
        if self.is_alive():
            self._outer_pipe.send(("_shutdown", [None], {}))  # shut down the daemon process
            self._inner_pipe.send(("_SHUTDOWN", None))  # shut down background thread in master
            self.join(self.shutdown_timeout)
            if self.is_alive():
                logger.warning("Averager did not shut down within the grace period; terminating it the hard way.")
                self.terminate()
        else:
            logger.exception("Averager shutdown has no effect: the process is already not alive")

    async def _shutdown(self, timeout: Optional[float] = None) -> None:
        remaining_tasks = set()
        for group in self._running_groups.values():
            remaining_tasks.update(group.finalize(cancel=True))
        await asyncio.gather(*remaining_tasks)

    def __del__(self):
        if self._parent_pid == os.getpid() and self.is_alive():
            self.shutdown()

    def step(
        self,
        gather: Optional[GatheredData] = None,
        weight: Optional[float] = None,
        timeout: Optional[float] = None,
        allow_retries: bool = True,
        wait: bool = True,
    ) -> Union[Optional[Dict[PeerID, GatheredData]], MPFuture]:
        """
        Set up the averager to look for a group and run one round of averaging, return True on success, False on failure

        :param gather: optionally send this informaton to all peers in the next group and gather it from every groupmate
          (this operation is known as all-gather). The gathered data will be available as the output of this function.
        :param weight: averaging weight for this peer, int or float, must be strictly positive
        :param allow_retries: if averager fails to run one round of allreduce, this option will allow it to try again
          within the specified timeout
        :param timeout: if averager was unable to *find* a group in this many seconds, consider allreduce failedK
        :param wait: if True (default), return when finished. Otherwise return MPFuture and run in background.
        :returns: on success, update averaged_tensors and return group info; on failure, return None
        """
        if self.mode == AveragingMode.AUX and weight is not None:
            logger.warning("Averager is running in auxiliary mode, weight is unused.")
        if weight is None:
            weight = float(self.mode != AveragingMode.AUX)
        assert isinstance(weight, (int, float)) and weight >= 0, f"Expected a positive int/float, got {type(weight)}"

        future = MPFuture()
        gather_binary = self.serializer.dumps(
            gather
        )  # serialize here to avoid loading modules in the averager process
        self._outer_pipe.send(
            (
                "_step",
                [],
                dict(
                    future=future,
                    gather_binary=gather_binary,
                    weight=weight,
                    allow_retries=allow_retries,
                    timeout=timeout,
                ),
            )
        )
        return future.result() if wait else future

    async def _step(
        self, *, future: MPFuture, gather_binary: bytes, weight: float, allow_retries: bool, timeout: Optional[float]
    ):
        start_time = get_dht_time()

        try:
            while not future.done():
                try:
                    self._pending_group_assembled.clear()
                    data_for_gather = self.serializer.dumps([weight, self.bandwidth, self.mode.value, gather_binary])
                    group_info = await self._matchmaking.look_for_group(
                        timeout=timeout, data_for_gather=data_for_gather
                    )
                    if group_info is None:
                        raise AllreduceException("Averaging step failed: could not find a group.")

                    future.set_result(
                        await asyncio.wait_for(
                            self._run_allreduce(group_info, **self.allreduce_kwargs), self._allreduce_timeout
                        )
                    )
                    # averaging is finished, loop will now exit

                except (
                    AllreduceException,
                    MatchmakingException,
                    AssertionError,
                    StopAsyncIteration,
                    asyncio.CancelledError,
                    asyncio.InvalidStateError,
                    P2PHandlerError,
                ) as e:
                    time_elapsed = get_dht_time() - start_time
                    if not allow_retries or (timeout is not None and timeout < time_elapsed):
                        logger.exception(f"Averager caught {repr(e)}")
                        future.set_exception(e)
                    else:
                        logger.warning(f"Averager caught {repr(e)}, retrying")

        except BaseException as e:
            if not future.done():
                future.set_exception(e)
            raise
        finally:
            if not future.done():
                future.set_exception(
                    RuntimeError(
                        "Internal sanity check failed: averager.step left future pending."
                        " Please report this to hivemind issues."
                    )
                )

    async def _run_allreduce(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run All-Reduce in a given group and update tensors in place, return gathered metadata"""
        try:
            weights, bandwidths, mode_ids, user_gathered = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered)))
            modes = tuple(map(AveragingMode, mode_ids))

            # compute optimal part sizes from peer bandwidths; TODO: replace with proper load balancing
            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            )

            async with self.get_tensors_async() as local_tensors:
                allreduce = AllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id,
                    tensors=local_tensors,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    weights=weights,
                    gathered=user_gathered,
                    modes=modes,
                    **kwargs,
                )

                with self.register_allreduce_group(group_info.group_id, allreduce):

                    # actually run all-reduce
                    averaging_outputs = [output async for output in allreduce]

                    if modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX:
                        assert len(local_tensors) == len(self._averaged_tensors)
                        for tensor, update in zip(local_tensors, averaging_outputs):
                            tensor.add_(update, alpha=self._averaging_alpha)
                        self.last_updated = get_dht_time()

                return allreduce.gathered
        except BaseException as e:
            logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")

    @contextlib.contextmanager
    def register_allreduce_group(self, group_id: GroupID, allreduce: AllReduceRunner):
        """registers a given all-reduce runner to listen for incoming connections"""
        try:
            self._running_groups[group_id] = allreduce
            self._pending_group_assembled.set()
            yield
        finally:
            self._running_groups.pop(group_id, None)
            self._pending_group_assembled.set()

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

    @contextlib.asynccontextmanager
    async def get_tensors_async(self) -> Sequence[torch.Tensor]:
        """Like get_tensors, but uses an asynchronous contextmanager"""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.lock_averaged_tensors.acquire)
            yield self._averaged_tensors
        finally:
            self.lock_averaged_tensors.release()

    async def rpc_join_group(
        self, request: averaging_pb2.JoinRequest, context: P2PContext
    ) -> AsyncIterator[averaging_pb2.MessageFromLeader]:
        """accept or reject a join request from another averager; if accepted, run him through allreduce steps"""
        async for response in self._matchmaking.rpc_join_group(request, context):
            yield response

    async def rpc_aggregate_part(
        self, stream: AsyncIterator[averaging_pb2.AveragingData], context: P2PContext
    ) -> AsyncIterator[averaging_pb2.AveragingData]:
        """a groupmate sends us a part of his tensor; we should average it with other peers and return the result"""
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
        download_key = f"{self._matchmaking.group_key_manager.prefix}.all_averagers"
        while True:
            if self.allow_state_sharing:
                asyncio.create_task(
                    asyncio.wait_for(
                        self.dht.store(
                            download_key,
                            subkey=self.peer_id.to_bytes(),
                            value=self.last_updated,
                            expiration_time=get_dht_time() + self._matchmaking.averaging_expiration,
                            return_future=True,
                        ),
                        timeout=self._matchmaking.averaging_expiration,
                    )
                )
            await asyncio.sleep(self._matchmaking.averaging_expiration)

    async def rpc_download_state(
        self, _request: averaging_pb2.DownloadRequest, _context: P2PContext
    ) -> AsyncIterator[averaging_pb2.DownloadData]:
        """
        Get the up-to-date trainer state from a peer.
        The state consists of two parts: (serialized_metadata, tensors)

         - serialized_metadata is a small serialized bytestring meant to store scalars and hyperparameters
         - tensors is a sequence of pytorch tensors that represent model parameters or optimizer statistics
        """
        if not self.allow_state_sharing:
            return  # deny request and direct peer to the next prospective averager
        metadata, tensors = await self._get_current_state_from_host_process()

        for tensor in tensors:
            for part in split_for_streaming(serialize_torch_tensor(tensor)):
                if metadata is not None:
                    yield averaging_pb2.DownloadData(tensor_part=part, metadata=metadata)
                    metadata = None
                else:
                    yield averaging_pb2.DownloadData(tensor_part=part)

    def get_current_state(self) -> Tuple[Any, Sequence[torch.Tensor]]:
        """
        Get current state and send it to a peer. executed in the host process. Meant to be overriden.
        :returns: a tuple of (small metadata, sequence of torch tensors)
        :note: metadata must be seriablizable with self.serializer (default = MSGPackSerializer)
        """
        with self.get_tensors() as tensors:
            return dict(group_key=self.get_group_bits()), tensors

    async def _get_current_state_from_host_process(self):
        """Executed in the averager process inside rpc_download_state"""
        future = MPFuture()
        self._inner_pipe.send(("_TRIGGER_GET_CURRENT_STATE", future))
        return await future

    def load_state_from_peers(self, wait=True) -> Optional[Tuple[Any, Sequence[torch.Tensor]]]:
        """
        Try to download the latest optimizer state one of the existing peer.
        :returns: on success, return a 2-tuple with (metadata, tensors), where

        - metadata is a small object containing metadata (e.g. hyperparameters, scalars, etc)
        - tensors is a sequence of pytorch tensors meant to contain peer's model weights and optimizer statistics

        The exact contents of both metadata and tensors are determined by get_current_state method
        """
        future = MPFuture()
        self._outer_pipe.send(("_load_state_from_peers", [], dict(future=future)))
        return future.result() if wait else future

    async def _load_state_from_peers(self, future: MPFuture):
        try:
            key_manager = self._matchmaking.group_key_manager
            peer_priority, _ = self.dht.get(f"{key_manager.prefix}.all_averagers", latest=True) or ({}, None)
            peer_priority = {
                PeerID(peer_id): float(info.value)
                for peer_id, info in peer_priority.items()
                if isinstance(info, ValueWithExpiration) and isinstance(info.value, (float, int))
            }

            if not isinstance(peer_priority, dict) or len(peer_priority) == 0:
                logger.info(f"Averager could not load state from peers: peer dict empty or corrupted {peer_priority}.")
                future.set_result(None)
                return

            metadata = None
            for peer in sorted(peer_priority.keys(), key=peer_priority.get, reverse=True):
                if peer != self.peer_id:
                    logger.info(f"Downloading parameters from peer {peer}")
                    try:
                        stub = self.get_stub(self._p2p, peer, namespace=self.prefix)
                        stream = stub.rpc_download_state(averaging_pb2.DownloadRequest())
                        current_tensor_parts, tensors = [], []
                        async for message in stream:
                            if message.metadata:
                                metadata = self.serializer.loads(message.metadata)
                            if message.tensor_part.dtype and current_tensor_parts:
                                # tensor_part.dtype indicates the start of the new tensor, so we should wrap up this one
                                tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))
                                current_tensor_parts = []
                            current_tensor_parts.append(message.tensor_part)
                        if current_tensor_parts:
                            tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))

                        if not metadata:
                            logger.debug(f"Peer {peer} did not send its state.")
                            continue

                        logger.info(f"Finished downloading state from {peer}")
                        future.set_result((metadata, tensors))
                        self.last_updated = get_dht_time()
                        return
                    except Exception as e:
                        logger.exception(f"Failed to download state from {peer} - {repr(e)}")

        finally:
            if not future.done():
                logger.warning("Averager could not load state from peers: all requests have failed.")
                future.set_result(None)

    def get_group_bits(self, wait: bool = True):
        """
        :param wait: if True, return bits immediately. Otherwise return awaitable MPFuture
        :returns: averager's current group key bits (without prefix)
        """
        future = MPFuture()
        self._outer_pipe.send(("_get_group_bits", [], dict(future=future)))
        return future.result() if wait else future

    async def _get_group_bits(self, future: MPFuture):
        future.set_result(self._matchmaking.group_key_manager.group_bits)

    def set_group_bits(self, group_bits: str, wait: bool = True):
        """
        :param group_bits: group bits (string of '0' or '1') to be used in averager's group key
        :param wait: if True, wait until the update is confirmed by the averager. Otherwise return immediately
        """
        future = MPFuture()
        assert all(bit in "01" for bit in group_bits)
        self._outer_pipe.send(("_set_group_bits", [], dict(group_bits=group_bits, future=future)))
        return future.result() if wait else future

    async def _set_group_bits(self, group_bits: str, future: MPFuture):
        try:
            self._matchmaking.group_key_manager.group_bits = group_bits
            return future.set_result(None)
        except Exception as e:
            if not future.done():
                future.set_exception(e)


def is_power_of_two(n):
    """Check whether n is a power of 2"""
    return (n != 0) and (n & (n - 1) == 0)


def _background_thread_fetch_current_state(
    serializer: SerializerBase, pipe: mp.connection.Connection, get_current_state_ref: weakref.WeakMethod
):
    """
    Executed in the host process as a background thread. Fetches the averager state when asked by peers.
    :param serializer: a serializer with which to convert metadata into bytes
    :param pipe: DecentralizedAverager's control pipe (from host process side)
    :param get_current_state_ref: a WeakMethod wrapped around DecentralizedAverager.get_current_state (instance-bound)
    """
    while True:
        try:
            trigger, future = pipe.recv()
        except BaseException as e:
            logger.debug(f"Averager background thread finished: {repr(e)}")
            break

        if trigger == "_SHUTDOWN":
            break

        assert trigger == "_TRIGGER_GET_CURRENT_STATE"
        try:
            get_current_state = get_current_state_ref()
            if get_current_state is None:
                break
            state_metadata, state_tensors = get_current_state()
            del get_current_state

            state_metadata = serializer.dumps(state_metadata)
            state_tensors = tuple(
                tensor.cpu().detach().requires_grad_(tensor.requires_grad) for tensor in state_tensors
            )
            # note: we cast tensors to CPU on host side to avoid initializing cuda in the guest process
            future.set_result((state_metadata, state_tensors))
        except BaseException as e:
            future.set_exception(e)
            logger.warning(e)
            continue


def compute_schema_hash(tensors: Sequence[torch.Tensor]) -> bytes:
    """A hash that describes follower's tensor shapes, dtypes, devices, but not the actual values"""
    schema_dicts = [
        {
            field_name: str(field_value)
            for field_name, field_value in asdict(TensorDescriptor.from_tensor(tensor)).items()
        }
        for tensor in tensors
    ]
    return DHTID.generate(source=schema_dicts).to_bytes()
