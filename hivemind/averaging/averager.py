""" A background process that averages your tensors with peers """

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import multiprocessing as mp
import os
import random
import threading
import weakref
from dataclasses import asdict
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from hivemind.averaging.allreduce import AllreduceException, AllReduceRunner, AveragingMode, GroupID
from hivemind.averaging.control import AveragingStage, StepControl
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import Matchmaking, MatchmakingException
from hivemind.averaging.partition import DEFAULT_PART_SIZE_BYTES
from hivemind.compression import CompressionBase, CompressionInfo, NoCompression, deserialize_torch_tensor
from hivemind.dht import DHT, DHTID
from hivemind.p2p import P2P, P2PContext, P2PDaemonError, P2PHandlerError, PeerID, ServicerBase
from hivemind.proto import averaging_pb2
from hivemind.utils import MPFuture, TensorDescriptor, get_logger
from hivemind.utils.asyncio import (
    achain,
    aiter_with_timeout,
    anext,
    as_aiter,
    azip,
    enter_asynchronously,
    switch_to_uvloop,
)
from hivemind.utils.serializer import MSGPackSerializer, SerializerBase
from hivemind.utils.streaming import combine_from_streaming, split_for_streaming
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
    :param min_matchmaking_time: when looking for group, wait for requests for at least this many seconds
    :param compression: optionally compress tensors with this compression algorithm before running all-reduce
    :param state_compression: a separate compression strategy for load_state_from_peers (default = no compression)
    :param tensor_infos: CompressionInfo for each respective tensor; this determines how the tensor will be comressed
    :param averaging_alpha: optional "learning rate" for averaging. If specified, local parameters will be shifted
      towards the (estimated) average by this coefficient. By default, local parameters are set equal to average.
    :param request_timeout: when looking for group, wait for a response from leader for at most this many seconds.
    :note: request_timeout must be smaller than min_matchmaking_time to avoid potential deadlocks.
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
    :param declare_state_period: re-declare averager as a donor for load_state_from_peers every this many seconds
    :param allreduce_timeout: spend at most this many seconds for allreduce (after group is formed)
    :param next_chunk_timeout: during all-reduce and load_state_from_peers, if peer does not send next data chunk in
      this number of seconds, consider it failed and proceed with remaining peers. default: no timeout
    :param sender_timeout: during all_reduce, any sender that fails to send tensor chunk within this many seconds from
      previous chunk will be marked as failed and excluded from averaging. default: equal to next_chunk_timeout
    :param reducer_timeout: during all_reduce, any reducer that fails to send results chunk within this many seconds
      from previous chunk will be marked as failed and excluded from averaging. default: 2 * sender_timeout
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
    _pending_groups_registered: asyncio.Event
    _state_updated: asyncio.Event
    _p2p: P2P
    serializer = MSGPackSerializer

    def __init__(
        self,
        averaged_tensors: Sequence[torch.Tensor],
        dht: DHT,
        *,
        start: bool,
        prefix: str,
        target_group_size: Optional[int] = None,
        min_group_size: int = 2,
        initial_group_bits: str = "",
        min_matchmaking_time: float = 5.0,
        request_timeout: float = 3.0,
        averaging_alpha: float = 1.0,
        part_size_bytes: int = DEFAULT_PART_SIZE_BYTES,
        allreduce_timeout: Optional[float] = None,
        next_chunk_timeout: Optional[float] = None,
        sender_timeout: Optional[float] = None,
        reducer_timeout: Optional[float] = None,
        compression: CompressionBase = NoCompression(),
        state_compression: CompressionBase = NoCompression(),
        tensor_infos: Optional[Sequence[CompressionInfo]] = None,
        bandwidth: Optional[float] = None,
        min_vector_size: int = 0,
        auxiliary: bool = False,
        allow_state_sharing: Optional[bool] = None,
        declare_state_period: float = 30,
        client_mode: Optional[bool] = None,
        daemon: bool = True,
        shutdown_timeout: float = 5,
    ):
        assert "." not in prefix, "group prefix must be a string without trailing '.'"
        assert bandwidth is None or (
            bandwidth >= 0 and np.isfinite(np.float32(bandwidth))
        ), "bandwidth must be a non-negative float32"
        assert all(bit in "01" for bit in initial_group_bits)
        assert not client_mode or not auxiliary, "auxiliary peers must accept incoming connections"

        super().__init__()
        self.dht = dht
        self.prefix = prefix

        if client_mode is None:
            client_mode = dht.client_mode
        if sender_timeout is None:
            sender_timeout = next_chunk_timeout
        if reducer_timeout is None:
            reducer_timeout = 2 * sender_timeout if sender_timeout is not None else None

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
        for tensor in self._averaged_tensors:
            assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
            tensor.share_memory_()
        self.total_size = sum(map(torch.Tensor.numel, self._averaged_tensors))
        self.schema_hash = compute_schema_hash(self._averaged_tensors)
        self.shutdown_timeout = shutdown_timeout
        self.next_chunk_timeout = next_chunk_timeout
        self.bandwidth = bandwidth

        self.matchmaking_kwargs = dict(
            servicer_type=type(self),
            prefix=prefix,
            initial_group_bits=initial_group_bits,
            target_group_size=target_group_size,
            min_group_size=min_group_size,
            request_timeout=request_timeout,
            min_matchmaking_time=min_matchmaking_time,
        )
        self.allreduce_kwargs = dict(
            compression=compression,
            part_size_bytes=part_size_bytes,
            min_vector_size=min_vector_size,
            sender_timeout=sender_timeout,
            reducer_timeout=reducer_timeout,
        )
        self._averaging_alpha, self._allreduce_timeout = averaging_alpha, allreduce_timeout
        self._running_groups: Dict[GroupID, asyncio.Future[AllReduceRunner]] = {}

        self._inner_pipe, self._outer_pipe = mp.Pipe(duplex=True)  # a control pipe used to communicate with daemon

        self._allow_state_sharing = mp.Value(ctypes.c_bool, 0)
        self._state_sharing_priority = mp.Value(ctypes.c_double, 0)

        if allow_state_sharing is None:
            allow_state_sharing = not client_mode and not auxiliary
        self.allow_state_sharing = allow_state_sharing
        self.declare_state_period = declare_state_period
        self.state_compression = state_compression
        self.tensor_infos = tensor_infos

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
            raise ValueError("Cannot allow state sharing: averager in client mode cannot share its state")
        else:
            old_value, self._allow_state_sharing.value = self._allow_state_sharing.value, value
            if value != old_value:
                self._outer_pipe.send(("_trigger_declare_load_state", [], {}))

    @property
    def state_sharing_priority(self) -> float:
        """Others will preferentially downloading state from peers with highest priority."""
        return float(self._state_sharing_priority.value)

    @state_sharing_priority.setter
    def state_sharing_priority(self, value: float):
        if value and self.client_mode:
            raise ValueError("State sharing priority is unused: averager in client mode cannot share its state")
        else:
            old_value, self._state_sharing_priority.value = self._state_sharing_priority.value, value
            if self.allow_state_sharing and value != old_value:
                self._outer_pipe.send(("_trigger_declare_load_state", [], {}))

    async def _trigger_declare_load_state(self):
        # note: previously tried to set mp.Event instead of this. Awaiting it in executor caused degradation in py39
        self._state_updated.set()

    @property
    def peer_id(self) -> PeerID:
        return self.dht.peer_id

    @property
    def request_timeout(self):
        return self._matchmaking.request_timeout

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

        pipe_semaphore = asyncio.Semaphore(value=0)
        loop.add_reader(self._inner_pipe.fileno(), pipe_semaphore.release)

        async def _run():
            try:
                self._p2p = await self.dht.replicate_p2p()
                if not self.client_mode:
                    await self.add_p2p_handlers(self._p2p, namespace=self.prefix)
                else:
                    logger.debug("The averager is running in client mode")

                self._matchmaking = Matchmaking(
                    self._p2p,
                    self.schema_hash,
                    self.dht,
                    client_mode=self.client_mode,
                    **self.matchmaking_kwargs,
                )
                if not self.client_mode:
                    asyncio.create_task(self._declare_for_download_periodically())

                self._state_updated = asyncio.Event()
                self._pending_groups_registered = asyncio.Event()
                self._pending_groups_registered.set()
            except Exception as e:
                # Loglevel is DEBUG since normally the exception is propagated to the caller
                logger.debug(e, exc_info=True)
                self._ready.set_exception(e)
                return
            self._ready.set_result(None)

            while True:
                try:
                    await asyncio.wait_for(pipe_semaphore.acquire(), timeout=self.request_timeout)
                except asyncio.TimeoutError:
                    pass
                if not self._inner_pipe.poll():
                    continue
                try:
                    method, args, kwargs = self._inner_pipe.recv()
                except (OSError, ConnectionError, RuntimeError) as e:
                    logger.exception(e)
                    await asyncio.sleep(self.request_timeout)
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
            self._outer_pipe.send(("_shutdown", [self.shutdown_timeout], {}))  # shut down the daemon process
            self._inner_pipe.send(("_SHUTDOWN", None))  # shut down background thread in master
            self.join(self.shutdown_timeout)
            if self.is_alive():
                logger.warning("Averager did not shut down within the grace period; terminating it the hard way")
                self.terminate()
        else:
            logger.exception("Averager shutdown has no effect: the process is already not alive")

    async def _shutdown(self, timeout: Optional[float]) -> None:
        if not self.client_mode:
            await self.remove_p2p_handlers(self._p2p, namespace=self.prefix)

        remaining_tasks = set()
        for group in self._running_groups.values():
            remaining_tasks.update(group.finalize(cancel=True))
        await asyncio.wait_for(asyncio.gather(*remaining_tasks), timeout)

    def __del__(self):
        if self._parent_pid == os.getpid() and self.is_alive():
            self.shutdown()

    def step(
        self,
        gather: Optional[GatheredData] = None,
        scheduled_time: Optional[DHTExpiration] = None,
        weight: Optional[float] = None,
        timeout: Optional[float] = None,
        allow_retries: bool = True,
        require_trigger: bool = False,
        wait: bool = True,
    ) -> Union[Optional[Dict[PeerID, GatheredData]], StepControl]:
        """
        Set up the averager to look for a group and run one round of averaging, return True on success, False on failure

        :param gather: optionally send this informaton to all peers in the next group and gather it from every groupmate
          (this operation is known as all-gather). The gathered data will be available as the output of this function.
        :param scheduled_time: when matchmaking, assume that all-reduce will begin at this moment.
          By default, schedule all-reduce current time plus min_matchmaking_time seconds
        :param weight: averaging weight for this peer, int or float, must be strictly positive
        :param allow_retries: if averager fails to run one round of allreduce, this option will allow it to try again
          within the specified timeout
        :param require_trigger: if True, await for user to call .allow_allreduce() before running all-reduce
        :param timeout: if averager was unable to *find* a group in this many seconds, consider allreduce failed
        :param wait: if True (default), return when finished. Otherwise return StepControl and run in background.
        :returns: on success, update averaged_tensors and return group info; on failure, return None
        """
        if self.mode == AveragingMode.AUX and weight is not None:
            logger.warning("Averager is running in auxiliary mode, weight is unused")
        if scheduled_time is None:
            scheduled_time = get_dht_time() + self.matchmaking_kwargs["min_matchmaking_time"]
        if weight is None:
            weight = float(self.mode != AveragingMode.AUX)
        deadline = get_dht_time() + timeout if timeout is not None else float("inf")
        assert isinstance(weight, (int, float)) and weight >= 0, f"Expected a positive int/float, got {type(weight)}"
        assert not (wait and require_trigger), "Non-asynchronous step cannot wait for trigger (use wait=False)"
        assert scheduled_time < deadline, "Scheduled start time does not fit within timeout"

        user_data_for_gather = self.serializer.dumps(gather)  # serialize here to avoid imports in the averager process
        data_for_gather = self.serializer.dumps([self.bandwidth, self.mode.value, user_data_for_gather])
        step = StepControl(
            scheduled_time=scheduled_time,
            deadline=deadline,
            allow_retries=allow_retries,
            weight=weight,
            data_for_gather=data_for_gather,
        )

        future_for_init = MPFuture()
        self._outer_pipe.send(("_step", [], dict(step=step, future_for_init=future_for_init)))
        step.attach(*future_for_init.result())

        if not require_trigger:
            step.allow_allreduce()
        return step.result() if wait else step

    async def _step(self, *, step: StepControl, future_for_init: MPFuture):
        try:
            trigger, cancel = MPFuture(), MPFuture()
            step.attach(trigger, cancel)
            future_for_init.set_result((trigger, cancel))

            async def find_peers_or_notify_cancel():
                group_info = await self._matchmaking.look_for_group(step)
                if not step.triggered:
                    step.stage = AveragingStage.AWAITING_TRIGGER
                    await step.wait_for_trigger()
                return group_info

            while not step.done():
                try:
                    self._pending_groups_registered.clear()
                    step.stage = AveragingStage.LOOKING_FOR_GROUP
                    matchmaking_task = asyncio.create_task(find_peers_or_notify_cancel())
                    check_cancel_task = asyncio.create_task(step.wait_for_cancel())

                    await asyncio.wait({matchmaking_task, check_cancel_task}, return_when=asyncio.FIRST_COMPLETED)
                    if step.cancelled():
                        matchmaking_task.cancel()
                        raise asyncio.CancelledError()
                    else:
                        check_cancel_task.cancel()

                    group_info = await matchmaking_task

                    if group_info is None:
                        raise AllreduceException("Averaging step failed: could not find a group")

                    with self._register_allreduce_group(group_info):
                        step.stage = AveragingStage.RUNNING_ALLREDUCE

                        step.set_result(
                            await asyncio.wait_for(
                                self._aggregate_with_group(
                                    group_info,
                                    tensor_infos=self.tensor_infos,
                                    weight=step.weight,
                                    **self.allreduce_kwargs,
                                ),
                                timeout=self._allreduce_timeout,
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
                    P2PDaemonError,
                ) as e:
                    if step.done() or not step.allow_retries or get_dht_time() >= step.deadline:
                        if not step.cancelled():
                            logger.exception(e)
                        if not step.done():
                            step.set_exception(e)
                    else:
                        logger.warning(f"{self.__class__.__name__} caught {repr(e)}, retrying")

        except BaseException as e:
            if not step.done():
                step.set_exception(e)
            raise
        finally:
            step.stage = AveragingStage.FINISHED
            if not step.done():
                step.set_exception(
                    RuntimeError(
                        "Internal sanity check failed: averager.step left future pending."
                        " Please report this to hivemind issues."
                    )
                )

    @contextlib.contextmanager
    def _register_allreduce_group(self, group_info: GroupInfo):
        """Register a given group for one or more all-reduce rounds"""
        try:
            self._running_groups[group_info.group_id] = asyncio.Future()
            self._pending_groups_registered.set()
            yield
        finally:
            maybe_future = self._running_groups.pop(group_info.group_id, None)
            if maybe_future is not None and not maybe_future.done():
                logger.warning(f"All-reduce group {group_info.group_id} did not finish.")
            self._pending_groups_registered.set()

    async def _aggregate_with_group(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run aggregation in a given group and update tensors in place, return gathered metadata"""
        try:
            bandwidths, mode_ids, user_gathered_bytes = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)))
            modes = tuple(map(AveragingMode, mode_ids))

            # compute optimal part sizes from peer bandwidths; TODO: replace with proper load balancing
            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            )

            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                await self._run_allreduce_inplace_(local_tensors, group_info, peer_fractions=peer_fractions, **kwargs)
                return user_gathered
        except BaseException as e:
            if isinstance(e, Exception):
                logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")

    async def _run_allreduce_inplace_(
        self, tensors: Sequence[torch.Tensor], group_info: GroupInfo, group_id: Optional[bytes] = None, **kwargs
    ):
        """Run one allreduce process to average tensors inplace. Can be called more than a few times in one aggregation process"""
        group_id = group_info.group_id if group_id is None else group_id

        runner = AllReduceRunner(
            p2p=self._p2p,
            servicer_type=type(self),
            prefix=self.prefix,
            tensors=tensors,
            group_id=group_id,
            ordered_peer_ids=group_info.peer_ids,
            **kwargs,
        )
        assert group_id in self._running_groups, f"Group id {group_id} was not registered in _register_allreduce_group"
        self._running_groups[group_id].set_result(runner)

        if runner.modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX:
            async for tensor, update in azip(as_aiter(*tensors), runner):
                tensor.add_(update, alpha=self._averaging_alpha)
                self.last_updated = get_dht_time()
                self._state_updated.set()
        else:
            async for _ in runner:
                raise ValueError("aux peers should not receive averaged tensors")

    @contextlib.contextmanager
    def get_tensors(self) -> Sequence[torch.Tensor]:
        """
        A contextmanager that gives user access to averaged tensors.
        It is guaranteed that the averager will not modify tensors while this context is active.
        Please do not modify the yielded tensors in-place after the context is released.
        """
        with self.lock_averaged_tensors:
            yield self._averaged_tensors

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
            await self._pending_groups_registered.wait()

        future = self._running_groups.get(request.group_id)
        if future is None:
            yield averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)
            return

        group = await future
        async for message in group.rpc_aggregate_part(achain(as_aiter(request), stream), context):
            yield message

    async def _declare_for_download_periodically(self):
        download_key = f"{self._matchmaking.group_key_manager.prefix}.all_averagers"
        sharing_was_allowed = self.allow_state_sharing
        while True:
            expiration_time = get_dht_time() + self.declare_state_period
            if self.allow_state_sharing or sharing_was_allowed:
                # notify either if sharing is allowed or if it was just switched off (to overwrite previous message)
                asyncio.create_task(
                    asyncio.wait_for(
                        self.dht.store(
                            download_key,
                            subkey=self.peer_id.to_bytes(),
                            value=self.state_sharing_priority if self.allow_state_sharing else None,
                            expiration_time=expiration_time,
                            return_future=True,
                        ),
                        timeout=expiration_time - get_dht_time(),
                    )
                )
                sharing_was_allowed = self.allow_state_sharing

            # report again either in state_declare_period or after the field was changed by the user
            self._state_updated.clear()
            try:
                await asyncio.wait_for(self._state_updated.wait(), timeout=max(0.0, expiration_time - get_dht_time()))
            except asyncio.TimeoutError:
                pass

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
        metadata, tensors, infos = await self._get_current_state_from_host_process()
        if infos is None:
            infos = [CompressionInfo.from_tensor(tensor, key=i) for i, tensor in enumerate(tensors)]
        assert len(tensors) == len(infos)

        for tensor, info in zip(tensors, infos):
            for part in split_for_streaming(self.state_compression.compress(tensor, info, allow_inplace=False)):
                if metadata is not None:
                    yield averaging_pb2.DownloadData(tensor_part=part, metadata=metadata)
                    metadata = None
                else:
                    yield averaging_pb2.DownloadData(tensor_part=part)

    def get_current_state(self) -> Tuple[Any, Sequence[torch.Tensor], Sequence[CompressionInfo]]:
        """
        Get current state and send it to a peer. executed in the host process. Meant to be overriden.
        :returns: a tuple of (small metadata, sequence of torch tensors)
        :note: metadata must be seriablizable with self.serializer (default = MSGPackSerializer)
        """
        with self.get_tensors() as tensors:
            return dict(group_key=self.get_group_bits()), tensors, self.tensor_infos

    async def _get_current_state_from_host_process(self):
        """Executed in the averager process inside rpc_download_state"""
        future = MPFuture()
        self._inner_pipe.send(("_TRIGGER_GET_CURRENT_STATE", future))
        return await future

    def load_state_from_peers(
        self, wait: bool = True, timeout: Optional[float] = None
    ) -> Optional[Tuple[Any, Sequence[torch.Tensor]]]:
        """
        Try to download the latest optimizer state one of the existing peer.
        :returns: on success, return a 2-tuple with (metadata, tensors), where

        - metadata is a small object containing metadata (e.g. hyperparameters, scalars, etc)
        - tensors is a sequence of pytorch tensors meant to contain peer's model weights and optimizer statistics

        The exact contents of both metadata and tensors are determined by get_current_state method
        """
        future = MPFuture()
        self._outer_pipe.send(("_load_state_from_peers", [], dict(timeout=timeout, future=future)))
        return future.result(timeout=timeout) if wait else future

    async def _load_state_from_peers(self, future: MPFuture, timeout: Optional[float] = None):
        if timeout is not None:
            timeout = self.next_chunk_timeout if self.next_chunk_timeout is not None else self.request_timeout
        try:
            key_manager = self._matchmaking.group_key_manager
            peer_priority, _ = self.dht.get(f"{key_manager.prefix}.all_averagers", latest=True) or ({}, None)
            peer_priority = {
                PeerID(peer_id): (float(info.value), random.random())  # using randomness as a tie breaker
                for peer_id, info in peer_priority.items()
                if isinstance(info, ValueWithExpiration) and isinstance(info.value, (float, int))
            }

            if not isinstance(peer_priority, dict) or len(peer_priority) == 0:
                logger.info(f"Averager could not load state from peers: peer dict empty or corrupted {peer_priority}")
                future.set_result(None)
                return

            metadata = None
            for peer in sorted(peer_priority.keys(), key=peer_priority.get, reverse=True):
                if peer != self.peer_id:
                    logger.info(f"Downloading parameters from peer {peer}")
                    try:
                        stub = self.get_stub(self._p2p, peer, namespace=self.prefix)
                        stream = await stub.rpc_download_state(averaging_pb2.DownloadRequest())
                        current_tensor_parts, tensors = [], []

                        # TODO merge this with hivemind.compression.deserialize_tensor_stream
                        async for message in aiter_with_timeout(stream, timeout=timeout):
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
                            logger.debug(f"Peer {peer} did not send its state")
                            continue

                        logger.info(f"Finished downloading state from {peer}")
                        future.set_result((metadata, tensors))
                        return
                    except Exception as e:
                        logger.exception(f"Failed to download state from {peer} - {repr(e)}")

        finally:
            if not future.done():
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
            state = get_current_state()
            assert 0 < len(state) <= 3
            if len(state) != 3:
                state = tuple(state + (None,) * (3 - len(state)))
            state_metadata, state_tensors, tensor_infos = state
            del get_current_state

            state_metadata = serializer.dumps(state_metadata)
            state_tensors = tuple(
                tensor.cpu().detach().requires_grad_(tensor.requires_grad) for tensor in state_tensors
            )
            # note: we cast tensors to CPU on host side to avoid initializing cuda in the guest process
            future.set_result((state_metadata, state_tensors, tensor_infos))
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
