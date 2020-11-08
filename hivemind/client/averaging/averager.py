""" Utilities for averaging common layers in decentralized training """

from __future__ import annotations

import ctypes
import os
from typing import Sequence, Optional, Tuple, Any, Union, Awaitable
from concurrent.futures.thread import ThreadPoolExecutor
from functools import cached_property
import multiprocessing as mp
import asyncio

import torch
import uvloop
import grpc

import hivemind
from hivemind.dht import get_dht_time, DHTExpiration
from hivemind.utils import get_logger, Endpoint, Port, MPFuture
from hivemind.utils.grpc import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.proto import averaging_pb2, averaging_pb2_grpc

from hivemind.client.averaging.protocol import AveragingOutcome, ProtocolState, AnyProtocolState, \
    Idle, LookingForGroup, LeaderWaitingForFollowers, FollowerWaitingForLeader, RunningAllReduce

logger = get_logger(__file__)


class DecentralizedAverager(mp.Process, averaging_pb2_grpc.DecentralizedAveragingServicer):
    """
    Gating function averaging service. A trainer can run this service in background to periodically average his gating
    function with other trainers. The averaging pattern is chosen so that (1) you only need to average with a small
    group of peers at a time, but (2) all trainers will converge to global average in a logarithmic number of steps.

    Why averaging is valid: see https://github.com/learning-at-home/hivemind/issues/95#issuecomment-688806705
    On global convergence: see https://github.com/learning-at-home/hivemind/issues/95#issuecomment-717719400

    :param averaged_tensors: a sequence of pytorch tensors that will be averaged in each all-reduce
    :param dht: a DHT node that will be used to find groups
    :param start: if True, starts the background process immediately
    :param bucket_size: averager will try to form groups of approximately this size
    :param initial_ndim: the averager will initially consider bucket_size ^ initial_ndim buckets
      but it may change this value based on target_group_size and the number of peers in each bucket
    :param timeout: consider allreduce failed if there was no activity for this many **seconds**
    :param listen: if True (default), this averager will accept incoming requests from other peers and perform allreduce
            if False, the averager will register as a freeloader and attempt to fetch vectors from other averagers
    :param listen_on: network interface, e.g. "0.0.0.0:1337" or "localhost:*" (* means pick any port) or "[::]:7654"
    :param receiver_threads: uses this many threads to await on input pipe. Default = 1 should be enough in most cases
    :param channel_options: options for grpc.aio.insecure_channel, e.g. [('grpc.enable_retries', 0)]
          see https://grpc.github.io/grpc/core/group__grpc__arg__keys.html for a list of all options
    :param kwargs: extra parameters forwarded to in grpc.aio.server

    You can perform averaging using DecentralizedOptimizer (see below) or by manually running each step as such:

    >>> model = create_my_model()
    >>> averaging = DecentralizedAverager(model.parameters(), dht, start=True)
    >>> averaging.look_for_group()
    >>> for i in range(num_training_steps):
    >>>     train_on_batch(model)
    >>>     if averaging.found_group:
    >>>          averaged_parameters = averaging.all_reduce(inplace=False)
    >>>          with torch.no_grad():
    >>>              for param, averaged_param in zip(model.parameters(), averaged_parameters):
    >>>                  param[...] = averaged_param
    >>>          averaging.look_for_group()
    """

    def __init__(self, averaged_tensors: Sequence[torch.Tensor], dht: hivemind.dht.DHT, *, start: bool,
                 initial_ndim: int = 2, bucket_size: int = 16, max_size: Union[int, str] = 'squared',
                 timeout: float = 15, listen: bool = True, listen_on: Endpoint = '0.0.0.0:*', receiver_threads: int = 1,
                 channel_options: Optional[Sequence[Tuple[str, Any]]] = None, **kwargs):
        super().__init__()
        self.dht = dht
        max_size = bucket_size ** 2 if max_size == 'squared' else max_size
        assert isinstance(max_size, int) and max_size >= bucket_size, "max_size must be an integer >= bucket_size"
        self.initial_ndim, self.bucket_size, self.max_size, self.timeout = initial_ndim, bucket_size, max_size, timeout
        self.server_opts = listen, listen_on, receiver_threads, kwargs
        self.channel_options = channel_options
        self._pipe, self.pipe = mp.Pipe(duplex=True)  # a control pipe used to communicate with a background process
        self._port = mp.Value(ctypes.c_uint32, 0)  # assigned when averager starts, accessible via self.port
        self.ready = mp.Event()
        self._state: Optional[ProtocolState] = None

        self.averaged_tensors = tuple(averaged_tensors)
        for tensor in self.averaged_tensors:
            assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
            tensor.share_memory_()

        if start:
            self.run_in_background(await_ready=True)

    @property
    def state(self) -> AnyProtocolState:
        assert os.getpid() == self.pid, "Protocol state can only be accessed from inside a running averager process"
        self._state = self._state if self._state is not None else Idle()
        return self._state

    @state.setter
    def state(self, state: AnyProtocolState):
        self._state = state

    @cached_property
    def lock_concurrent_requests(self):
        assert os.getpid() == self.pid, "This lock is only available from inside the averager process"
        return asyncio.Lock()

    def run(self):
        """ Serve DecentralizedAverager forever. This function will not return until the averager is shut down """
        if asyncio.get_event_loop().is_running():
            asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop

        uvloop.install()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        listen, listen_on, receiver_threads, server_kwargs = self.server_opts
        pipe_awaiter = ThreadPoolExecutor(receiver_threads)

        async def _run():
            if listen:
                grpc.aio.init_grpc_aio()
                server = grpc.aio.server(**server_kwargs)
                averaging_pb2_grpc.add_DecentralizedAveragingServicer_to_server(self, server)
                found_port = server.add_insecure_port(listen_on)
                assert found_port != 0, f"Failed to listen to {listen_on}"
                self._port.value = found_port
                await server.start()
                self.ready.set()
            else:
                raise NotImplementedError("Client-only averaging is not implemented yet.")

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
            raise TimeoutError("Server didn't notify .ready in {timeout} seconds")

    def shutdown(self) -> None:
        """ Shut down the averager process """
        #TODO send message to pipe:
        # if self._state is LeaderWaitingForPeers, cancel it and wait for termination
        # if waiting for leader, cancel the call to leader
        # if looking for group, remove entries from DHT
        if self.is_alive():
            self.terminate()
        else:
            logger.warning("DHT shutdown has no effect: the process is not alive")

    @property
    def port(self) -> Optional[Port]:
        return self._port.value if self._port.value != 0 else None

    def _get(self, peer: Endpoint) -> averaging_pb2_grpc.DecentralizedAveragingStub:
        """ get a GatingFunctionAveragingStub that sends requests to a given peer """
        channel = grpc.aio.insecure_channel(peer, options=self.channel_options)
        return averaging_pb2_grpc.DecentralizedAveragingStub(channel)

    def step(self) -> bool:
        """ Run the averaging protocol: look for group, then run allreduce inside that group """
        raise NotImplementedError()

    def group_allreduce(self, public_endpoint: Optional[Endpoint] = None, expiration: Optional[DHTExpiration] = None,
                        connect_to: Optional[Endpoint] = None, return_future: bool = False,
                        ) -> Union[AveragingOutcome, Awaitable[AveragingOutcome]]:
        """
        Set up the averager to look for a group and run all-reduce once, optionally await and return outcome
        :param public_endpoint: public endpoint that other peers can use to access this averager TODO remove in favor of dht.get_my_endpoint!
        :param expiration: optionally specify time by which the node should finish looking for group
        :param connect_to: if specified, try to connect directly to the specified leader node
            otherwise, look for a suitable leader in DHT
        :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
        :returns: if wait is True, returns all-reduce outcome
        """
        expiration = expiration if expiration is not None else float('inf')
        assert isinstance(expiration, DHTExpiration)
        assert public_endpoint is not None, "Inferring endpoint is not implemented yet"

        future, _future = MPFuture.make_pair()
        self.pipe.send(('_group_allreduce', [], dict(public_endpoint=public_endpoint, expiration=expiration,
                                                     connect_to=connect_to, future=_future)))
        return future if return_future else future.result()

    async def _group_allreduce(self, *, public_endpoint: Endpoint, expiration: DHTExpiration,
                               connect_to: Optional[Endpoint], future: MPFuture):
        if not isinstance(self.state, Idle):
            outcome = AveragingOutcome(was_accepted_to_group=False, message=f"Averager is busy, state: {self._state}")
            await self._set_allreduce_outcome(outcome)
            future.set_result(outcome)

        assert isinstance(self.state, Idle)
        self.state = self.state.look_for_group(public_endpoint, expiration)
        if connect_to:
            outcome = await self.try_join_group(recipient=connect_to)
            await self._set_allreduce_outcome(outcome)
            future.set_result(outcome)
        else:
            logger.debug("waiting for peers to join my group")
            await self._allreduce_finished()

    async def _allreduce_finished(self) -> AveragingOutcome:
        """ waits for some background task to finish allreduce """
        if self._allreduce_finished_cond is None:
            self._allreduce_finished_cond = asyncio.Condition()
        async with self._allreduce_finished_cond:
            await self._allreduce_finished_cond.wait()
        assert self._last_allreduce_outcome is not None
        return self._last_allreduce_outcome

    async def _set_allreduce_outcome(self, outcome: AveragingOutcome):
        """ set allreduce outcome, notify everyone who is awaiting _allreduce_finished() """
        if self._allreduce_finished_cond is None:
            self._allreduce_finished_cond = asyncio.Condition()
        self._last_allreduce_outcome = outcome
        logger.debug(f"finished allreduce with outcome = {outcome}")
        async with self._allreduce_finished_cond:
            self._allreduce_finished_cond.notify_all()

    async def rpc_group_allreduce(self, request: averaging_pb2.MessageToLeader, context: grpc.ServicerContext):
        """ A peer wants me to be his leader. I will coordinate his actions with the rest of my group. Maybe. """
        assert os.getpid() == self.pid, "this method is only available from inside a running averager process"

        # stage 1: run basic checks and accept or reject the participant
        async with self.lock_concurrent_requests:
            self.state, message = self.state.on_follower_request(request)

            yield message
            if message.code != averaging_pb2.ACCEPTED:
                return

            # at this point, peer passed all our checks and we added him to our (possibly just created) group
            group_state = self.state
            assert isinstance(group_state, LeaderWaitingForFollowers), f"I should be a leader, but got {self.state}"
            assert request.my_endpoint in group_state.group_endpoints

            if len(group_state.group_endpoints) >= self.bucket_size:
                group_state.group_assembled.set()

        try:  # ... wait for enough peers to assemble, handle closed connections in finally
            time_before_expiration = group_state.leader_expiration - get_dht_time()
            try:
                logger.debug(f"Waiting for group to assmble, timeout={group_state.ordered_group_endpoints:.3f}s")
                await asyncio.wait_for(group_state.group_assembled.wait(), timeout=time_before_expiration)
            except asyncio.TimeoutError:
                logger.debug(f"Reached timeout")

            if self.state is group_state:
                logger.debug(f"Starting AllReduce for shards: {group_state.ordered_group_endpoints}")
                self.state = group_state.begin_allreduce()
                asyncio.create_task(self.run_allreduce_part(self.state))

            if isinstance(self.state, RunningAllReduce) and self.state.group_id == group_state.group_id:
                logger.debug(f"Sending allreduce data to peer {request.my_endpoint}")
                assert request.my_endpoint in group_state.ordered_group_endpoints
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.BEGIN_ALLREDUCE,
                                                      group_endpoints=group_state.ordered_group_endpoints)
                await self.state.outcome.wait()  # wait for allreduce to finish (in run_allreduce_part)
            else:
                yield averaging_pb2.MessageFromLeader(code=averaging_pb2.GROUP_DISBANDED)
                return

        finally:
            if self.state is group_state and not group_state.group_assembled.is_set():
                # we get here if the current peer decided to cancel his request before we triggered allreduce
                group_state.remove_follower(request.my_endpoint)
                if group_state.group_size <= 1:
                    self.state = group_state.disband_group()

    ### TODO everything below is a work in progress
    async def try_join_group(self, recipient: Endpoint) -> AveragingOutcome:
        """
        :param recipient: ask this node to be your group leader. Get acccepted or get directions.
        :returns: outcome - whether you got accepted, whether allreduce succeeded, etc.
        """
        assert os.getpid() == self.pid, "this method is only available from inside a running averager process"
        assert isinstance(self._state, LookingForGroup), f"you are not looking for group now ({self._state})"
        logger.debug(f"Attempting to join the group of {recipient} (state={self._state})")
        async with self.lock_concurrent_requests:
            if not isinstance(self._state, LookingForGroup):
                return AveragingOutcome(was_accepted_to_group=False, reason="I got into another group first.")

            stream = self._get(recipient).rpc_group_allreduce( #TODO appropriate typing for steam
                averaging_pb2.MessageToLeader(scheme_hash=b"TODO", leader_endpoint=recipient,
                                              expiration=self._state.my_expiration))
            message = await stream.read()
            if message.code != averaging_pb2.ACCEPTED:
                logger.error(message)
                raise NotImplementedError("TODO handle errors")

            assert message.follower_endpoint == self._state.my_endpoint
            self._state = FollowerWaitingForLeader(self._state.my_endpoint, self._state.my_expiration,
                                                   recipient, message.group_id)

        assert isinstance(self._state, FollowerWaitingForLeader)
        while message.code != averaging_pb2.BEGIN_ALLREDUCE:
            message = await stream.read()
            if message.code == averaging_pb2.GROUP_DISBANDED:
                return AllReduceOutcome(was_accepted_to_group=True, group_dismissed_by_leader=True)
                # TODO should we close stream explicitly?
            # TODO account for timeout / non-response!

        assert message.code == averaging_pb2.BEGIN_ALLREDUCE
        group_endpoints = tuple(message.group)
        my_part_index = group_endpoints.index(self._state.my_endpoint)
        self._state = RunningAllReduce(self._state.my_endpoint, set(group_endpoints), part_index=my_part_index)
        raise NotImplementedError("Actually run allreduce")

        #TODO finally: self._state = Idle()

    async def run_allreduce_part(self, allreduce: RunningAllReduce) -> Sequence[torch.Tensor]:
        """ Send data around following the butterfly all-reduce protocol, return averaged tensors or error """
        assert os.getpid() == self.pid, "this method is only available from inside a running averager process"
        logger.debug(f"Running all_reduce task {allreduce}")
        # TODO save the results somewhere, trigger some event - make sure averager can get the results
        raise NotImplementedError()


    async def rpc_part_averaging(self, request: averaging_pb2.AveragingData, context: grpc.ServicerContext):
        """ A peer sent me his local tensor part. I'll use it to compute the average and return that average to him """
        can_recover_from_error = True
        try:
            assert self._state == ProtocolState.RUNNING_ALLREDUCE, "currently not running allreduce"
            assert not self._finished_accumulating.is_set(), "I already finished accumulating"
            assert request.group_id == self._my_group_id, "group_id doesn't match"
            peer: Endpoint = context.peer()
            assert peer in self._my_group, "You're not in my group"
            assert peer not in self._received_data_from, "I already received a chunk from you"

            can_recover_from_error = False
            self._received_data_from.add(peer)
            self._my_part_accumulator += deserialize_torch_tensor(request.tensor)

            if len(self._received_data_from) >= len(self._my_group):
                assert self._received_data_from == self._my_group, f"Group endpoints ({self._my_group}) do not match " \
                                                                   f"actual peer endpoints ({self._received_data_from})"
                self._my_part_after_averaging = self._my_part_accumulator / len(self._my_group)
                self._finished_accumulating.set()

            await asyncio.wait_for(self._finished_accumulating.wait(), timeout=self._my_expiration - get_dht_time())
            assert self._my_part_after_averaging is not None, "Internal error: averaged vector not found"

            return serialize_torch_tensor(self._my_part_after_averaging, request.tensor.compression, allow_inplace=False)

        except Exception as e:
            if not can_recover_from_error:
                logger.warning(f"DecentralizedAverager failed all-reduce due to {e}")
                self.TODO()
            else:
                logger.warning(f"DecentralizedAverager encountered a correctable error during all-reduce: {e}")
            return averaging_pb2.AveragingData(error=repr(e))

