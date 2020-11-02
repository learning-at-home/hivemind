""" Utilities for averaging common layers in decentralized training """

from __future__ import annotations

import ctypes
import os
from typing import Sequence, Iterable, Optional, Tuple, Any, Set, AsyncGenerator
from dataclasses import dataclass, field
from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing as mp
import asyncio

import torch
import uvloop
import grpc

import hivemind
from hivemind.dht import get_dht_time, DHTExpiration
from hivemind.utils import nested_flatten, get_logger, Endpoint, deserialize_torch_tensor, serialize_torch_tensor, Port
from hivemind.proto import averaging_pb2, averaging_pb2_grpc

logger = get_logger(__file__)


class DecentralizedOptimizer:
    def __init__(self, optimizer: torch.optim.optimizer.Optimizer, *, dht: hivemind.dht.DHT,
                 average_optimizer: bool = False, averaged_tensors: Optional[Iterable[torch.Tensor]] = None, **kwargs):
        """
        A wrapper for torch optimizer that will periodically average model parameters with other peers.
        :param optimizer: an normal pytorch optimizer to be wrapped
        :param dht: a DHT node that will be used to find groups
        :param average_optimizer: if True, all-reduce will aggregate both model parameters and optimizer,
           otherwise average only model parameters (or averaged_tensors, if specified)
        :param averaged_tensors: manually specify all tensors that should be averaged
        :param kwargs: see DecentralizedAverager parameters
        """
        self.optimizer, self.dht, self._averager, self._called_step = optimizer, dht, None, False
        assert not average_optimizer or (averaged_tensors is None), "Please use either average_optimizer=True or" \
                                                                    " averaged_tensors, but not both."
        self.averager_opts = average_optimizer, averaged_tensors, kwargs

    @property
    def averager(self) -> DecentralizedAverager:
        if not self._called_step:
            raise ValueError("DecentralizedOptimizer.averager will be created when you call .step() for the first time")
        if self._averager is not None:
            return self._averager

        average_optimizer, averaged_tensors, kwargs = self.averager_opts
        if averaged_tensors is None:
            averaged_tensors = [param for param_group in self.optimizer.param_groups for param in param_group['params']]
            if average_optimizer:
                found_optimizer_stats = False
                for value in nested_flatten(self.optimizer.state_dict()):
                    if isinstance(value, torch.Tensor) and value not in averaged_tensors:
                        averaged_tensors.append(value)
                        found_optimizer_stats = True
                if not found_optimizer_stats:
                    logger.warning("Using average_optimizer=True, but found no optimizer statistics. Make sure your "
                                   "optimizer has tensors in its .state_dict().")
        else:
            averaged_tensors = list(averaged_tensors)

        self._averager = DecentralizedAverager(averaged_tensors, dht=self.dht, start=True, **kwargs)
        return self._averager

    def step(self, *args, **kwargs):
        step_result = self.optimizer.step(*args, **kwargs)
        if self.averager.found_group:
            self.averager.all_reduce(inplace=True)  # TODO background averaging a-la hogwild
        if not self.averager.looking_for_group:
            self.averager.look_for_group()
        return step_result

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def __repr__(self):
        return f"DecentralizedOptimizer({repr(self.optimizer)})"

    def __del__(self):
        logger.info("Deleting DecentralizedOptimizer, shutting down background averaging process")
        if self._averager is not None:
            self._averager.shutdown()


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
                 bucket_size: int = 16, initial_ndim: int = 2, timeout: float = 15,
                 listen: bool = True, listen_on: Endpoint = '0.0.0.0:*', receiver_threads: int = 1,
                 channel_options: Optional[Sequence[Tuple[str, Any]]] = None, **kwargs):
        super().__init__()
        self.dht = dht
        self.bucket_size, self.initial_ndim, self.timeout = bucket_size, initial_ndim, timeout
        self.server_opts = listen, listen_on, receiver_threads, kwargs
        self.channel_options = channel_options
        self._pipe, self.pipe = mp.Pipe(duplex=True)  # a control pipe used to communicate with a background process
        self._port = mp.Value(ctypes.c_uint32, 0)  # assigned when averager starts, accessible via self.port
        self.ready = mp.Event()

        self.averaged_tensors = tuple(averaged_tensors)
        for tensor in self.averaged_tensors:
            assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
            tensor.share_memory_()

        # internal protocol state variables, only available from inside a running averager process
        self._state: ProtocolState = Idle()

        if start:
            self.run_in_background(await_ready=True)

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

    def test_run(self, group_leader: Optional[Endpoint] = None):
        raise NotImplementedError()

    async def attempt_join_group(self, recipient: Endpoint) -> AllreduceOutcome:
        """
        :param recipient: ask this node to be your group leader. Get acccepted or get directions.
        :returns: a tuple of (outcome,
        """
        assert os.getpid() == self.pid, "this method is only available from inside a running averager process"
        assert isinstance(self._state, LookingForGroup), f"you are not looking for group now ({self._state})"
        async with self._state.one_request_at_a_time:
            if not isinstance(self._state, LookingForGroup):
                return AllreduceOutcome(was_accepted_to_group=False, reason="I was accepted to another group first.")

            stream = self._get(recipient).rpc_group_allreduce( #TODO appropriate typing for steam
                averaging_pb2.MessageToLeader(scheme_hash=b"TODO", leader_endpoint=recipient,
                                              expiration=self._state.my_expiration))
            message = await stream.read()
            if message.code != averaging_pb2.ACCEPTED:
                raise NotImplementedError("TODO handle errors")


            assert message.follower_endpoint == self._state.my_endpoint
            self._state = FollowerWaitingForLeader(self._state.my_endpoint, self._state.my_expiration,
                                                   recipient, message.group_id)

        assert isinstance(self._state, FollowerWaitingForLeader)
        while message.code != averaging_pb2.BEGIN_ALLREDUCE:
            message = await stream.read()
            if message.code == averaging_pb2.GROUP_DISMISSED:
                return AllreduceOutcome(was_accepted_to_group=True, group_dismissed_by_leader=True)
                # TODO should we close stream explicitly?
            elif message.code == averaging_pb2.GROUP_HEARTBEAT:
                continue
            # TODO account for timeout / non-response!

        assert message.code == averaging_pb2.BEGIN_ALLREDUCE
        group_endpoints = tuple(message.group_endpoints)
        my_part_index = group_endpoints.index(self._state.my_endpoint)
        self._state = RunningAllReduce(self._state.my_endpoint, set(group_endpoints), part_index=my_part_index)
        raise NotImplementedError("Actually run allreduce")

        #TODO finally: self._state = Idle()

    async def rpc_group_allreduce(self, request: averaging_pb2.MessageToLeader, context: grpc.ServicerContext):
        """ A peer wants me to be his leader. I will coordinate his actions with the rest of my group. Maybe. """
        assert os.getpid() == self.pid, "this method is only available from inside a running averager process"
        if isinstance(self._state, Idle):
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_LOOKING_FOR_GROUP)
        elif isinstance(self._state, RunningAllReduce):
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.ALREADY_RUNNING)
        elif isinstance(self._state, FollowerWaitingForLeader):
            return averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_A_LEADER,
                                                   suggested_leader=self._state.leader_endpoint)
        elif isinstance(self._state, LookingForGroup):
            if self._state.my_expiration > (request.expiration or float('-inf')):
                return averaging_pb2.MessageFromLeader(code=averaging_pb2.BAD_EXPIRATION_TIME)
            async with self._state.one_request_at_a_time:
                TODO_make_sure_we_havent_found_something_else

                if self._state in (ProtocolState.LOOKING_FOR_GROUP, ProtocolState.LEADER_WAITING_FOR_PEERS):
                    if self._state == ProtocolState.LOOKING_FOR_GROUP:
                        logger.debug(f"Starting a new group as a leader.")
                        self._state = ProtocolState.LEADER_WAITING_FOR_PEERS
                        self._my_group = set()

                    peer: Endpoint = context.peer()
                    logger.debug(f"Adding {peer} to my group, new size = {len(self._my_group)}")
        #TODO if follower closes channel, make sure we exclude him from group immediately!

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


class ProtocolState:
    """ The current state of the averaging protocol with auxiliary variables """
    def __init__(self):  # note: we could use ABC, but it would be too verbose
        raise ValueError("Please use one of the subclasses")


@dataclass
class Idle(ProtocolState):
    """ the averager is not interested in averaging at this time """


@dataclass
class LookingForGroup(ProtocolState):
    """ i am currently looking for group in a dht """
    my_endpoint: Endpoint
    my_expiration: DHTExpiration
    one_request_at_a_time: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)


@dataclass
class LeaderWaitingForPeers(ProtocolState):
    """ i am a leader of a group, waiting for more peers (or timeout) """
    my_endpoint: Endpoint
    group_expiration: DHTExpiration
    group_endpoints: Set[Endpoint]
    group_id: bytes


@dataclass
class FollowerWaitingForLeader(ProtocolState):
    """ i am in a group, but not a leader; waiting for leader to begin all-reduce """
    my_endpoint: Endpoint
    my_expiration: DHTExpiration
    leader_endpoint: Endpoint
    group_id: bytes


@dataclass
class RunningAllReduce(ProtocolState):
    my_endpoint: Endpoint
    group_endpoints: Set[Endpoint]
    part_index: int

    accumulator: Optional[torch.Tensor] = field(default=None, init=False)  # the sum of incoming vector parts
    average_tensor: Optional[torch.Tensor] = field(default=None, init=False)  # accumulator / group size
    received_from: Set[Endpoint] = field(default_factory=set, init=False)  # peers that have sent me their chunk
    finished_accumulating: asyncio.Event = field(default_factory=asyncio.Event, init=False)

    async def accumulate(self, source: Endpoint, part: torch.Tensor) -> torch.Tensor:
        """ Add your vector to accumulator, wait for all other vectors to be added, return the average """
        assert not self.finished_accumulating.is_set(), "averaging is already finished"
        assert source in self.group_endpoints and source not in self.received_from, "unexpected source endpoint"
        if self.accumulator is None:
            self.accumulator = part.clone()
        else:
            assert part.shape == self.accumulator.shape
            self.accumulator.add_(part)

        self.received_from.add(source)
        if len(self.received_from) == len(self.group_endpoints):
            self.average_tensor = self.accumulator.div_(len(self.received_from))
            self.finished_accumulating.set()
        else:
            await self.finished_accumulating.wait()  # wait for other peers to send their part

        assert self.average_tensor is not None
        return self.average_tensor


@dataclass
class AllreduceOutcome:
    was_accepted_to_group: bool = False
    reason: Any = None
    suggested_leader: Optional[Endpoint] = None
    was_leader: bool = False
    group_dismissed_by_leader: bool = False
    started_allreduce: bool = False
    succeeded_allreduce: bool = False


def split_into_chunks(tensors: Sequence[torch.Tensor], group_size: int) -> Tuple[torch.Tensor]:
    """ combines averaged_tensors into one tensor and splits them into equal chunks of size group_size """
    flat_tensor = torch.cat(tuple(map(torch.Tensor.flatten, tensors)))
    chunk_slices = torch.linspace(start=0, end=len(flat_tensor), steps=group_size + 1, dtype=torch.int64).numpy()
    chunk_slices[-1] = len(flat_tensor)
    return tuple(torch.as_tensor(flat_tensor[chunk_slices[i]: chunk_slices[i + 1]]) for i in range(group_size))


def restore_from_chunks(chunks: Sequence[torch.Tensor], shapes: Sequence[torch.Size]) -> Tuple[torch.Tensor, ...]:
    """ restores the original tensor shapes from chunks obtained by split_into_chunks """
    flat_tensor = torch.cat(list(chunks))
    result_sizes = tuple(map(torch.Size.numel, shapes))
    flat_original_tensors = torch.split_with_sizes(flat_tensor, result_sizes)
    return tuple(map(torch.Tensor.reshape, flat_original_tensors, shapes))




