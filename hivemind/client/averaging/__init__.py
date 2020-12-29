""" A background process that averages your tensors with peers """

from __future__ import annotations

import random
import ctypes
from typing import Sequence, Optional, Tuple, Any, Union, Dict, AsyncIterator
from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing as mp
import asyncio

import torch
import uvloop
import grpc

import hivemind
from hivemind.client.averaging.allreduce import AllReduceRunner, AllreduceException, GroupID
from hivemind.client.averaging.matchmaking import Matchmaking
from hivemind.utils import get_logger, Endpoint, Port, MPFuture, replace_port, GRPC_KEEPALIVE_OPTIONS
from hivemind.proto import averaging_pb2, averaging_pb2_grpc, runtime_pb2

# flavour types
StreamCallToLeader = grpc.aio.UnaryStreamCall[averaging_pb2.JoinRequest, averaging_pb2.MessageFromLeader]

INITIAL_GROUP_NBITS = 3
logger = get_logger(__file__)


class DecentralizedAverager(mp.Process, averaging_pb2_grpc.DecentralizedAveragingServicer):
    """
    **Warning!** Decentralized averager is in active development, some critical functionality is still underway

    Parameter averaging service. A trainer can run this service in background to periodically average his parameters
    with other trainers. The averaging pattern is chosen so that (1) you only need to average with a small
    group of peers at a time, but (2) all trainers will converge to global average in a logarithmic number of steps.

    :param averaged_tensors: a sequence of pytorch tensors that will be averaged in each all-reduce
    :param dht: a DHT node that will be used to find groups
    :param start: if True, starts the background process immediately

    :param prefix: a shared prefix for all group keys
    :param target_group_size: attempts to form groups with up to this many peers (recommended: a power of 2, e.g. 16)
    :param initial_group_bits: a string of bits ('0' and '1') that define initial group key (bucket index)
      by default, sample a random bit sequence of length {INITIAL_GROUP_NBITS}
    :param averaging_expiration: attempt to find a group for this many seconds, otherwise try again
      note - this expiration time only applies to looking for group, passing tensors in allreduce may take more time
    :param compression_type: optionally compress tensors with this compression algorithm before sending them to peers
    :param allreduce_timeout: spend at most this many seconds for allreduce (after group is formed)

    :param listen: if True (default), this averager will accept incoming requests from other peers and perform allreduce
            if False, the averager will register as a freeloader and attempt to fetch vectors from other averagers
    :param listen_on: network interface, e.g. "0.0.0.0:1337" or "localhost:*" (* means pick any port) or "[::]:7654"
    :param receiver_threads: uses this many threads to await on input pipe. Default = 1 should be enough in most cases
    :param channel_options: options for grpc.aio.insecure_channel, e.g. [('grpc.enable_retries', 0)]
          see https://grpc.github.io/grpc/core/group__grpc__arg__keys.html for a list of all options
    :param kwargs: extra parameters forwarded to grpc.aio.server
    You can perform averaging using DecentralizedOptimizer (see below) or by manually running each step as such:

    >> TODO add a working example here
    """
    _matchmaking: Matchmaking
    _pending_group_assembled: asyncio.Event

    def __init__(self, averaged_tensors: Sequence[torch.Tensor], dht: hivemind.dht.DHT, *, start: bool,
                 prefix: str, target_group_size: int, min_group_size: int = 1, initial_group_bits: Optional[str] = None,
                 averaging_expiration: float = 15, allreduce_timeout: Optional[float] = None,
                 compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
                 listen_on: Endpoint = '0.0.0.0:*', receiver_threads: int = 1,
                 channel_options: Optional[Sequence[Tuple[str, Any]]] = None, **kwargs):
        assert '.' not in prefix, "group prefix must be a string without ."
        if is_power_of_two(target_group_size):
            logger.warning("It is recommended to set target_group_size to a power of 2.")
        if initial_group_bits is None:
            initial_group_bits = ''.join(random.choices('01', k=INITIAL_GROUP_NBITS))
            logger.debug(f"Initializing with random {INITIAL_GROUP_NBITS}-bit group index: {initial_group_bits}")
        assert len(initial_group_bits) >= INITIAL_GROUP_NBITS and all(bit in '01' for bit in initial_group_bits)

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
        self._running_groups: Dict[GroupID, AllReduceRunner] = {}  # one or more assembled groups that run all-reduce

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
        if self._averager_endpoint is None:
            self._averager_endpoint = replace_port(self.listen_on, self.port if self.port is not None else '*')
            logger.debug(f"Assuming averager endpoint to be {self._averager_endpoint}")
        return self._averager_endpoint

    def __repr__(self):
        return f"{self.__class__.__name__}({self.endpoint})"

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
            server = grpc.aio.server(**self.kwargs, options=GRPC_KEEPALIVE_OPTIONS)
            averaging_pb2_grpc.add_DecentralizedAveragingServicer_to_server(self, server)
            found_port = server.add_insecure_port(self.listen_on)
            assert found_port != 0, f"Failed to listen to {self.listen_on}"
            self._port.value = found_port
            self._matchmaking = Matchmaking(self.endpoint, self.averaged_tensors, self.dht, **self.matchmaking_kwargs)
            self._pending_group_assembled = asyncio.Event()
            self._pending_group_assembled.set()
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

    def step(self, timeout: Optional[float] = None, return_future=False) -> Union[Sequence[torch.Tensor], MPFuture]:
        """
        Set up the averager to look for a group and run one round of averaging, then return the averaged tensors

        :param timeout: if averager was unable to *find* a group in this many seconds, consider allreduce failedK
        :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
        """
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_step', [], dict(future=_future, timeout=timeout)))
        return future if return_future else future.result()

    async def _step(self, *, future: MPFuture, timeout: Optional[float]):
        group_id = None
        try:
            self._pending_group_assembled.clear()
            allreduce_group = await self._matchmaking.look_for_group(timeout=timeout)
            group_id = allreduce_group.group_id
            if allreduce_group is not None:
                self._running_groups[group_id] = allreduce_group
                self._pending_group_assembled.set()
                future.set_result(await asyncio.wait_for(allreduce_group.run(), self.allreduce_timeout))
            else:
                raise AllreduceException(f"{self} - group_allreduce failed, unable to find a group")

        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            self._pending_group_assembled.set()
            if group_id is not None:
                _ = self._running_groups.pop(group_id, None)

    async def rpc_join_group(self, request: averaging_pb2.JoinRequest, context: grpc.ServicerContext
                             ) -> AsyncIterator[averaging_pb2.MessageFromLeader]:
        """ accept or reject a join request from another averager; if accepted, run him through allreduce steps """
        async for response in self._matchmaking.rpc_join_group(request, context):
            yield response

    async def rpc_aggregate_part(self, request: averaging_pb2.AveragingData, context: grpc.ServicerContext):
        """ a groupmate sends us a part of his tensor; we should average it with other peers and return the result """
        if request.group_id not in self._running_groups and not self._pending_group_assembled.is_set():
            # this handles a special case when leader accepted us to group AND began allreduce right away,
            # but his response with group_id was delayed and other peers got to us first
            await self._pending_group_assembled.wait()
        if request.group_id not in self._running_groups:
            return averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)
        else:
            return await self._running_groups[request.group_id].rpc_aggregate_part(request, context)


def is_power_of_two(n):
    """ Check whether n is a power of 2 """
    return (n != 0) and (n & (n - 1) == 0)
