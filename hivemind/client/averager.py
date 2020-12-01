""" A background process that averages your tensors with peers """

from __future__ import annotations

import math
import random
import ctypes
from typing import Sequence, Optional, Tuple, Any, Union, Awaitable, Dict
from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing as mp
import asyncio

import torch
import uvloop
import grpc

import hivemind
from hivemind.dht import get_dht_time, DHTExpiration
from hivemind.utils import get_logger, Endpoint, Port, MPFuture
from hivemind.client.allreduce import GroupAllReduce, GroupID, AllreduceException
from hivemind.proto import averaging_pb2, averaging_pb2_grpc

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
    :param timeout: consider allreduce failed if there was no activity for this many **seconds**
    :param initial_group_bits: TODO
    :param target_group_size:
    :param listen: if True (default), this averager will accept incoming requests from other peers and perform allreduce
            if False, the averager will register as a freeloader and attempt to fetch vectors from other averagers
    :param listen_on: network interface, e.g. "0.0.0.0:1337" or "localhost:*" (* means pick any port) or "[::]:7654"
    :param receiver_threads: uses this many threads to await on input pipe. Default = 1 should be enough in most cases
    :param channel_options: options for grpc.aio.insecure_channel, e.g. [('grpc.enable_retries', 0)]
          see https://grpc.github.io/grpc/core/group__grpc__arg__keys.html for a list of all options
    :param kwargs: extra parameters forwarded to in grpc.aio.server
    You can perform averaging using DecentralizedOptimizer (see below) or by manually running each step as such:

    >> TODO add a working example
    """

    def __init__(self, averaged_tensors: Sequence[torch.Tensor], dht: hivemind.dht.DHT, *, start: bool,
                 target_group_size: int, initial_group_bits: Optional[str] = None,
                 timeout: float = 15, listen_on: Endpoint = '0.0.0.0:*',
                 receiver_threads: int = 1, channel_options: Optional[Sequence[Tuple[str, Any]]] = None, **kwargs):
        if target_group_size != 2 ** target_group_size.bit_length():
            logger.warning("It is recommended to set target_group_size to a power of 2.")
        if initial_group_bits is None:
            nbits = max(4, int(math.ceil(math.log2(target_group_size))) * 2)
            initial_group_bits = ''.join(random.choices('01', k=nbits))
            logger.debug(f"Initializing with random {nbits}-bit group index: {initial_group_bits}")

        super().__init__()
        self.dht = dht
        self.listen_on, self.receiver_threads, self.kwargs = listen_on, receiver_threads, kwargs
        self.max_size = target_group_size if target_group_size is not None else float('inf')
        self.timeout = timeout
        self.channel_options = channel_options
        self._pipe, self.pipe = mp.Pipe(duplex=True)  # a control pipe used to communicate with a background process
        self._port = mp.Value(ctypes.c_uint32, 0)  # assigned when averager starts, accessible via self.port
        self._averager_endpoint: Optional[Endpoint] = None
        self.ready = mp.Event()

        self._lock_looking_for_group: Optional[asyncio.Lock] = None
        self._forming_group: Optional[GroupAllReduce] = None  # a group currently in the making (None = not looking)
        self._running_groups: Dict[GroupID, GroupAllReduce] = {}  # one or more groups running all-reduce in background

        self.averaged_tensors = tuple(averaged_tensors)
        for tensor in self.averaged_tensors:
            assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
            tensor.share_memory_()

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

    def run(self):
        """ Serve DecentralizedAverager forever. This function will not return until the averager is shut down """
        if asyncio.get_event_loop().is_running():
            asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop

        uvloop.install()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        listen_on, receiver_threads, server_kwargs = self.listen_on, self.receiver_threads, self.kwargs
        pipe_awaiter = ThreadPoolExecutor(receiver_threads)
        self._lock_looking_for_group = asyncio.Lock()

        async def _run():
            grpc.aio.init_grpc_aio()
            server = grpc.aio.server(**server_kwargs)
            averaging_pb2_grpc.add_DecentralizedAveragingServicer_to_server(self, server)
            found_port = server.add_insecure_port(listen_on)
            assert found_port != 0, f"Failed to listen to {listen_on}"
            self._port.value = found_port
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

    def run_group_allreduce(self, return_future=False) -> Union[Sequence[torch.Tensor], Awaitable[Sequence[torch.Tensor]]]:
        """
        Set up the averager to look for a group and run all-reduce once, then return the averaged tensors

        :note: this function implemented for debugging and will be removed in future versions
        :param my_endpoint: public endpoint of this averager
        :param leader_endpoint: if specified, attempts to join this peer's group
        :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
        """
        expiration = get_dht_time() + self.timeout
        assert isinstance(expiration, DHTExpiration)

        future, _future = MPFuture.make_pair()
        # self.pipe.send(('_run_group_allreduce', [], dict(my_endpoint=my_endpoint, expiration=expiration,
        #                                                  leader_endpoint=leader_endpoint, future=_future)))
        return future if return_future else future.result()

    async def _group_allreduce(self, *, my_endpoint: Endpoint, expiration: DHTExpiration,
                               leader_endpoint: Optional[Endpoint], future: MPFuture):
        group_allreduce = GroupAllReduce(my_endpoint, expiration, self.averaged_tensors)


        try:
            if leader_endpoint is None:
                async with self._lock_forming_a_group:
                    group_allreduce.start_new_group(max_size=self.max_size)
                    self._forming_group = self._pending_groups[group_allreduce.group_id] = group_allreduce
                    await asyncio.wait_for(group_allreduce.assembled_group, expiration - get_dht_time())

                future.set_result(await group_allreduce.run_allreduce())
            else:
                async with self._lock_forming_a_group:
                    accepted = await group_allreduce.request_join_group(leader_endpoint)
                    if not accepted:
                        group_allreduce.set_exception(AllreduceException(f"Rejected by {leader_endpoint}"))
                        raise group_allreduce.exception()

                    self._forming_group = self._pending_groups[group_allreduce.group_id] = group_allreduce
                    started_allreduce = await group_allreduce.wait_for_allreduce()

                    if started_allreduce:
                        future.set_result(await group_allreduce.run_allreduce())
                    else:
                        future.set_exception(group_allreduce.exception())

        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            _ = self._pending_groups.pop(group_allreduce.group_id, None)
            if group_allreduce is self._forming_group:
                self._forming_group = None

    async def rpc_group_allreduce(self, request: averaging_pb2.PeerInfo, context: grpc.ServicerContext):
        """ A peer wants me to be his leader. I will coordinate his actions with the rest of my group. Maybe. """
        if self._forming_group is None:
            yield averaging_pb2.MessageFromLeader(code=averaging_pb2.NOT_LOOKING_FOR_GROUP)
            return
        async for message in self._forming_group.handle_join_request(request):
            yield message

    async def rpc_aggregate_part(self, request: averaging_pb2.AveragingData, context: grpc.ServicerContext):
        if request.group_id not in self._pending_groups:
            return averaging_pb2.AveragingData(code=averaging_pb2.PROTOCOL_VIOLATION)
        return await self._pending_groups[request.group_id].handle_accumulate_request(request)
