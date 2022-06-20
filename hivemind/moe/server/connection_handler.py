import asyncio
import multiprocessing as mp
from typing import AsyncIterator, Dict, Iterable, List, Optional, Tuple, Union

import torch

from hivemind.compression import deserialize_tensor_stream, deserialize_torch_tensor, serialize_torch_tensor
from hivemind.dht import DHT
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.moe.server.task_pool import TaskPool
from hivemind.p2p import P2PContext, ServicerBase
from hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE, P2P
from hivemind.proto import runtime_pb2
from hivemind.utils import MPFuture, MSGPackSerializer, as_aiter, get_logger, nested_flatten
from hivemind.utils.asyncio import amap_in_executor, switch_to_uvloop
from hivemind.utils.streaming import split_for_streaming
from hivemind.utils.tensor_descr import BatchTensorDescriptor

logger = get_logger(__name__)


class ConnectionHandler(mp.context.ForkProcess, ServicerBase):
    """
    A process that accepts incoming requests to experts and submits them into the corresponding TaskPool.

    :note: ConnectionHandler is designed so as to allow using multiple handler processes for the same port
    :param dht: a running hivemind.dht.DHT, used to let other peers connect to this one
    :param module_backends: a dict [UID -> ModuleBackend] with all active experts
    """

    def __init__(self, dht: DHT, module_backends: Dict[str, ModuleBackend]):
        super().__init__()
        self.dht, self.module_backends = dht, module_backends
        self._p2p: Optional[P2P] = None

        self.ready = MPFuture()

    def run(self):
        torch.set_num_threads(1)
        loop = switch_to_uvloop()

        async def _run():
            try:
                self._p2p = await self.dht.replicate_p2p()
                await self.add_p2p_handlers(self._p2p, balanced=True)

                # wait forever
                await asyncio.Future()

            except Exception as e:
                self.ready.set_exception(e)
                return

        self.ready.set_result(None)

        try:
            loop.run_until_complete(_run())
        except KeyboardInterrupt:
            logger.debug("Caught KeyboardInterrupt, shutting down")

    async def rpc_info(self, request: runtime_pb2.ExpertUID, context: P2PContext) -> runtime_pb2.ExpertInfo:
        module_info = self.module_backends[request.uid].get_info()
        return runtime_pb2.ExpertInfo(serialized_info=MSGPackSerializer.dumps(module_info))

    async def _gather_inputs(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> Tuple[str, List[torch.Tensor]]:
        expert_uid = None

        def _unpack(req: runtime_pb2.ExpertRequest) -> Iterable[runtime_pb2.Tensor]:
            nonlocal expert_uid

            if expert_uid is None:
                expert_uid = req.uid
            elif expert_uid != req.uid:
                raise ValueError("Expert uids differ in one request")

            return req.tensors

        tensors_stream = amap_in_executor(_unpack, requests)
        inputs = await deserialize_tensor_stream(tensors_stream)
        return expert_uid, inputs

    async def _process_inputs(
        self,
        inputs: List[torch.Tensor],
        pool: TaskPool,
        schema: Union[BatchTensorDescriptor, Tuple[BatchTensorDescriptor, ...]],
    ) -> List[runtime_pb2.Tensor]:
        return [
            serialize_torch_tensor(result, proto.compression, allow_inplace=True)
            for result, proto in zip(await pool.submit_task(*inputs), nested_flatten(schema))
        ]

    async def rpc_forward(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        expert = self.module_backends[request.uid]
        return runtime_pb2.ExpertResponse(
            tensors=await self._process_inputs(inputs, expert.forward_pool, expert.outputs_schema)
        )

    async def rpc_forward_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertRequest]:
        uid, inputs = await self._gather_inputs(requests, context)
        expert = self.module_backends[uid]
        output_split = [
            part
            for tensor in await self._process_inputs(inputs, expert.forward_pool, expert.outputs_schema)
            for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
        ]

        async for part in as_aiter(*output_split):
            yield runtime_pb2.ExpertResponse(tensors=[part])

    async def rpc_backward(
        self, request: runtime_pb2.ExpertRequest, context: P2PContext
    ) -> runtime_pb2.ExpertResponse:
        inputs_and_grads = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        expert = self.module_backends[request.uid]
        return runtime_pb2.ExpertResponse(
            tensors=await self._process_inputs(inputs_and_grads, expert.backward_pool, expert.grad_inputs_schema)
        )

    async def rpc_backward_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertResponse]:
        uid, inputs_and_grads = await self._gather_inputs(requests, context)
        expert = self.module_backends[uid]
        output_split = [
            part
            for tensor in await self._process_inputs(inputs_and_grads, expert.backward_pool, expert.grad_inputs_schema)
            for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
        ]

        async for part in as_aiter(*output_split):
            yield runtime_pb2.ExpertResponse(tensors=[part])
