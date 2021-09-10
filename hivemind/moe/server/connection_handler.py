import asyncio
import multiprocessing as mp
import pickle
from typing import AsyncIterator, Dict

import torch

from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.dht import DHT
from hivemind.moe.server.expert_backend import ExpertBackend
from hivemind.p2p import P2PContext, ServicerBase
from hivemind.proto import runtime_pb2
from hivemind.utils import MPFuture, asingle, get_logger, nested_flatten
from hivemind.utils.asyncio import switch_to_uvloop

logger = get_logger(__name__)


class ConnectionHandler(mp.context.ForkProcess, ServicerBase):
    """
    A process that accepts incoming requests to experts and submits them into the corresponding TaskPool.

    :note: ConnectionHandler is designed so as to allow using multiple handler processes for the same port.
    :param listen_on: network interface, e.g. "0.0.0.0:1337" or "localhost:*" (* means pick any port) or "[::]:7654"
    :param experts: a dict [UID -> ExpertBackend] with all active experts
    """

    def __init__(self, dht: DHT, experts: Dict[str, ExpertBackend]):
        super().__init__()
        self.dht, self.experts = dht, experts

        self.ready = MPFuture()

    def run(self):
        torch.set_num_threads(1)
        loop = switch_to_uvloop()

        async def _run():
            try:
                self._p2p = await self.dht.replicate_p2p()
                await self.add_p2p_handlers(self._p2p)

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
        return runtime_pb2.ExpertInfo(serialized_info=pickle.dumps(self.experts[request.uid].get_info()))

    async def rpc_forward(
        self, stream: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> runtime_pb2.ExpertResponse:
        request = await asingle(stream)

        inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        future = self.experts[request.uid].forward_pool.submit_task(*inputs)
        serialized_response = [
            serialize_torch_tensor(tensor, proto.compression, allow_inplace=True)
            for tensor, proto in zip(await future, nested_flatten(self.experts[request.uid].outputs_schema))
        ]

        return runtime_pb2.ExpertResponse(tensors=serialized_response)

    async def rpc_backward(
        self, stream: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> runtime_pb2.ExpertResponse:
        request = await asingle(stream)

        inputs_and_grad_outputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        future = self.experts[request.uid].backward_pool.submit_task(*inputs_and_grad_outputs)
        serialized_response = [
            serialize_torch_tensor(tensor, proto.compression, allow_inplace=True)
            for tensor, proto in zip(await future, nested_flatten(self.experts[request.uid].grad_inputs_schema))
        ]
        return runtime_pb2.ExpertResponse(tensors=serialized_response)
