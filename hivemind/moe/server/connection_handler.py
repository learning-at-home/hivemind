import multiprocessing as mp
import os
from typing import Dict

import grpc
import torch

from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.moe.server.expert_backend import ExpertBackend
from hivemind.proto import runtime_pb2, runtime_pb2_grpc as runtime_grpc
from hivemind.utils import Endpoint, MSGPackSerializer, get_logger, nested_flatten
from hivemind.utils.asyncio import switch_to_uvloop
from hivemind.utils.grpc import GRPC_KEEPALIVE_OPTIONS

logger = get_logger(__name__)


class ConnectionHandler(mp.context.ForkProcess):
    """
    A process that accepts incoming requests to experts and submits them into the corresponding TaskPool.

    :note: ConnectionHandler is designed so as to allow using multiple handler processes for the same port.
    :param listen_on: network interface, e.g. "0.0.0.0:1337" or "localhost:*" (* means pick any port) or "[::]:7654"
    :param experts: a dict [UID -> ExpertBackend] with all active experts
    """

    def __init__(self, listen_on: Endpoint, experts: Dict[str, ExpertBackend]):
        super().__init__()
        self.listen_on, self.experts = listen_on, experts
        self.ready = mp.Event()

    def run(self):
        torch.set_num_threads(1)
        loop = switch_to_uvloop()

        async def _run():
            grpc.aio.init_grpc_aio()
            logger.debug(f"Starting, pid {os.getpid()}")
            server = grpc.aio.server(
                options=GRPC_KEEPALIVE_OPTIONS
                + (
                    ("grpc.so_reuseport", 1),
                    ("grpc.max_send_message_length", -1),
                    ("grpc.max_receive_message_length", -1),
                )
            )
            runtime_grpc.add_ConnectionHandlerServicer_to_server(self, server)

            found_port = server.add_insecure_port(self.listen_on)
            assert found_port != 0, f"Failed to listen to {self.listen_on}"

            await server.start()
            self.ready.set()
            await server.wait_for_termination()
            logger.debug(f"ConnectionHandler terminated: (pid={os.getpid()})")

        try:
            loop.run_until_complete(_run())
        except KeyboardInterrupt:
            logger.debug("Caught KeyboardInterrupt, shutting down")

    async def info(self, request: runtime_pb2.ExpertUID, context: grpc.ServicerContext):
        return runtime_pb2.ExpertInfo(serialized_info=MSGPackSerializer.dumps(self.experts[request.uid].get_info()))

    async def forward(self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext):
        inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        future = self.experts[request.uid].forward_pool.submit_task(*inputs)
        serialized_response = [
            serialize_torch_tensor(tensor, proto.compression, allow_inplace=True)
            for tensor, proto in zip(await future, nested_flatten(self.experts[request.uid].outputs_schema))
        ]

        return runtime_pb2.ExpertResponse(tensors=serialized_response)

    async def backward(self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext):
        inputs_and_grad_outputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        future = self.experts[request.uid].backward_pool.submit_task(*inputs_and_grad_outputs)
        serialized_response = [
            serialize_torch_tensor(tensor, proto.compression, allow_inplace=True)
            for tensor, proto in zip(await future, nested_flatten(self.experts[request.uid].grad_inputs_schema))
        ]
        return runtime_pb2.ExpertResponse(tensors=serialized_response)
