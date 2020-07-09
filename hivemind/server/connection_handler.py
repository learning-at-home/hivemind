import asyncio
import multiprocessing as mp
import os
import pickle
from typing import Dict
import torch

import grpc.experimental.aio
import uvloop

from hivemind.runtime.expert_backend import ExpertBackend
from hivemind.utils import compile_grpc, get_logger, serialize_torch_tensor, deserialize_torch_tensor, Endpoint

with open(os.path.join(os.path.dirname(__file__), 'connection_handler.proto')) as f_proto:
    runtime_pb2, runtime_grpc = compile_grpc(f_proto.read())

logger = get_logger(__name__)


class ConnectionHandler(mp.Process):
    """
    A process that accepts incoming requests to experts and submits them into the corresponding TaskPool.

    :note: ConnectionHandler is designed so as to allow using multiple handler processes for the same port.
    :param listen_on: network interface, e.g. "0.0.0.0:1337" or "localhost:*" (* means pick any port) or "[::]:7654"
    :param experts: a dict [UID -> ExpertBackend] with all active experts
    :param max_message_length: maximum size of incoming requests and responses (in bytes)
    """

    def __init__(self, listen_on: Endpoint, experts: Dict[str, ExpertBackend], max_message_length=100 * 1024 * 1024):
        super().__init__()
        self.listen_on, self.experts, self.max_message_length = listen_on, experts, max_message_length
        self.ready = mp.Event()

    def run(self):
        torch.set_num_threads(1)
        uvloop.install()
        loop = asyncio.new_event_loop()

        async def _run():
            grpc.experimental.aio.init_grpc_aio()
            logger.debug(f'Starting, pid {os.getpid()}')
            server = grpc.experimental.aio.server(options=[
                                                      ('grpc.so_reuseport', 1),  # TODO ('grpc.optimization_target', 'throughput')
                                                      ('grpc.max_send_message_length', self.max_message_length),
                                                      ('grpc.max_receive_message_length', self.max_message_length)
                                                  ])
            runtime_grpc.add_ConnectionHandlerServicer_to_server(self, server)

            found_port = server.add_insecure_port(self.listen_on)
            assert found_port != 0, f"Failed to listen to {self.listen_on}"

            await server.start()
            self.ready.set()
            await server.wait_for_termination()
            logger.debug(f"ConnectionHandler terminated: (pid={os.getpid()})")

        loop.run_until_complete(_run())

    async def info(self, request: runtime_pb2.ExpertUID, context: grpc.ServicerContext):
        return runtime_pb2.ExpertInfo(serialized_info=pickle.dumps(self.experts[request.uid].get_info()))

    async def forward(self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext):
        inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        future = self.experts[request.uid].forward_pool.submit_task(*inputs)
        response = await future.async_result()
        serialized_response = [serialize_torch_tensor(tensor) for tensor in response]
        return runtime_pb2.ExpertResponse(tensors=serialized_response)

    async def backward(self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext):
        inputs_and_grad_outputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        future = self.experts[request.uid].backward_pool.submit_task(*inputs_and_grad_outputs)
        response = await future.async_result()
        serialized_response = [serialize_torch_tensor(tensor) for tensor in response]
        return runtime_pb2.ExpertResponse(tensors=serialized_response)
