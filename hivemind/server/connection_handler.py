import asyncio
import multiprocessing as mp
import os
import pickle
from typing import Dict
import torch

import grpc.experimental.aio
import uvloop
from concurrent.futures import ThreadPoolExecutor

from hivemind.runtime.expert_backend import ExpertBackend
from hivemind.utils import compile_grpc, get_logger, serialize_torch_tensor, deserialize_torch_tensor

with open(os.path.join(os.path.dirname(__file__), 'connection_handler.proto')) as f_proto:
    runtime_pb2, runtime_grpc = compile_grpc(f_proto.read())

logger = get_logger(__name__)


class ConnectionHandler(mp.Process):
    def __init__(self, addr: str, port: int, experts: Dict[str, ExpertBackend]):
        super().__init__()
        self.endpoint = f"{addr}:{port}"
        self.experts = experts
        self.ready = mp.Event()

    def run(self):
        torch.set_num_threads(1)
        uvloop.install()
        loop = asyncio.new_event_loop()
        loop.set_default_executor(ThreadPoolExecutor(10))

        async def _run():
            grpc.experimental.aio.init_grpc_aio()
            logger.debug(f'Starting, pid {os.getpid()}')
            server = grpc.experimental.aio.server(ThreadPoolExecutor(10),
                                                  options=[
                                                      ('grpc.max_send_message_length', 50 * 1024 * 1024),
                                                      ('grpc.max_receive_message_length', 50 * 1024 * 1024)
                                                  ])
            runtime_grpc.add_ConnectionHandlerServicer_to_server(self, server)

            found_port = server.add_insecure_port(self.endpoint)
            assert found_port != 0, f"Failed to listen to {self.endpoint}"
            self.ready.set()

            await server.start()
            await server.wait_for_termination()
            logger.debug(f"ConnectionHandler terminated: (pid={os.getpid()})")

        loop.run_until_complete(_run())

    async def info(self, request: runtime_pb2.ExpertUID, context: grpc.ServicerContext):
        info = self.experts[request.uid].metadata
        return runtime_pb2.ExpertInfo(serialized_info=pickle.dumps(info))

    async def forward(self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext):
        inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        future = await self.experts[request.uid].forward_pool.submit_task(*inputs)
        response = await future.async_result()
        serialized_response = [serialize_torch_tensor(tensor) for tensor in response]
        return runtime_pb2.ExpertResponse(tensors=serialized_response)

    async def backward(self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext):
        inputs_and_grad_outputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        future = await self.experts[request.uid].backward_pool.submit_task(*inputs_and_grad_outputs)
        response = await future.async_result()
        serialized_response = [serialize_torch_tensor(tensor) for tensor in response]
        return runtime_pb2.ExpertResponse(tensors=serialized_response)
