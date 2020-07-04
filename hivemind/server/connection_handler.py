import asyncio
import multiprocessing as mp
import os
import pickle
from typing import Dict

import grpc.experimental.aio
import uvloop

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
        uvloop.install()
        loop = asyncio.new_event_loop()

        async def _run():
            grpc.experimental.aio.init_grpc_aio()
            logger.debug(f'Starting, pid {os.getpid()}')
            server = grpc.experimental.aio.server(options=[
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
        info = self.experts[request.uid].get_info()
        return runtime_pb2.ExpertInfo(serialized_info=pickle.dumps(info))

    async def forward(self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext):
        inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        response = self.experts[request.uid].forward_pool.submit_task(*inputs).result()
        serialized_response = [serialize_torch_tensor(tensor) for tensor in response]
        return runtime_pb2.ExpertResponse(tensors=serialized_response)

    async def backward(self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext):
        inputs_and_grad_outputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        response = self.experts[request.uid].backward_pool.submit_task(*inputs_and_grad_outputs).result()
        serialized_response = [serialize_torch_tensor(tensor) for tensor in response]
        return runtime_pb2.ExpertResponse(tensors=serialized_response)
