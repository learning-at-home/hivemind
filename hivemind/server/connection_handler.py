import os
from socket import socket
from typing import Tuple, Dict
import multiprocessing as mp
import asyncio
import grpc.experimental.aio
import numpy as np
import torch

from hivemind.runtime.expert_backend import ExpertBackend
from hivemind.utils import PytorchSerializer, Connection, compile_grpc



with open(os.path.join(os.path.dirname(__file__), 'connection_handler.proto')) as f_proto:
    runtime_pb2, runtime_grpc = compile_grpc(f_proto.read())

class ConnectionHandler(mp.Process):
    def __init__(self, addr: str, port: int, experts: Dict[str, ExpertBackend]):
        super().__init__()
        self.endpoint = f"{addr}:{port}"
        self.experts = experts
        self.ready = mp.Event()
        
    def run(self):
        async def _run():
            print(f'Starting, pid {os.getpid()}')
            server = grpc.experimental.aio.server() #TODO check out params
            runtime_grpc.add_ConnectionHandlerServicer_to_server(self, server)
            # TODO there's got to be a better name than ConnectionHandlerServicer
            
            found_port = server.add_insecure_port(self.endpoint)
            assert found_port != 0, f"Failed to listen to {self.endpoint}"
            self.ready.set()
            
            await server.start()
            await server.wait_for_termination()
            print("ConnectionHandler terminated: (pid={os.getpid()})")
        return asyncio.new_event_loop().run_until_complete(_run())
            
    async def info(self, request: runtime_pb2.ExpertUID, context: grpc.ServicerContext):
        raise NotImplementedError()
        
    async def forward(self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext):
        return runtime_pb2.ExpertResponse(tensors=request.tensors)
    
    async def backward(self, request: runtime_pb2.ExpertRequest, context: grpc.ServicerContext):
        return runtime_pb2.ExpertResponse(tensors=request.tensors)
    
    @staticmethod
    def serialize_torch_tensor(tensor: torch.Tensor) -> runtime_pb2.Tensor:
        array = tensor.numpy()
        proto = runtime_pb2.Tensor(
            buffer=array.tobytes(),
            size=array.shape,
            dtype=array.dtype.name,
            requires_grad=tensor.requires_grad) 
        return proto

    @staticmethod
    def deserialize_torch_tensor(tensor: runtime_pb2.Tensor) -> torch.Tensor:
        array = np.frombuffer(tensor.buffer, dtype=np.dtype(tensor.dtype))
        return torch.as_tensor(array).view(tuple(tensor.size)).requires_grad_(tensor.requires_grad)
        # TODO if you experience segfault or something, replace as_tensor with tensor


def handle_connection(connection_tuple: Tuple[socket, str], experts: Dict[str, ExpertBackend]):
    with Connection(*connection_tuple) as connection:
        try:
            header = connection.recv_header()
            payload = PytorchSerializer.loads(connection.recv_raw())

            if header == 'fwd_':
                uid, inputs = payload
                response = experts[uid].forward_pool.submit_task(*inputs).result()
            elif header == 'bwd_':
                uid, inputs_and_grad_outputs = payload
                response = experts[uid].backward_pool.submit_task(*inputs_and_grad_outputs).result()
            elif header == 'info':
                uid = payload
                response = experts[uid].get_info()
            else:
                raise NotImplementedError(f"Unknown header: {header}")

            connection.send_raw('rest', PytorchSerializer.dumps(response))
        except RuntimeError:
            # socket connection broken
            pass
