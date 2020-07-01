import os
import pickle
from typing import Tuple, Optional

import grpc
import grpc.experimental.aio
import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from ..utils import nested_flatten, DUMMY, nested_pack, nested_compare, compile_grpc, serialize_torch_tensor, \
    deserialize_torch_tensor

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'server', 'connection_handler.proto')) as f_proto:
    runtime_pb2, runtime_grpc = compile_grpc(f_proto.read())


class RemoteExpert(nn.Module):
    """
    A simple module that runs forward/backward of an expert hosted on a remote machine.
    Works seamlessly with pytorch autograd. (this is essentially a simple RPC function)

    Warning: RemoteExpert currently assumes that you provide it with correct input shapes.
    Sending wrong input shapes can cause RemoteExpert to freeze indefinitely due to error in runtime.

    :param uid: unique expert identifier
    :param host: hostname where server operates
    :param port: port to which server listens
    """

    def __init__(self, uid, host='127.0.0.1', port=8080):
        super().__init__()
        self.uid, self.host, self.port = uid, host, port
        self._info = None

    def forward(self, *args, **kwargs):
        """ Call RemoteExpert for the specified inputs and return its output(s). Compatible with pytorch.autograd. """
        assert len(kwargs) == len(self.info['keyword_names']), f"Keyword args should be {self.info['keyword_names']}"
        kwargs = {key: kwargs[key] for key in self.info['keyword_names']}
        # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors

        forward_inputs = (args, kwargs)

        if not nested_compare(forward_inputs, self.info['forward_schema']):
            raise TypeError(f"Inputs do not match expert input schema. Did you pass the right number of parameters?")

        flat_outputs = _RemoteModuleCall.apply(DUMMY, self.uid, self.host, self.port, *nested_flatten(forward_inputs))
        # Note: we send DUMMY to prevent torch from excluding expert from backward if no other inputs require grad
        return nested_pack(flat_outputs, structure=self.info['outputs_schema'])

    @property
    def info(self):
        if self._info is None:
            with grpc.insecure_channel(f'{self.host}:{self.port}', options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024)
            ]) as channel:
                stub = runtime_grpc.ConnectionHandlerStub(channel)
                outputs = stub.info(runtime_pb2.ExpertUID(uid=self.uid))
            self._info = pickle.loads(outputs.serialized_info)
        return self._info

    def extra_repr(self):
        return f"uid={self.uid}, host={self.host}, port={self.port}"


class _RemoteModuleCall(torch.autograd.Function):
    """ Internal autograd-friendly call of a remote module. For applications, use RemoteExpert instead. """

    @staticmethod
    def forward(ctx, dummy: torch.Tensor,
                uid: str, host: str, port: int, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
        inputs = tuple(map(torch.Tensor.detach, inputs))  # detach to avoid pickling the computation graph
        ctx.uid, ctx.host, ctx.port = uid, host, port
        ctx.save_for_backward(*inputs)

        with grpc.insecure_channel(f'{ctx.host}:{ctx.port}', options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ]) as channel:
            stub = runtime_grpc.ConnectionHandlerStub(channel)
            outputs = stub.forward(runtime_pb2.ExpertRequest(uid=ctx.uid, tensors=[serialize_torch_tensor(tensor) for tensor in inputs]))

        deserialized_outputs = [deserialize_torch_tensor(tensor) for tensor in outputs.tensors]

        return tuple(deserialized_outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        payload = tuple(nested_flatten((ctx.saved_tensors, grad_outputs)))

        with grpc.insecure_channel(f'{ctx.host}:{ctx.port}', options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ]) as channel:
            stub = runtime_grpc.ConnectionHandlerStub(channel)
            grad_inputs = stub.backward(
                runtime_pb2.ExpertRequest(uid=ctx.uid, tensors=[serialize_torch_tensor(tensor) for tensor in payload]))

        deserialized_grad_inputs = [deserialize_torch_tensor(tensor) for tensor in grad_inputs.tensors]
        return (DUMMY, None, None, None, *deserialized_grad_inputs)
