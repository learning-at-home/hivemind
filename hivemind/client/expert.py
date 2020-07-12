import pickle
from typing import Tuple, Optional

import grpc
import grpc.experimental.aio
import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from hivemind.utils import nested_flatten, nested_pack, nested_compare, Endpoint
from hivemind.utils.grpc import serialize_torch_tensor, deserialize_torch_tensor, runtime_pb2, runtime_grpc

DUMMY = torch.empty(0, requires_grad=True)  # dummy tensor that triggers autograd in RemoteExpert


class RemoteExpert(nn.Module):
    """
    A simple module that runs forward/backward of an expert hosted on a remote machine.
    Works seamlessly with pytorch autograd. (this is essentially a simple RPC function)

    Warning: RemoteExpert currently assumes that you provide it with correct input shapes.
    Sending wrong input shapes can cause RemoteExpert to freeze indefinitely due to error in runtime.

    :param uid: unique expert identifier
    :param endpoint: network endpoint of a server that services that expert, e.g. "201.123.321.99:1337" or "[::]:8080"
    """

    def __init__(self, uid, endpoint: Endpoint):
        super().__init__()
        self.uid, self.endpoint = uid, endpoint
        self._channel, self._stub, self._info = None, None, None  # to be filled via @property

    @property
    def stub(self):
        if self._channel is None:
            self._channel = grpc.insecure_channel(self.endpoint, options=[
                ('grpc.max_send_message_length', -1),
                ('grpc.max_receive_message_length', -1)
            ])
        if self._stub is None:
            self._stub = runtime_grpc.ConnectionHandlerStub(self._channel)
        return self._stub

    def __del__(self):
        if self._channel is not None:
            self._channel.close()

    def forward(self, *args, **kwargs):
        """ Call RemoteExpert for the specified inputs and return its output(s). Compatible with pytorch.autograd. """
        assert len(kwargs) == len(self.info['keyword_names']), f"Keyword args should be {self.info['keyword_names']}"
        kwargs = {key: kwargs[key] for key in self.info['keyword_names']}

        # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors

        forward_inputs = (args, kwargs)

        if not nested_compare(forward_inputs, self.info['forward_schema']):
            raise TypeError(f"Inputs do not match expert input schema. Did you pass the right number of parameters?")

        flat_outputs = _RemoteModuleCall.apply(DUMMY, self.uid, self.stub, *nested_flatten(forward_inputs))
        # Note: we send DUMMY to prevent torch from excluding expert from backward if no other inputs require grad
        return nested_pack(flat_outputs, structure=self.info['outputs_schema'])

    @property
    def info(self):
        if self._info is None:
            outputs = self.stub.info(runtime_pb2.ExpertUID(uid=self.uid))
            self._info = pickle.loads(outputs.serialized_info)
        return self._info

    def extra_repr(self):
        return f"uid={self.uid}, endpoint={self.endpoint}"


class _RemoteModuleCall(torch.autograd.Function):
    """ Internal autograd-friendly call of a remote module. For applications, use RemoteExpert instead. """

    @staticmethod
    def forward(ctx, dummy: torch.Tensor, uid: str, stub: runtime_grpc.ConnectionHandlerStub,
                *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
        inputs = tuple(map(torch.Tensor.detach, inputs))  # detach to avoid pickling the computation graph
        ctx.uid, ctx.stub = uid, stub
        ctx.save_for_backward(*inputs)

        outputs = stub.forward(
            runtime_pb2.ExpertRequest(uid=ctx.uid, tensors=[serialize_torch_tensor(tensor) for tensor in inputs]))

        deserialized_outputs = [deserialize_torch_tensor(tensor) for tensor in outputs.tensors]

        return tuple(deserialized_outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        payload = tuple(nested_flatten((ctx.saved_tensors, grad_outputs)))

        grad_inputs = ctx.stub.backward(
            runtime_pb2.ExpertRequest(uid=ctx.uid, tensors=[serialize_torch_tensor(tensor) for tensor in payload]))

        deserialized_grad_inputs = [deserialize_torch_tensor(tensor) for tensor in grad_inputs.tensors]
        return (DUMMY, None, None, *deserialized_grad_inputs)
