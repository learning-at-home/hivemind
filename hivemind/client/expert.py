from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from ..utils import nested_flatten, DUMMY, PytorchSerializer, nested_pack, nested_compare, Connection


class RemoteExpert(nn.Module):
    """
    A simple module that runs forward/backward of an expert hosted on a remote machine.
    Works seamlessly with pytorch autograd. (this is essentially a simple RPC function)

    Warning: RemoteExpert currently assumes that you provide it with correct input shapes.
    Sending wrong input shapes can cause RemoteExpert to freeze indefinitely due to error in runtime.

    :param uid: unique expert identifier
    :param host: hostname where Server operates
    :param port: port to which Server listens
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
            connection = Connection.create(self.host, self.port)
            connection.send_raw('info', PytorchSerializer.dumps(self.uid))
            self._info = PytorchSerializer.loads(connection.recv_message()[1])
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

        connection = Connection.create(ctx.host, ctx.port)
        connection.send_raw('fwd_', PytorchSerializer.dumps((ctx.uid, inputs)))
        rtype, msg = connection.recv_message()
        assert len(msg) != 0, "ExpertBackend.forward did not respond"
        *outputs, rng_state = tuple(PytorchSerializer.loads(msg))
        ctx.rng_state = rng_state
        return outputs  # flattened expert outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        connection = Connection.create(ctx.host, ctx.port)
        payload = tuple(nested_flatten((ctx.saved_tensors, grad_outputs, ctx.rng_state)))
        connection.send_raw('bwd_', PytorchSerializer.dumps((ctx.uid, payload)))
        rtype, msg = connection.recv_message()
        assert len(msg) != 0, "ExpertBackend.backward did not respond"
        grad_inputs = PytorchSerializer.loads(msg)
        return (DUMMY, None, None, None, *grad_inputs)
