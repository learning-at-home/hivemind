import asyncio
import pickle
from typing import Any, Dict, Optional, Tuple, Type
from threading import Thread

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable


from hivemind.utils.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.utils import nested_compare, nested_flatten, nested_pack
from hivemind.p2p import P2P, PeerInfo, StubBase
from hivemind.proto import runtime_pb2
from hivemind.moe.server import ConnectionHandler


DUMMY = torch.empty(0, requires_grad=True)  # dummy tensor that triggers autograd in RemoteExpert


ConnectionHandlerStub = ConnectionHandler._stub_type


def _get_expert_stub(p2p: P2P, server_peer_info: PeerInfo) -> ConnectionHandlerStub:
    return ConnectionHandler.get_stub(p2p, server_peer_info.peer_id)


class RemoteExpert(nn.Module):
    """
    A simple module that runs forward/backward of an expert hosted on a remote machine.
    Works seamlessly with pytorch autograd. (this is essentially a simple RPC function)
    Warning: RemoteExpert currently assumes that you provide it with correct input shapes.
    Sending wrong input shapes can cause RemoteExpert to freeze indefinitely due to error in runtime.
    :param uid: unique expert identifier
    :param endpoint: network endpoint of a server that services that expert, e.g. "201.123.321.99:1337" or "[::]:8080"
    """

    def __init__(self, uid, p2p: P2P, server_peer_info: PeerInfo):
        super().__init__()
        self.uid, self.p2p, self.server_peer_info = uid, p2p, server_peer_info
        self._info = None

        self.loop = asyncio.new_event_loop()

        def _run(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        Thread(target=_run, args=(self.loop,)).start()

    @property
    def stub(self) -> StubBase:
        return _get_expert_stub(self.p2p, self.server_peer_info)

    def forward(self, *args, **kwargs):
        """Call RemoteExpert for the specified inputs and return its output(s). Compatible with pytorch.autograd."""
        assert len(kwargs) == len(self.info["keyword_names"]), f"Keyword args should be {self.info['keyword_names']}"
        kwargs = {key: kwargs[key] for key in self.info["keyword_names"]}

        # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors

        forward_inputs = (args, kwargs)

        if not nested_compare(forward_inputs, self.info["forward_schema"]):
            raise TypeError(f"Inputs do not match expert input schema. Did you pass the right number of parameters?")

        flat_outputs = _RemoteModuleCall.apply(
            DUMMY,
            self.uid,
            self.stub,
            self.loop,
            self.info,
            *nested_flatten(forward_inputs),
        )

        # Note: we send DUMMY to prevent torch from excluding expert from backward if no other inputs require grad
        return nested_pack(flat_outputs, structure=self.info["outputs_schema"])

    @property
    def info(self):
        if self._info is None:
            outputs = asyncio.run_coroutine_threadsafe(
                self.stub.rpc_info(runtime_pb2.ExpertUID(uid=self.uid)),
                self.loop
            ).result()
            self._info = pickle.loads(outputs.serialized_info)
        return self._info

    def extra_repr(self):
        return f"uid={self.uid}, endpoint={self.endpoint}"


class _RemoteModuleCall(torch.autograd.Function):
    """Internal autograd-friendly call of a remote module. For applications, use RemoteExpert instead."""

    @staticmethod
    def forward(
        ctx,
        dummy: torch.Tensor,
        uid: str,
        stub: ConnectionHandlerStub,
        loop: asyncio.AbstractEventLoop,
        info: Dict[str, Any],
        *inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
        # detach to avoid pickling the computation graph
        inputs = tuple(tensor.cpu().detach() for tensor in inputs)
        ctx.uid, ctx.stub, ctx.info, ctx.loop = uid, stub, info, loop
        ctx.save_for_backward(*inputs)

        serialized_tensors = [
            serialize_torch_tensor(inp, proto.compression)
            for inp, proto in zip(inputs, nested_flatten(info["forward_schema"]))
        ]

        outputs = asyncio.run_coroutine_threadsafe(
            stub.rpc_forward(runtime_pb2.ExpertRequest(uid=ctx.uid, tensors=serialized_tensors)),
            loop,
        ).result()

        deserialized_outputs = [deserialize_torch_tensor(tensor) for tensor in outputs.tensors]

        return tuple(deserialized_outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        grad_outputs_cpu = tuple(tensor.cpu() for tensor in grad_outputs)
        inputs_and_grad_outputs = tuple(nested_flatten((ctx.saved_tensors, grad_outputs_cpu)))
        backward_schema = tuple(nested_flatten((ctx.info["forward_schema"], ctx.info["outputs_schema"])))
        serialized_tensors = [
            serialize_torch_tensor(tensor, proto.compression)
            for tensor, proto in zip(inputs_and_grad_outputs, backward_schema)
        ]

        grad_inputs = asyncio.run_coroutine_threadsafe(
            ctx.stub.rpc_backward(runtime_pb2.ExpertRequest(uid=ctx.uid, tensors=serialized_tensors)),
            ctx.loop,
        ).result()

        deserialized_grad_inputs = [deserialize_torch_tensor(tensor) for tensor in grad_inputs.tensors]
        return (DUMMY, None, None, None, *deserialized_grad_inputs)
