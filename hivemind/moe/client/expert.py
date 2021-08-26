import asyncio
from concurrent.futures import Future
import multiprocessing as mp
import pickle
from typing import Any, Dict, Optional, Tuple, Awaitable
from threading import Thread
from queue import Queue

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable


#from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.utils.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.utils import nested_compare, nested_flatten, nested_pack, switch_to_uvloop
from hivemind.p2p import P2P, PeerInfo, StubBase
from hivemind.proto import runtime_pb2


DUMMY = torch.empty(0, requires_grad=True)  # dummy tensor that triggers autograd in RemoteExpert


import hivemind

#ConnectionHandlerStub = hivemind.moe.server.connection_handler.ConnectionHandler._stub_type


def _get_expert_stub(p2p: P2P, server_peer_info: PeerInfo): # -> ConnectionHandlerStub:
    return hivemind.moe.server.connection_handler.ConnectionHandler.get_stub(p2p, server_peer_info.peer_id)


class RemoteExpert(nn.Module):
    """
    A simple module that runs forward/backward of an expert hosted on a remote machine.
    Works seamlessly with pytorch autograd. (this is essentially a simple RPC function)
    Warning: RemoteExpert currently assumes that you provide it with correct input shapes.
    Sending wrong input shapes can cause RemoteExpert to freeze indefinitely due to error in runtime.
    :param uid: unique expert identifier
    """

    def __init__(self, uid, server_peer_info: PeerInfo, p2p: Optional[P2P] = None):
        super().__init__()
        self.uid, self.server_peer_info = uid, server_peer_info
        self._info = None

        if p2p is None:
            self.p2p = _RemoteModuleCall.run_coroutine(P2P.create())
        else:
            self.p2p = p2p

        _RemoteModuleCall.run_coroutine(self.p2p._client.connect(server_peer_info.peer_id, server_peer_info.addrs))

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
        flat_outputs = _RemoteModuleCall.apply(DUMMY, self.uid, self.stub, self.info, *nested_flatten(forward_inputs))

        # Note: we send DUMMY to prevent torch from excluding expert from backward if no other inputs require grad
        return nested_pack(flat_outputs, structure=self.info["outputs_schema"])

    @property
    def info(self):
        if self._info is None:
            outputs = _RemoteModuleCall.run_coroutine(self.stub.rpc_info(runtime_pb2.ExpertUID(uid=self.uid)))
            self._info = pickle.loads(outputs.serialized_info)
        return self._info

    def extra_repr(self):
        return f"uid={self.uid}, server_peer_info={self.server_peer_info}"


class _RemoteModuleCall(torch.autograd.Function):
    """Internal autograd-friendly call of a remote module. For applications, use RemoteExpert instead."""

    _task_queue: Queue = Queue()
    _event_thread: Optional[Thread] = None

    @classmethod
    def _run(cls):
        loop = switch_to_uvloop()

        async def receive_tasks():
            while True:
                cor, future = cls._task_queue.get()
                try:
                    result = await cor
                except Exception as e:
                    future.set_exception(e)
                    continue

                future.set_result(result)

        loop.run_until_complete(receive_tasks())

    @classmethod
    def run_coroutine(cls, coro: Awaitable, return_future: bool = False):
        if cls._event_thread is None:
            cls._event_thread = Thread(target=cls._run)
            cls._event_thread.start()

        future = Future()
        cls._task_queue.put((coro, future))

        if return_future:
            return future

        return future.result()

    @classmethod
    def forward(
        cls,
        ctx,
        dummy: torch.Tensor,
        uid: str,
        stub,#: ConnectionHandlerStub,
        info: Dict[str, Any],
        *inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
        # detach to avoid pickling the computation graph
        inputs = tuple(tensor.cpu().detach() for tensor in inputs)
        ctx.uid, ctx.stub, ctx.info = uid, stub, info
        ctx.save_for_backward(*inputs)

        serialized_tensors = [
            serialize_torch_tensor(inp, proto.compression)
            for inp, proto in zip(inputs, nested_flatten(info["forward_schema"]))
        ]

        outputs = cls.run_coroutine(
            stub.rpc_forward(runtime_pb2.ExpertRequest(uid=ctx.uid, tensors=serialized_tensors)),
        )

        deserialized_outputs = [deserialize_torch_tensor(tensor) for tensor in outputs.tensors]

        return tuple(deserialized_outputs)

    @classmethod
    @once_differentiable
    def backward(cls, ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        grad_outputs_cpu = tuple(tensor.cpu() for tensor in grad_outputs)
        inputs_and_grad_outputs = tuple(nested_flatten((ctx.saved_tensors, grad_outputs_cpu)))
        backward_schema = tuple(nested_flatten((ctx.info["forward_schema"], ctx.info["outputs_schema"])))
        serialized_tensors = [
            serialize_torch_tensor(tensor, proto.compression)
            for tensor, proto in zip(inputs_and_grad_outputs, backward_schema)
        ]

        grad_inputs = cls.run_coroutine(
            ctx.stub.rpc_backward(runtime_pb2.ExpertRequest(uid=ctx.uid, tensors=serialized_tensors)),
        )

        deserialized_grad_inputs = [deserialize_torch_tensor(tensor) for tensor in grad_inputs.tensors]
        return (DUMMY, None, None, None, *deserialized_grad_inputs)
