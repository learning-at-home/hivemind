import os
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any, Awaitable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

import hivemind
from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.dht import DHT
from hivemind.p2p import P2P, PeerInfo, StubBase
from hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils import (
    MSGPackSerializer,
    amap_in_executor,
    as_aiter,
    nested_compare,
    nested_flatten,
    nested_pack,
    switch_to_uvloop,
)
from hivemind.utils.grpc import gather_from_rpc, split_for_streaming
from hivemind.utils.mpfuture import MPFuture

DUMMY = torch.empty(0, requires_grad=True)  # dummy tensor that triggers autograd in RemoteExpert


def _get_expert_stub(p2p: P2P, server_peer_info: PeerInfo):  # -> ConnectionHandlerStub:
    return hivemind.moe.server.connection_handler.ConnectionHandler.get_stub(p2p, server_peer_info.peer_id)


@dataclass(frozen=True)
class RemoteExpertInfo:
    uid: str
    peer_info: PeerInfo


class RemoteExpert(nn.Module):
    """
    A simple module that runs forward/backward of an expert hosted on a remote machine.
    Works seamlessly with pytorch autograd. (this is essentially a simple RPC function)
    Warning: RemoteExpert currently assumes that you provide it with correct input shapes.
    Sending wrong input shapes can cause RemoteExpert to freeze indefinitely due to error in runtime.
    :param uid: unique expert identifier
    """

    def __init__(self, expert_info: RemoteExpertInfo, p2p: P2P):
        super().__init__()
        self._info, self.p2p = expert_info, p2p
        self._rpc_info = None

    @property
    def uid(self):
        return self._info.uid

    @property
    def server_peer_info(self):
        return self._info.peer_info

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
        if self._rpc_info is None:
            outputs = RemoteExpertWorker.run_coroutine(self.stub.rpc_info(runtime_pb2.ExpertUID(uid=self.uid)))
            self._rpc_info = MSGPackSerializer.loads(outputs.serialized_info)
        return self._rpc_info

    def extra_repr(self):
        return f"uid={self.uid}, server_peer_info={self.server_peer_info}"


class RemoteExpertWorker:
    """Local thread for managing async tasks related to RemoteExpert"""

    _task_queue: Queue = Queue()
    _event_thread: Optional[Thread] = None
    _pid: int = -1

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
                if not future.cancelled():
                    future.set_result(result)

        loop.run_until_complete(receive_tasks())

    @classmethod
    def run_coroutine(cls, coro: Awaitable, return_future: bool = False):
        if cls._event_thread is None or cls._pid != os.getpid():
            cls._pid = os.getpid()
            cls._event_thread = Thread(target=cls._run, daemon=True)
            cls._event_thread.start()

        future = Future()
        cls._task_queue.put((coro, future))

        if return_future:
            return future

        result = future.result()
        return result

    @classmethod
    def _spawn_experts(cls, infos: Sequence[Optional[RemoteExpertInfo]], p2p: P2P) -> List[Optional[RemoteExpert]]:
        experts: List[Optional[RemoteExpert]] = []
        for i in infos:
            if i is not None:
                experts.append(RemoteExpert(i, p2p))
            else:
                experts.append(None)
        return experts

    @classmethod
    def spawn_experts(cls, infos: Sequence[Optional[RemoteExpertInfo]], dht: DHT) -> List[Optional[RemoteExpert]]:
        p2p = cls.run_coroutine(dht.replicate_p2p())
        return cls._spawn_experts(infos, p2p)

    @classmethod
    def spawn_experts_future(
        cls, infos: MPFuture[Sequence[Optional[RemoteExpertInfo]]], dht: DHT
    ) -> Future[List[Optional[RemoteExpert]]]:
        async def _unpack():
            p2p = cls.run_coroutine(dht.replicate_p2p(), True)
            return cls.spawn_experts(await infos, await p2p)

        return cls.run_coroutine(_unpack, True)

    @classmethod
    def spawn_experts_bulk(
        cls, infos: Sequence[Sequence[Optional[RemoteExpertInfo]]], dht: DHT
    ) -> List[List[Optional[RemoteExpert]]]:
        return [cls.spawn_experts(exps, dht) for exps in infos]

    @classmethod
    def spawn_experts_bulk_future(
        cls, infos: Future[Sequence[Sequence[Optional[RemoteExpertInfo]]]], dht: DHT
    ) -> MPFuture[List[List[Optional[RemoteExpert]]]]:
        async def _unpack():
            return cls.spawn_experts_bulk(await infos, dht)

        return cls.run_coroutine(_unpack, True)


async def _backward_stream(uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub) -> List[torch.Tensor]:
    split = tuple(p for t in serialized_tensors for p in split_for_streaming(t, DEFAULT_MAX_MSG_SIZE // 2))

    grad_inputs = await stub.rpc_backward_stream(
        amap_in_executor(
            lambda t: runtime_pb2.ExpertRequest(uid=uid, tensors=[t]),
            as_aiter(*split),
        ),
    )

    return await gather_from_rpc(grad_inputs, lambda r: r.tensors, deserialize_torch_tensor)


async def _backward(uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub) -> List[torch.Tensor]:
    grad_inputs: runtime_pb2.ExpertResponse = await stub.rpc_backward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors))
    )
    return [deserialize_torch_tensor(t) for t in grad_inputs.tensors]


async def expert_backward(
    uid: str, inputs_and_grads: Sequence[torch.Tensor], compressions: Iterable, stub
) -> List[torch.Tensor]:
    serialized_tensors = (
        serialize_torch_tensor(tensor, compression) for tensor, compression in zip(inputs_and_grads, compressions)
    )

    size = 0
    for t in inputs_and_grads:
        size += t.element_size() * t.nelement()
        if size >= DEFAULT_MAX_MSG_SIZE:
            return await _backward_stream(uid, serialized_tensors, stub)
    else:
        return await _backward(uid, serialized_tensors, stub)


async def _forward_stream(uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub) -> List[torch.Tensor]:
    split = tuple(p for t in serialized_tensors for p in split_for_streaming(t, DEFAULT_MAX_MSG_SIZE // 2))

    outputs = await stub.rpc_forward_stream(
        amap_in_executor(
            lambda t: runtime_pb2.ExpertRequest(uid=uid, tensors=[t]),
            as_aiter(*split),
        ),
    )

    return await gather_from_rpc(outputs, lambda r: r.tensors, deserialize_torch_tensor)


async def _forward(uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub) -> List[torch.Tensor]:
    outputs: runtime_pb2.ExpertResponse = await stub.rpc_forward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors))
    )
    return [deserialize_torch_tensor(t) for t in outputs.tensors]


async def expert_forward(uid: str, inputs: Sequence[torch.Tensor], compressions: Iterable, stub) -> List[torch.Tensor]:
    serialized_tensors = (
        serialize_torch_tensor(tensor, compression) for tensor, compression in zip(inputs, compressions)
    )
    size = 0
    for t in inputs:
        size += t.element_size() * t.nelement()
        if size >= DEFAULT_MAX_MSG_SIZE:
            return await _forward_stream(uid, serialized_tensors, stub)
    else:
        return await _forward(uid, serialized_tensors, stub)


class _RemoteModuleCall(torch.autograd.Function):
    """Internal autograd-friendly call of a remote module. For applications, use RemoteExpert instead."""

    @classmethod
    def forward(
        cls,
        ctx,
        dummy: torch.Tensor,
        uid: str,
        stub,  #: ConnectionHandlerStub,
        info: Dict[str, Any],
        *inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
        # detach to avoid pickling the computation graph
        inputs = tuple(tensor.cpu().detach() for tensor in inputs)
        ctx.uid, ctx.stub, ctx.info = uid, stub, info
        ctx.save_for_backward(*inputs)

        deserialized_outputs = RemoteExpertWorker.run_coroutine(
            expert_forward(uid, inputs, (p.compression for p in nested_flatten(info["forward_schema"])), stub)
        )

        return tuple(deserialized_outputs)

    @classmethod
    @once_differentiable
    def backward(cls, ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        grad_outputs_cpu = tuple(tensor.cpu() for tensor in grad_outputs)
        inputs_and_grad_outputs = tuple(nested_flatten((ctx.saved_tensors, grad_outputs_cpu)))
        backward_schema = tuple(nested_flatten((ctx.info["forward_schema"], ctx.info["outputs_schema"])))

        deserialized_grad_inputs = RemoteExpertWorker.run_coroutine(
            expert_backward(ctx.uid, inputs_and_grad_outputs, (p.compression for p in backward_schema), ctx.stub)
        )

        return (DUMMY, None, None, None, *deserialized_grad_inputs)
