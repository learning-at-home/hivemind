import os
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any, Awaitable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from multiaddr import Multiaddr
from torch.autograd.function import once_differentiable

import hivemind
from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.dht import DHT
from hivemind.p2p import P2P, PeerInfo, StubBase
from hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID
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
from hivemind.utils.grpc import gather_from_grpc, split_for_streaming
from hivemind.utils.mpfuture import MPFuture

DUMMY = torch.empty(0, requires_grad=True)  # dummy tensor that triggers autograd in RemoteExpert


def _get_expert_stub(p2p: P2P, server_peer_info: PeerInfo):  # -> ConnectionHandlerStub:
    return hivemind.moe.server.connection_handler.ConnectionHandler.get_stub(p2p, server_peer_info.peer_id)


@dataclass(frozen=True)
class RemoteExpertInfo:
    uid: str
    peer_id: str
    addrs: Sequence[str]

    @property
    def as_peer_info(self) -> Tuple[str, PeerInfo]:
        return self.uid, PeerInfo(
            peer_id=PeerID.from_base58(self.peer_id), addrs=tuple(Multiaddr(a) for a in self.addrs)
        )


class RemoteExpert(nn.Module):
    """
    A simple module that runs forward/backward of an expert hosted on a remote machine.
    Works seamlessly with pytorch autograd. (this is essentially a simple RPC function)
    Warning: RemoteExpert currently assumes that you provide it with correct input shapes.
    Sending wrong input shapes can cause RemoteExpert to freeze indefinitely due to error in runtime.
    :param uid: unique expert identifier
    """

    def __init__(self, uid, server_peer_info: PeerInfo, p2p: P2P):
        super().__init__()
        self.uid, self.server_peer_info, self.p2p = uid, server_peer_info, p2p
        self._info = None

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
            outputs = RemoteExpertWorker.run_coroutine(self.stub.rpc_info(runtime_pb2.ExpertUID(uid=self.uid)))
            self._info = MSGPackSerializer.loads(outputs.serialized_info)
        return self._info

    def extra_repr(self):
        return f"uid={self.uid}, server_peer_info={self.server_peer_info}"


class RemoteExpertWorker:
    """Local thread for managing async tasks related to RemoteExpert"""

    _task_queue: Queue = Queue()
    _event_thread: Optional[Thread] = None
    _pid: int = 0

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
                uid, peer_info = i.as_peer_info
                experts.append(RemoteExpert(uid, peer_info, p2p))
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
    ) -> MPFuture[List[Optional[RemoteExpert]]]:
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
        cls, infos: MPFuture[Sequence[Sequence[Optional[RemoteExpertInfo]]]], dht: DHT
    ) -> MPFuture[List[List[Optional[RemoteExpert]]]]:
        async def _unpack():
            return cls.spawn_experts_bulk(await infos, dht)

        return cls.run_coroutine(_unpack, True)



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

        serialized_tensors = [
            serialize_torch_tensor(inp, proto.compression)
            for inp, proto in zip(inputs, nested_flatten(info["forward_schema"]))
        ]

        size = 0
        for t in inputs:
            size += t.element_size() * t.nelement()
            if size >= DEFAULT_MAX_MSG_SIZE:
                deserialized_outputs = cls.forward_partial(serialized_tensors, ctx, stub)
                break
        else:
            deserialized_outputs = cls.forward_oneshot(serialized_tensors, ctx, stub)

        return tuple(deserialized_outputs)

    @classmethod
    def forward_partial(cls, serialized_tensors: List[runtime_pb2.Tensor], ctx, stub) -> List[torch.Tensor]:
        split = [p for t in serialized_tensors for p in split_for_streaming(t, DEFAULT_MAX_MSG_SIZE // 2)]

        outputs = RemoteExpertWorker.run_coroutine(
            stub.rpc_forward_partial(
                amap_in_executor(
                    lambda t: runtime_pb2.ExpertRequest(
                        uid=ctx.uid,
                        tensors=[
                            t,
                        ],
                    ),
                    as_aiter(*split),
                ),
            )
        )

        return RemoteExpertWorker.run_coroutine(
            gather_from_grpc(outputs, lambda r: r.tensors, deserialize_torch_tensor)
        )

    @classmethod
    def forward_oneshot(cls, serialized_tensors: List[runtime_pb2.Tensor], ctx, stub) -> List[torch.Tensor]:

        outputs = RemoteExpertWorker.run_coroutine(
            stub.rpc_forward(runtime_pb2.ExpertRequest(uid=ctx.uid, tensors=serialized_tensors))
        )

        return [deserialize_torch_tensor(t) for t in outputs.tensors]

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

        size = 0
        for t in inputs_and_grad_outputs:
            size += t.element_size() * t.nelement()
            if size >= DEFAULT_MAX_MSG_SIZE:
                deserialized_grad_inputs = cls.backward_partial(serialized_tensors, ctx)
                break
        else:
            deserialized_grad_inputs = cls.backward_oneshot(serialized_tensors, ctx)

        return (DUMMY, None, None, None, *deserialized_grad_inputs)

    @classmethod
    @once_differentiable
    def backward_partial(cls, serialized_tensors: List[runtime_pb2.Tensor], ctx) -> List[torch.Tensor]:
        split = tuple(p for t in serialized_tensors for p in split_for_streaming(t, DEFAULT_MAX_MSG_SIZE // 2))

        grad_inputs = RemoteExpertWorker.run_coroutine(
            ctx.stub.rpc_backward_partial(
                amap_in_executor(
                    lambda t: runtime_pb2.ExpertRequest(
                        uid=ctx.uid,
                        tensors=[
                            t,
                        ],
                    ),
                    as_aiter(*split),
                ),
            )
        )

        return RemoteExpertWorker.run_coroutine(
            gather_from_grpc(grad_inputs, lambda r: r.tensors, deserialize_torch_tensor)
        )

    @classmethod
    @once_differentiable
    def backward_oneshot(cls, serialized_tensors: List[runtime_pb2.Tensor], ctx) -> List[torch.Tensor]:
        grad_inputs = RemoteExpertWorker.run_coroutine(
            ctx.stub.rpc_backward(runtime_pb2.ExpertRequest(uid=ctx.uid, tensors=serialized_tensors))
        )

        return [deserialize_torch_tensor(t) for t in grad_inputs.tensors]
