from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest
import torch

import hivemind
from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.moe.server.expert_backend import ExpertBackend
from hivemind.moe.server.task_pool import TaskPool
from hivemind.p2p.p2p_daemon_bindings.control import P2PHandlerError
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import amap_in_executor, iter_as_aiter
from hivemind.utils.serializer import MSGPackSerializer
from hivemind.utils.streaming import combine_and_deserialize_from_streaming, split_for_streaming
from hivemind.utils.tensor_descr import BatchTensorDescriptor


@pytest.mark.forked
@pytest.mark.asyncio
async def test_connection_handler():
    handler = ConnectionHandler(
        hivemind.DHT(start=True),
        dict(expert1=DummyExpertBackend("expert1", k=1), expert2=DummyExpertBackend("expert2", k=2)),
    )
    handler.start()

    client_dht = hivemind.DHT(start=True, client_mode=True, initial_peers=handler.dht.get_visible_maddrs())
    client_stub = ConnectionHandler.get_stub(await client_dht.replicate_p2p(), handler.dht.peer_id)

    inputs = torch.randn(1, 2)
    inputs_long = torch.randn(2**21, 2)

    # forward unary
    response = await client_stub.rpc_forward(
        runtime_pb2.ExpertRequest(uid="expert1", tensors=[serialize_torch_tensor(inputs)])
    )
    outputs = deserialize_torch_tensor(response.tensors[0])
    assert len(response.tensors) == 1
    assert torch.allclose(outputs, inputs * 1)

    response = await client_stub.rpc_forward(
        runtime_pb2.ExpertRequest(uid="expert2", tensors=[serialize_torch_tensor(inputs)])
    )
    outputs = deserialize_torch_tensor(response.tensors[0])
    assert len(response.tensors) == 1
    assert torch.allclose(outputs, inputs * 2)

    # forward streaming
    split = (
        p for t in [serialize_torch_tensor(inputs_long)] for p in split_for_streaming(t, chunk_size_bytes=2**16)
    )
    output_generator = await client_stub.rpc_forward_stream(
        amap_in_executor(
            lambda tensor_part: runtime_pb2.ExpertRequest(uid="expert2", tensors=[tensor_part]),
            iter_as_aiter(split),
        ),
    )
    outputs_list = [part async for part in output_generator]
    del output_generator
    assert len(outputs_list) == 8  # message size divided by DEFAULT_MAX_MSG_SIZE

    results = await combine_and_deserialize_from_streaming(
        amap_in_executor(lambda r: r.tensors, iter_as_aiter(outputs_list)), deserialize_torch_tensor
    )
    assert len(results) == 1
    assert torch.allclose(results[0], inputs_long * 2)

    # forward errors
    with pytest.raises(P2PHandlerError):
        # no such expert: fails with P2PHandlerError KeyError('expert3')
        await client_stub.rpc_forward(
            runtime_pb2.ExpertRequest(uid="expert3", tensors=[serialize_torch_tensor(inputs)])
        )

    with pytest.raises(P2PHandlerError):
        # bad input shape: P2PHandlerError("AssertionError") raised by DummyPool.submit_task
        await client_stub.rpc_forward(
            runtime_pb2.ExpertRequest(uid="expert1", tensors=[serialize_torch_tensor(torch.arange(5))])
        )

    # backward unary
    response = await client_stub.rpc_backward(
        runtime_pb2.ExpertRequest(
            uid="expert2", tensors=[serialize_torch_tensor(inputs * -1), serialize_torch_tensor(inputs)]
        )
    )
    outputs = deserialize_torch_tensor(response.tensors[0])
    assert len(response.tensors) == 1
    assert torch.allclose(outputs, inputs * -2)

    # backward streaming
    split = (
        p
        for t in [serialize_torch_tensor(inputs_long * 3), serialize_torch_tensor(inputs_long * 0)]
        for p in split_for_streaming(t, chunk_size_bytes=2**16)
    )
    output_generator = await client_stub.rpc_backward_stream(
        amap_in_executor(
            lambda tensor_part: runtime_pb2.ExpertRequest(uid="expert1", tensors=[tensor_part]),
            iter_as_aiter(split),
        ),
    )
    results = await combine_and_deserialize_from_streaming(
        amap_in_executor(lambda r: r.tensors, output_generator), deserialize_torch_tensor
    )
    assert len(results) == 1
    assert torch.allclose(results[0], inputs_long * 3)

    # backward errors
    with pytest.raises(P2PHandlerError):
        # bad input schema: fails with P2PHandlerError IndexError('tuple index out of range')
        await client_stub.rpc_backward(runtime_pb2.ExpertRequest(uid="expert2", tensors=[]))

    with pytest.raises(P2PHandlerError):
        # backward fails: empty stream
        output_generator = await client_stub.rpc_backward_stream(
            amap_in_executor(
                lambda tensor_part: runtime_pb2.ExpertRequest(uid="expert2", tensors=[tensor_part]),
                iter_as_aiter([]),
            ),
        )
        results = await combine_and_deserialize_from_streaming(
            amap_in_executor(lambda r: r.tensors, output_generator), deserialize_torch_tensor
        )
        assert len(results) == 1
        assert torch.allclose(results[0], inputs_long * 3)

    # check that handler did not crash after failed request
    await client_stub.rpc_forward(runtime_pb2.ExpertRequest(uid="expert1", tensors=[serialize_torch_tensor(inputs)]))

    # info
    response = await client_stub.rpc_info(runtime_pb2.ExpertUID(uid="expert1"))
    assert MSGPackSerializer.loads(response.serialized_info) == dict(name="expert1")

    response = await client_stub.rpc_info(runtime_pb2.ExpertUID(uid="expert2"))
    assert MSGPackSerializer.loads(response.serialized_info) == dict(name="expert2")

    with pytest.raises(P2PHandlerError):
        await client_stub.rpc_info(runtime_pb2.ExpertUID(uid="expert999"))

    handler.terminate()
    handler.join()


class DummyPool(TaskPool):
    def __init__(self, k: float):
        self.k = k

    async def submit_task(self, *inputs: torch.Tensor):
        await asyncio.sleep(0.01)
        print(type(inputs), inputs[0].shape)
        assert inputs[0].shape[-1] == 2
        return [inputs[0] * self.k]


class DummyExpertBackend(ExpertBackend):
    def __init__(self, name: str, k: float):
        self.name = name
        self.outputs_schema = [BatchTensorDescriptor.from_tensor(torch.randn(1, 2))]
        self.grad_inputs_schema = [BatchTensorDescriptor.from_tensor(torch.randn(1, 2))]
        self.forward_pool = DummyPool(k)
        self.backward_pool = DummyPool(k)

    def get_info(self) -> Dict[str, Any]:
        """Get expert parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(name=self.name)
