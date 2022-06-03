from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest
import torch

from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.dht import DHT
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.moe.server.expert_backend import ExpertBackend
from hivemind.moe.server.task_pool import TaskPool
from hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE
from hivemind.p2p.p2p_daemon_bindings.control import P2PHandlerError
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import amap_in_executor, iter_as_aiter
from hivemind.utils.serializer import MSGPackSerializer
from hivemind.utils.streaming import combine_and_deserialize_from_streaming, split_for_streaming
from hivemind.utils.tensor_descr import BatchTensorDescriptor

LONG_INPUT_SIZE = 2**21


class DummyPool(TaskPool):
    def __init__(self, k: float):
        self.k = k

    async def submit_task(self, *inputs: torch.Tensor):
        await asyncio.sleep(0.01)
        if inputs[0].shape[-1] != 2:
            raise ValueError("wrong input shape")
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


@pytest.fixture()
async def stub():
    server_dht = DHT(start=True)
    experts = {
        "expert1": DummyExpertBackend("expert1", k=1),
        "expert2": DummyExpertBackend("expert2", k=2),
    }
    handler = ConnectionHandler(server_dht, experts)
    handler.start()

    client_dht = DHT(start=True, client_mode=True, initial_peers=server_dht.get_visible_maddrs())
    p2p = await client_dht.replicate_p2p()
    client_stub = ConnectionHandler.get_stub(p2p, server_dht.peer_id)
    yield client_stub

    handler.terminate()
    handler.join()


@pytest.fixture
def small_input():
    return torch.randn(1, 2)


@pytest.fixture
def long_input():
    input = torch.randn(LONG_INPUT_SIZE, 2)
    n_chunks = (input.nelement() * input.element_size() + DEFAULT_MAX_MSG_SIZE - 1) // DEFAULT_MAX_MSG_SIZE
    return input, n_chunks


@pytest.mark.forked
@pytest.mark.asyncio
async def test_forward_unary(stub, small_input):
    response = await stub.rpc_forward(
        runtime_pb2.ExpertRequest(uid="expert1", tensors=[serialize_torch_tensor(small_input)])
    )
    outputs = deserialize_torch_tensor(response.tensors[0])
    assert len(response.tensors) == 1
    assert torch.allclose(outputs, small_input * 1)

    response = await stub.rpc_forward(
        runtime_pb2.ExpertRequest(uid="expert2", tensors=[serialize_torch_tensor(small_input)])
    )
    outputs = deserialize_torch_tensor(response.tensors[0])
    assert len(response.tensors) == 1
    assert torch.allclose(outputs, small_input * 2)


@pytest.mark.forked
@pytest.mark.asyncio
async def test_forward_streaming(stub, long_input):
    input, n_chunks = long_input
    split = (
        p
        for t in [serialize_torch_tensor(input)]
        for p in split_for_streaming(t, chunk_size_bytes=DEFAULT_MAX_MSG_SIZE)
    )
    output_generator = await stub.rpc_forward_stream(
        amap_in_executor(
            lambda tensor_part: runtime_pb2.ExpertRequest(uid="expert2", tensors=[tensor_part]),
            iter_as_aiter(split),
        ),
    )
    outputs_list = [part async for part in output_generator]
    del output_generator
    assert len(outputs_list) == n_chunks

    results = await combine_and_deserialize_from_streaming(
        amap_in_executor(lambda r: r.tensors, iter_as_aiter(outputs_list)), deserialize_torch_tensor
    )
    assert len(results) == 1
    assert torch.allclose(results[0], input * 2)


@pytest.mark.forked
@pytest.mark.asyncio
async def test_forward_errors(stub, small_input):
    # no such expert: fails with P2PHandlerError KeyError('expert3')
    with pytest.raises(P2PHandlerError):
        await stub.rpc_forward(runtime_pb2.ExpertRequest(uid="expert3", tensors=[serialize_torch_tensor(small_input)]))

    # bad input shape: P2PHandlerError("AssertionError") raised by DummyPool.submit_task
    with pytest.raises(P2PHandlerError):
        await stub.rpc_forward(
            runtime_pb2.ExpertRequest(uid="expert1", tensors=[serialize_torch_tensor(torch.arange(5))])
        )


@pytest.mark.forked
@pytest.mark.asyncio
async def test_backward_unary(stub, small_input):
    response = await stub.rpc_backward(
        runtime_pb2.ExpertRequest(
            uid="expert2", tensors=[serialize_torch_tensor(small_input * -1), serialize_torch_tensor(small_input)]
        )
    )
    outputs = deserialize_torch_tensor(response.tensors[0])
    assert len(response.tensors) == 1
    assert torch.allclose(outputs, small_input * -2)


@pytest.mark.forked
@pytest.mark.asyncio
async def test_backward_streaming(stub, long_input):
    input, _ = long_input
    split = (
        p
        for t in [serialize_torch_tensor(input * 3), serialize_torch_tensor(input * 0)]
        for p in split_for_streaming(t, chunk_size_bytes=DEFAULT_MAX_MSG_SIZE)
    )
    output_generator = await stub.rpc_backward_stream(
        amap_in_executor(
            lambda tensor_part: runtime_pb2.ExpertRequest(uid="expert1", tensors=[tensor_part]),
            iter_as_aiter(split),
        ),
    )
    results = await combine_and_deserialize_from_streaming(
        amap_in_executor(lambda r: r.tensors, output_generator), deserialize_torch_tensor
    )
    assert len(results) == 1
    assert torch.allclose(results[0], input * 3)


@pytest.mark.forked
@pytest.mark.asyncio
async def test_backward_errors(stub, small_input, long_input):
    long, _ = long_input
    # bad input schema: fails with P2PHandlerError IndexError('tuple index out of range')
    with pytest.raises(P2PHandlerError):
        await stub.rpc_backward(runtime_pb2.ExpertRequest(uid="expert2", tensors=[]))

    # backward fails: empty stream
    with pytest.raises(P2PHandlerError):
        output_generator = await stub.rpc_backward_stream(
            amap_in_executor(
                lambda tensor_part: runtime_pb2.ExpertRequest(uid="expert2", tensors=[tensor_part]),
                iter_as_aiter([]),
            ),
        )
        results = await combine_and_deserialize_from_streaming(
            amap_in_executor(lambda r: r.tensors, output_generator), deserialize_torch_tensor
        )
        assert len(results) == 1
        assert torch.allclose(results[0], long * 3)

    # check that handler did not crash after failed request
    await stub.rpc_forward(runtime_pb2.ExpertRequest(uid="expert1", tensors=[serialize_torch_tensor(small_input)]))


@pytest.mark.forked
@pytest.mark.asyncio
async def test_info(stub):
    response = await stub.rpc_info(runtime_pb2.ExpertUID(uid="expert1"))
    assert MSGPackSerializer.loads(response.serialized_info) == dict(name="expert1")

    response = await stub.rpc_info(runtime_pb2.ExpertUID(uid="expert2"))
    assert MSGPackSerializer.loads(response.serialized_info) == dict(name="expert2")

    with pytest.raises(P2PHandlerError):
        await stub.rpc_info(runtime_pb2.ExpertUID(uid="expert999"))
