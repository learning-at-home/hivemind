import asyncio
import multiprocessing as mp
import subprocess
from functools import partial
from typing import List

import numpy as np
import pytest
import torch

from hivemind.p2p import P2P
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID
from hivemind.proto import dht_pb2, runtime_pb2
from hivemind.utils import MSGPackSerializer
from hivemind.utils.compression import deserialize_torch_tensor, serialize_torch_tensor


def is_process_running(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


async def replicate_if_needed(p2p: P2P, replicate: bool):
    return await P2P.replicate(p2p._daemon_listen_port, p2p._host_port) if replicate else p2p


def bootstrap_addr(host_port, id_):
    return f'/ip4/127.0.0.1/tcp/{host_port}/p2p/{id_}'


def bootstrap_from(daemons: List[P2P]) -> List[str]:
    return [bootstrap_addr(d._host_port, d.id) for d in daemons]


@pytest.mark.asyncio
async def test_daemon_killed_on_del():
    p2p_daemon = await P2P.create()

    child_pid = p2p_daemon._child.pid
    assert is_process_running(child_pid)

    await p2p_daemon.shutdown()
    assert not is_process_running(child_pid)


@pytest.mark.asyncio
async def test_server_client_connection():
    server = await P2P.create()
    peers = await server._client.list_peers()
    assert len(peers) == 0

    nodes = bootstrap_from([server])
    client = await P2P.create(bootstrap=True, bootstrap_peers=nodes)
    await client.wait_for_at_least_n_peers(1)

    peers = await client._client.list_peers()
    assert len(peers) == 1
    peers = await server._client.list_peers()
    assert len(peers) == 1


@pytest.mark.asyncio
async def test_daemon_replica_does_not_affect_primary():
    p2p_daemon = await P2P.create()
    p2p_replica = await P2P.replicate(p2p_daemon._daemon_listen_port, p2p_daemon._host_port)

    child_pid = p2p_daemon._child.pid
    assert is_process_running(child_pid)

    await p2p_replica.shutdown()
    assert is_process_running(child_pid)

    await p2p_daemon.shutdown()
    assert not is_process_running(child_pid)


def handle_square(x):
    x = MSGPackSerializer.loads(x)
    return MSGPackSerializer.dumps(x ** 2)


def handle_add(args):
    args = MSGPackSerializer.loads(args)
    result = args[0]
    for i in range(1, len(args)):
        result = result + args[i]
    return MSGPackSerializer.dumps(result)


def handle_square_torch(x):
    tensor = runtime_pb2.Tensor()
    tensor.ParseFromString(x)
    tensor = deserialize_torch_tensor(tensor)
    result = tensor ** 2
    return serialize_torch_tensor(result).SerializeToString()


def handle_add_torch(args):
    args = MSGPackSerializer.loads(args)
    tensor = runtime_pb2.Tensor()
    tensor.ParseFromString(args[0])
    result = deserialize_torch_tensor(tensor)

    for i in range(1, len(args)):
        tensor = runtime_pb2.Tensor()
        tensor.ParseFromString(args[i])
        result = result + deserialize_torch_tensor(tensor)

    return serialize_torch_tensor(result).SerializeToString()


def handle_add_torch_with_exc(args):
    try:
        return handle_add_torch(args)
    except Exception:
        return b'something went wrong :('


@pytest.mark.parametrize(
    'should_cancel,replicate', [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ]
)
@pytest.mark.asyncio
async def test_call_unary_handler(should_cancel, replicate, handle_name="handle"):
    handler_cancelled = False

    async def ping_handler(request, context):
        try:
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            nonlocal handler_cancelled
            handler_cancelled = True
        return dht_pb2.PingResponse(
            peer=dht_pb2.NodeInfo(
                node_id=context.id.encode(), rpc_port=context.port),
            sender_endpoint=context.handle_name, available=True)

    server_primary = await P2P.create()
    server = await replicate_if_needed(server_primary, replicate)
    server_pid = server_primary._child.pid
    await server.add_unary_handler(handle_name, ping_handler, dht_pb2.PingRequest,
                                   dht_pb2.PingResponse)
    assert is_process_running(server_pid)

    nodes = bootstrap_from([server])
    client_primary = await P2P.create(bootstrap=True, bootstrap_peers=nodes)
    client = await replicate_if_needed(client_primary, replicate)
    client_pid = client_primary._child.pid
    assert is_process_running(client_pid)

    ping_request = dht_pb2.PingRequest(
        peer=dht_pb2.NodeInfo(node_id=client.id.encode(), rpc_port=client._host_port),
        validate=True)
    expected_response = dht_pb2.PingResponse(
        peer=dht_pb2.NodeInfo(node_id=server.id.encode(), rpc_port=server._host_port),
        sender_endpoint=handle_name, available=True)

    await client.wait_for_at_least_n_peers(1)
    libp2p_server_id = PeerID.from_base58(server.id)
    stream_info, reader, writer = await client._client.stream_open(libp2p_server_id, (handle_name,))

    await P2P.send_protobuf(ping_request, dht_pb2.PingRequest, writer)

    if should_cancel:
        writer.close()
        await asyncio.sleep(1)
        assert handler_cancelled
    else:
        result, err = await P2P.receive_protobuf(dht_pb2.PingResponse, reader)
        assert err is None
        assert result == expected_response
        assert not handler_cancelled

    await server.stop_listening()
    await server_primary.shutdown()
    assert not is_process_running(server_pid)

    await client_primary.shutdown()
    assert not is_process_running(client_pid)


@pytest.mark.asyncio
async def test_call_unary_handler_error(handle_name="handle"):
    async def error_handler(request, context):
        raise ValueError('boom')

    server = await P2P.create()
    server_pid = server._child.pid
    await server.add_unary_handler(handle_name, error_handler, dht_pb2.PingRequest, dht_pb2.PingResponse)
    assert is_process_running(server_pid)

    nodes = bootstrap_from([server])
    client = await P2P.create(bootstrap=True, bootstrap_peers=nodes)
    client_pid = client._child.pid
    assert is_process_running(client_pid)
    await client.wait_for_at_least_n_peers(1)

    ping_request = dht_pb2.PingRequest(
        peer=dht_pb2.NodeInfo(node_id=client.id.encode(), rpc_port=client._host_port),
        validate=True)
    libp2p_server_id = PeerID.from_base58(server.id)
    stream_info, reader, writer = await client._client.stream_open(libp2p_server_id, (handle_name,))

    await P2P.send_protobuf(ping_request, dht_pb2.PingRequest, writer)
    result, err = await P2P.receive_protobuf(dht_pb2.PingResponse, reader)
    assert result is None
    assert err.message == 'boom'

    await server.stop_listening()
    await server.shutdown()
    await client.shutdown()


@pytest.mark.parametrize(
    "test_input,expected,handle",
    [
        pytest.param(10, 100, handle_square, id="square_integer"),
        pytest.param((1, 2), 3, handle_add, id="add_integers"),
        pytest.param(([1, 2, 3], [12, 13]), [1, 2, 3, 12, 13], handle_add, id="add_lists"),
        pytest.param(2, 8, lambda x: MSGPackSerializer.dumps(MSGPackSerializer.loads(x) ** 3), id="lambda")
    ]
)
@pytest.mark.asyncio
async def test_call_peer_single_process(test_input, expected, handle, handler_name="handle"):
    server = await P2P.create()
    server_pid = server._child.pid
    await server.add_stream_handler(handler_name, handle)
    assert is_process_running(server_pid)

    nodes = bootstrap_from([server])
    client = await P2P.create(bootstrap=True, bootstrap_peers=nodes)
    client_pid = client._child.pid
    assert is_process_running(client_pid)

    await client.wait_for_at_least_n_peers(1)

    test_input_msgp = MSGPackSerializer.dumps(test_input)
    result_msgp = await client.call_peer_handler(server.id, handler_name, test_input_msgp)
    result = MSGPackSerializer.loads(result_msgp)
    assert result == expected

    await server.stop_listening()
    await server.shutdown()
    assert not is_process_running(server_pid)

    await client.shutdown()
    assert not is_process_running(client_pid)


async def run_server(handler_name, server_side, client_side, response_received):
    server = await P2P.create()
    server_pid = server._child.pid
    await server.add_stream_handler(handler_name, handle_square)
    assert is_process_running(server_pid)

    server_side.send(server.id)
    server_side.send(server._host_port)
    while response_received.value == 0:
        await asyncio.sleep(0.5)

    await server.stop_listening()
    await server.shutdown()
    assert not is_process_running(server_pid)


def server_target(handler_name, server_side, client_side, response_received):
    asyncio.run(run_server(handler_name, server_side, client_side, response_received))


@pytest.mark.asyncio
async def test_call_peer_different_processes():
    handler_name = "square"
    test_input = 2

    server_side, client_side = mp.Pipe()
    response_received = mp.Value(np.ctypeslib.as_ctypes_type(np.int32))
    response_received.value = 0

    proc = mp.Process(target=server_target, args=(handler_name, server_side, client_side, response_received))
    proc.start()

    peer_id = client_side.recv()
    peer_port = client_side.recv()

    nodes = [bootstrap_addr(peer_port, peer_id)]
    client = await P2P.create(bootstrap=True, bootstrap_peers=nodes)
    client_pid = client._child.pid
    assert is_process_running(client_pid)

    await client.wait_for_at_least_n_peers(1)

    test_input_msgp = MSGPackSerializer.dumps(2)
    result_msgp = await client.call_peer_handler(peer_id, handler_name, test_input_msgp)
    result = MSGPackSerializer.loads(result_msgp)
    assert np.allclose(result, test_input ** 2)
    response_received.value = 1

    await client.shutdown()
    assert not is_process_running(client_pid)

    proc.join()


@pytest.mark.parametrize(
    "test_input,expected",
    [
        pytest.param(torch.tensor([2]), torch.tensor(4)),
        pytest.param(
            torch.tensor([[1.0, 2.0], [0.5, 0.1]]),
            torch.tensor([[1.0, 2.0], [0.5, 0.1]]) ** 2),
    ]
)
@pytest.mark.asyncio
async def test_call_peer_torch_square(test_input, expected, handler_name="handle"):
    handle = handle_square_torch
    server = await P2P.create()
    await server.add_stream_handler(handler_name, handle)

    nodes = bootstrap_from([server])
    client = await P2P.create(bootstrap=True, bootstrap_peers=nodes)

    await client.wait_for_at_least_n_peers(1)

    inp = serialize_torch_tensor(test_input).SerializeToString()
    result_pb = await client.call_peer_handler(server.id, handler_name, inp)
    result = runtime_pb2.Tensor()
    result.ParseFromString(result_pb)
    result = deserialize_torch_tensor(result)
    assert torch.allclose(result, expected)

    await server.stop_listening()
    await server.shutdown()
    await client.shutdown()


@pytest.mark.parametrize(
    "test_input,expected",
    [
        pytest.param([torch.tensor([1]), torch.tensor([2])], torch.tensor([3])),
        pytest.param(
            [torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.tensor([[1.1, 1.2], [1.3, 1.4]])],
            torch.tensor([[1.2, 1.4], [1.6, 1.8]])),
    ]
)
@pytest.mark.asyncio
async def test_call_peer_torch_add(test_input, expected, handler_name="handle"):
    handle = handle_add_torch
    server = await P2P.create()
    await server.add_stream_handler(handler_name, handle)

    nodes = bootstrap_from([server])
    client = await P2P.create(bootstrap=True, bootstrap_peers=nodes)

    await client.wait_for_at_least_n_peers(1)

    inp = [serialize_torch_tensor(i).SerializeToString() for i in test_input]
    inp_msgp = MSGPackSerializer.dumps(inp)
    result_pb = await client.call_peer_handler(server.id, handler_name, inp_msgp)
    result = runtime_pb2.Tensor()
    result.ParseFromString(result_pb)
    result = deserialize_torch_tensor(result)
    assert torch.allclose(result, expected)

    await server.stop_listening()
    await server.shutdown()
    await client.shutdown()


@pytest.mark.parametrize(
    "replicate",
    [
        pytest.param(False, id="primary"),
        pytest.param(True, id="replica"),
    ]
)
@pytest.mark.asyncio
async def test_call_peer_error(replicate, handler_name="handle"):
    server_primary = await P2P.create()
    server = await replicate_if_needed(server_primary, replicate)
    await server.add_stream_handler(handler_name, handle_add_torch_with_exc)

    nodes = bootstrap_from([server])
    client_primary = await P2P.create(bootstrap=True, bootstrap_peers=nodes)
    client = await replicate_if_needed(client_primary, replicate)

    await client.wait_for_at_least_n_peers(1)

    inp = [serialize_torch_tensor(i).SerializeToString() for i in [torch.zeros((2, 3)), torch.zeros((3, 2))]]
    inp_msgp = MSGPackSerializer.dumps(inp)
    result = await client.call_peer_handler(server.id, handler_name, inp_msgp)
    assert result == b'something went wrong :('

    await server.stop_listening()
    await server_primary.shutdown()
    await client_primary.shutdown()


@pytest.mark.asyncio
async def test_handlers_on_different_replicas(handler_name="handle"):
    def handler(arg, key):
        return key

    server_primary = await P2P.create(bootstrap=False)
    server_id = server_primary.id
    await server_primary.add_stream_handler(handler_name, partial(handler, key=b'primary'))

    server_replica1 = await replicate_if_needed(server_primary, True)
    await server_replica1.add_stream_handler(handler_name + '1', partial(handler, key=b'replica1'))

    server_replica2 = await replicate_if_needed(server_primary, True)
    await server_replica2.add_stream_handler(handler_name + '2', partial(handler, key=b'replica2'))

    nodes = bootstrap_from([server_primary])
    client = await P2P.create(bootstrap=True, bootstrap_peers=nodes)
    await client.wait_for_at_least_n_peers(1)

    result = await client.call_peer_handler(server_id, handler_name, b'1')
    assert result == b"primary"

    result = await client.call_peer_handler(server_id, handler_name + '1', b'2')
    assert result == b"replica1"

    result = await client.call_peer_handler(server_id, handler_name + '2', b'3')
    assert result == b"replica2"

    await server_replica1.stop_listening()
    await server_replica2.stop_listening()

    # Primary does not handle replicas protocols
    with pytest.raises(asyncio.IncompleteReadError):
        await client.call_peer_handler(server_id, handler_name + '1', b'')
    with pytest.raises(asyncio.IncompleteReadError):
        await client.call_peer_handler(server_id, handler_name + '2', b'')

    await server_primary.stop_listening()
    await server_primary.shutdown()
    await client.shutdown()
