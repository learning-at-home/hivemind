import asyncio
import multiprocessing as mp
import subprocess

import numpy as np
import pytest

import hivemind.p2p
from hivemind.p2p import P2P

RUNNING = 'running'
NOT_RUNNING = 'not running'
CHECK_PID_CMD = '''
if ps -p {0} > /dev/null;
then
    echo "{1}"
else
    echo "{2}"
fi
'''


def is_process_running(pid: int) -> bool:
    cmd = CHECK_PID_CMD.format(pid, RUNNING, NOT_RUNNING)
    return subprocess.check_output(cmd, shell=True).decode('utf-8').strip() == RUNNING


@pytest.mark.asyncio
async def test_daemon_killed_on_del():
    p2p_daemon = await P2P.create()

    child_pid = p2p_daemon._child.pid
    assert is_process_running(child_pid)

    del p2p_daemon
    assert not is_process_running(child_pid)


def handle_square(x):
    return x ** 2


def handle_add(args):
    result = args[0]
    for i in range(1, len(args)):
        result = result + args[i]
    return result


@pytest.mark.parametrize(
    "test_input,handle",
    [
        pytest.param(10, handle_square, id="square_integer"),
        pytest.param((1, 2), handle_add, id="add_integers"),
        pytest.param(([1, 2, 3], [12, 13]), handle_add, id="add_lists"),
        pytest.param(2, lambda x: x ** 3, id="lambda")
    ]
)
@pytest.mark.asyncio
async def test_call_peer_single_process(test_input, handle, handler_name="handle"):
    server = await P2P.create()
    server_pid = server._child.pid
    await server.add_stream_handler(handler_name, handle)
    assert is_process_running(server_pid)

    client = await P2P.create()
    client_pid = client._child.pid
    assert is_process_running(client_pid)

    await asyncio.sleep(1)
    result = await client.call_peer_handler(server.id, handler_name, test_input)
    assert result == handle(test_input)

    await server.stop_listening()
    del server
    assert not is_process_running(server_pid)

    del client
    assert not is_process_running(client_pid)


@pytest.mark.asyncio
async def test_call_peer_different_processes():
    handler_name = "square"
    test_input = np.random.randn(2, 3)

    server_side, client_side = mp.Pipe()
    response_received = mp.Value(np.ctypeslib.as_ctypes_type(np.int32))
    response_received.value = 0

    async def run_server():
        server = await P2P.create()
        server_pid = server._child.pid
        await server.add_stream_handler(handler_name, handle_square)
        assert is_process_running(server_pid)

        server_side.send(server.id)
        while response_received.value == 0:
            await asyncio.sleep(0.5)

        await server.stop_listening()
        del server
        assert not is_process_running(server_pid)

    def server_target():
        asyncio.run(run_server())

    proc = mp.Process(target=server_target)
    proc.start()

    client = await P2P.create()
    client_pid = client._child.pid
    assert is_process_running(client_pid)

    await asyncio.sleep(1)
    peer_id = client_side.recv()

    result = await client.call_peer_handler(peer_id, handler_name, test_input)
    assert np.allclose(result, handle_square(test_input))
    response_received.value = 1

    del client
    assert not is_process_running(client_pid)

    proc.join()


@pytest.mark.parametrize(
    "test_input,handle",
    [
        pytest.param(np.random.randn(2, 3), handle_square, id="square"),
        pytest.param([np.random.randn(2, 3), np.random.randn(2, 3)], handle_add, id="add"),
    ]
)
@pytest.mark.asyncio
async def test_call_peer_numpy(test_input, handle, handler_name="handle"):
    server = await P2P.create()
    await server.add_stream_handler(handler_name, handle)
    client = await P2P.create()

    await asyncio.sleep(1)
    result = await client.call_peer_handler(server.id, handler_name, test_input)
    assert np.allclose(result, handle(test_input))


@pytest.mark.asyncio
async def test_call_peer_error(handler_name="handle"):
    server = await P2P.create()
    await server.add_stream_handler(handler_name, handle_add)
    client = await P2P.create()

    await asyncio.sleep(1)
    result = await client.call_peer_handler(server.id, handler_name,
                                            [np.zeros((2, 3)), np.zeros((3, 2))])
    assert type(result) == ValueError
