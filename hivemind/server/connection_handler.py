import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from socket import socket, AF_INET, SOCK_STREAM, SO_REUSEADDR, SOL_SOCKET, timeout
from typing import Tuple, Dict

from hivemind.runtime.expert_backend import ExpertBackend
from hivemind.utils import PytorchSerializer, AsyncConnection


def shutdown(sock):
    for task in asyncio.Task.all_tasks():
        task.cancel()
    sock.close()


class ConnectionHandler(mp.Process):
    def __init__(self, port, conn_handler_processes, experts):
        super().__init__()
        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self.sock.bind(('', port))
        self.sock.listen()
        self.sock.setblocking(False)
        self.conn_handler_processes = conn_handler_processes
        self.experts = experts

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.set_default_executor(ThreadPoolExecutor(1000))
        self.executor = ProcessPoolExecutor(self.conn_handler_processes, mp_context=mp.get_context('spawn'))
        asyncio.run(run_socket_server(self.sock, self.executor, self.experts))

    def shutdown(self):
        shutdown(self.sock)


async def run_socket_server(sock, pool, experts):
    while True:
        try:
            loop = asyncio.get_running_loop()
            conn_tuple = await loop.sock_accept(sock)
            loop.create_task(handle_connection(conn_tuple, experts, pool))
        except KeyboardInterrupt as e:
            print(f'Socket loop has caught {type(e)}, exiting')
            break
        except (timeout, BrokenPipeError, ConnectionResetError, NotImplementedError):
            continue


async def handle_connection(connection_tuple: Tuple[socket, str], experts: Dict[str, ExpertBackend], pool):
    with AsyncConnection(*connection_tuple) as connection:
        try:
            loop = asyncio.get_running_loop()
            header, raw_payload = await connection.recv_message()
            payload = await loop.run_in_executor(pool, PytorchSerializer.loads, raw_payload)

            if header == 'fwd_':
                uid, inputs = payload
                future = await experts[uid].forward_pool.submit_task(*inputs)
                response = await future.result()
            elif header == 'bwd_':
                uid, inputs_and_grad_outputs = payload
                future = await experts[uid].backward_pool.submit_task(*inputs_and_grad_outputs)
                response = await future.result()
            elif header == 'info':
                uid = payload
                response = experts[uid].metadata
            else:
                raise NotImplementedError(f"Unknown header: {header}")

            raw_response = await loop.run_in_executor(pool, PytorchSerializer.dumps, response)
            await connection.send_raw('rest', raw_response)
        except RuntimeError as e:
            raise e
            # socket connection broken
            pass
