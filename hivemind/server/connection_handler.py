import asyncio
import multiprocessing as mp
import signal
from socket import socket
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import os

import torch

from hivemind.utils import PytorchSerializer, get_logger


def ask_exit(loop, executor):
    executor.shutdown()
    loop.stop()


logger = get_logger(__name__)


def worker_init_fn():
    torch.set_num_threads(1)


class ConnectionHandler(mp.Process):
    def __init__(self, port, conn_handler_processes, experts):
        super().__init__()
        self.addr = ('', port)
        self.conn_handler_processes = conn_handler_processes
        self.experts = experts
        self.ready = mp.Event()

    def run(self):
        torch.set_num_threads(1)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        process_pool = ProcessPoolExecutor(self.conn_handler_processes, mp_context=mp.get_context('forkserver'),
                                           initializer=worker_init_fn)
        thread_ppol = ThreadPoolExecutor(self.conn_handler_processes * 2)
        sock = socket()
        sock.bind(self.addr)
        sock.setblocking(False)
        for signame in signal.SIGINT, signal.SIGTERM:
            self.loop.add_signal_handler(
                signame,
                partial(ask_exit, self.loop, process_pool))
        handle_connection_coro = partial(handle_connection, process_pool=process_pool, thread_pool=thread_ppol, experts=self.experts)
        start_server_fn = asyncio.start_server(handle_connection_coro, sock=sock)
        self.loop.run_until_complete(start_server_fn)
        self.ready.set()
        self.loop.run_forever()


async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, process_pool, thread_pool, experts):
    logger.debug(f'Receiving message from the connection')
    data = await reader.read()
    header = data[:4].decode()
    loop = asyncio.get_running_loop()
    logger.debug(f'Message received, deserializing')
    task_id, *payload = await loop.run_in_executor(process_pool, PytorchSerializer.loads, data[12:])
    logger.debug(f'{task_id} Payload deserialized, processing')

    if header == 'fwd_':
        uid, inputs = payload
        logger.debug(f'{task_id} Submitting task')
        future = await experts[uid].forward_pool.submit_task(inputs, executor=thread_pool)
        logger.debug(f'{task_id} Awaiting result from backend')
        response = await future.result()
    elif header == 'bwd_':
        uid, inputs_and_grad_outputs = payload
        logger.debug(f'{task_id} Submitting task')
        future = await experts[uid].backward_pool.submit_task(inputs_and_grad_outputs, executor=thread_pool)
        logger.debug(f'{task_id} Awaiting result from backend')
        response = await future.result()
    elif header == 'info':
        uid, = payload
        response = experts[uid].metadata
    else:
        raise NotImplementedError(f"Unknown header: {header}")

    logger.debug(f'{task_id} Serializing result')
    raw_response = await loop.run_in_executor(process_pool, PytorchSerializer.dumps, response)
    logger.debug(f'{task_id} Sending the result')

    writer.write('rest'.encode())
    writer.write(len(raw_response).to_bytes(8, byteorder='big'))
    writer.write(raw_response)
    await writer.drain()
    logger.debug(f'{task_id} Result sent')
