import asyncio
import multiprocessing as mp
import signal
from concurrent.futures import ProcessPoolExecutor
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
        print(os.getpid())
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.executor = ProcessPoolExecutor(self.conn_handler_processes, mp_context=mp.get_context('forkserver'),
                                            initializer=worker_init_fn)
        for signame in signal.SIGINT, signal.SIGTERM:
            self.loop.add_signal_handler(
                signame,
                partial(ask_exit, self.loop, self.executor))
        start_server_fn = asyncio.start_server(partial(handle_connection, pool=self.executor, experts=self.experts), *self.addr)
        self.loop.run_until_complete(start_server_fn)
        self.ready.set()
        self.loop.run_forever()


async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, pool, experts):
    logger.debug(f'Receiving message from the connection')
    data = await reader.read()
    header = data[:4].decode()
    loop = asyncio.get_running_loop()
    logger.debug(f'Message received, deserializing')
    task_id, *payload = await loop.run_in_executor(pool, PytorchSerializer.loads, data[12:])
    logger.debug(f'{task_id} Payload deserialized, processing')

    if header == 'fwd_':
        uid, inputs = payload
        logger.debug(f'{task_id} Submitting task')
        future = await experts[uid].forward_pool.submit_task(*inputs)
        logger.debug(f'{task_id} Awaiting result from backend')
        response = await future.result()
    elif header == 'bwd_':
        uid, inputs_and_grad_outputs = payload
        logger.debug(f'{task_id} Submitting task')
        future = await experts[uid].backward_pool.submit_task(*inputs_and_grad_outputs)
        logger.debug(f'{task_id} Awaiting result from backend')
        response = await future.result()
    elif header == 'info':
        uid, = payload
        response = experts[uid].metadata
    else:
        raise NotImplementedError(f"Unknown header: {header}")

    logger.debug(f'{task_id} Serializing result')
    raw_response = await loop.run_in_executor(pool, PytorchSerializer.dumps, response)
    logger.debug(f'{task_id} Sending the result')

    writer.write('rest'.encode())
    writer.write(len(raw_response).to_bytes(8, byteorder='big'))
    writer.write(raw_response)
    await writer.drain()
    logger.debug(f'{task_id} Result sent')
