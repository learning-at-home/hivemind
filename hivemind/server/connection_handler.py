import asyncio
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from uuid import uuid4
import signal
from functools import partial
from hivemind.utils import PytorchSerializer
import torch


def ask_exit(loop, executor):
    executor.shutdown()
    loop.stop()


logger = logging.getLogger(__name__)


def worker_init_fn():
    torch.set_num_threads(1)


class ConnectionHandler(mp.Process):
    def __init__(self, port, conn_handler_processes, experts):
        super().__init__()
        self.addr = ('', port)
        self.conn_handler_processes = conn_handler_processes
        self.experts = experts

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.executor = ProcessPoolExecutor(self.conn_handler_processes, mp_context=mp.get_context('spawn'))
        for signame in signal.SIGINT, signal.SIGTERM:
            self.loop.add_signal_handler(
                signame,
                partial(ask_exit, self.loop, self.executor))
        just_read = lambda reader, writer: just_read_fn(reader, writer, self.executor, self.experts)
        start_server_fn = asyncio.start_server(just_read, *self.addr)
        self.loop.run_until_complete(start_server_fn)
        self.loop.run_forever()


async def just_read_fn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, pool, experts):
    logger.info(f'Receiving message from the connection')
    data = await reader.read()
    header = data[:4].decode()
    length = int.from_bytes(data[4:12], byteorder='big')
    assert len(data) - 12 == length
    loop = asyncio.get_running_loop()
    logger.info(f'Message received, deserializing')
    task_id, *payload = await loop.run_in_executor(pool, PytorchSerializer.loads, data[12:])
    logger.info(f'{task_id} Payload deserialized, processing')

    if header == 'fwd_':
        uid, inputs = payload
        logger.info(f'{task_id} Submitting task')
        future = await experts[uid].forward_pool.submit_task(*inputs)
        logger.info(f'{task_id} Awaiting result from backend')
        response = await future.result()
    elif header == 'bwd_':
        uid, inputs_and_grad_outputs = payload
        logger.info(f'{task_id} Submitting task')
        future = await experts[uid].backward_pool.submit_task(*inputs_and_grad_outputs)
        logger.info(f'{task_id} Awaiting result from backend')
        response = await future.result()
    elif header == 'info':
        uid, = payload
        response = experts[uid].metadata
    else:
        raise NotImplementedError(f"Unknown header: {header}")

    logger.info(f'{task_id} Serializing result')
    raw_response = await loop.run_in_executor(pool, PytorchSerializer.dumps, response)
    logger.info(f'{task_id} Sending the result')

    writer.write('rest'.encode())
    await writer.drain()
    writer.write(len(raw_response).to_bytes(8, byteorder='big'))
    await writer.drain()
    writer.write(raw_response)
    await writer.drain()
    writer.close()
    await writer.wait_closed()
    logger.info(f'{task_id} Result sent')
