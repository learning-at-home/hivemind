import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from socket import socket
from uuid import uuid4

from hivemind.utils import get_logger
from hivemind.utils.connection import HEADER_SIZE, DESTINATION_LENGTH_SIZE, PAYLOAD_LENGTH_SIZE

logger = get_logger(__name__)


class ConnectionHandler(mp.Process):
    def __init__(self, port, conn_handler_processes, experts):
        super().__init__()
        self.addr = ('', port)
        self.conn_handler_processes = conn_handler_processes
        self.experts = experts
        self.ready = mp.Event()

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        thread_pool = ThreadPoolExecutor(200)
        thread_pool_send = ThreadPoolExecutor(10)
        self.loop.set_default_executor(thread_pool)
        sock = socket()
        sock.bind(self.addr)
        sock.setblocking(False)

        handle_connection_coro = partial(handle_connection, thread_pool=thread_pool_send, experts=self.experts)
        start_server_fn = asyncio.start_server(handle_connection_coro, sock=sock)
        self.loop.run_until_complete(start_server_fn)
        self.ready.set()
        self.loop.run_forever()


async def read_message(reader: asyncio.StreamReader):
    header_binary = await reader.readexactly(HEADER_SIZE)
    header = header_binary.decode()
    destination_length_binary = await reader.readexactly(DESTINATION_LENGTH_SIZE)
    destination_length = int.from_bytes(destination_length_binary, byteorder='big')
    destination_binary = await reader.readexactly(destination_length)
    destination = destination_binary.decode()
    payload_length_binary = await reader.readexactly(PAYLOAD_LENGTH_SIZE)
    payload_length = int.from_bytes(payload_length_binary, byteorder='big')
    payload = await reader.readexactly(payload_length)
    return header, destination, payload


async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, thread_pool, experts):
    # logger.debug(f'Receiving message from the connection')

    header, destination, payload = await read_message(reader)
    # logger.debug(f'Message received, deserializing')
    # task_id = str(uuid4())[:4]

    if header == 'fwd_':
        # logger.debug(f'{task_id} Submitting task')
        future = await experts[destination].forward_pool.submit_task(payload, executor=thread_pool)
        # logger.debug(f'{task_id} Awaiting result from backend')
        response = await future.result()
    elif header == 'bwd_':
        # logger.debug(f'{task_id} Submitting task')
        future = await experts[destination].backward_pool.submit_task(payload, executor=thread_pool)
        # logger.debug(f'{task_id} Awaiting result from backend')
        response = await future.result()
    elif header == 'info':
        response = experts[destination].metadata
    else:
        raise NotImplementedError(f"Unknown header: {header}")

    # logger.debug(f'{task_id} Sending the result')
    payload = ('rest'.encode()
               + int(0).to_bytes(DESTINATION_LENGTH_SIZE, byteorder='big')
               + len(response).to_bytes(PAYLOAD_LENGTH_SIZE, byteorder='big')
               + response)

    writer.write(payload)
    # await writer.drain()
    # logger.debug(f'{task_id} Result sent')
