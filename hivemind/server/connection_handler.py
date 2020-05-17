from socket import socket
from typing import Tuple, Dict
import asyncio

from hivemind.runtime.expert_backend import ExpertBackend
from hivemind.utils import PytorchSerializer, AsyncConnection


async def handle_connection(connection_tuple: Tuple[socket, str], experts: Dict[str, ExpertBackend], pool):
    with AsyncConnection(*connection_tuple) as connection:
        try:
            loop = asyncio.get_running_loop()
            header, raw_payload = await connection.recv_message()
            payload = await loop.run_in_executor(pool, PytorchSerializer.loads, raw_payload)

            if header == 'fwd_':
                uid, inputs = payload
                future = experts[uid].forward_pool.submit_task(*inputs)
                response = await loop.run_in_executor(pool, future.result)
            elif header == 'bwd_':
                uid, inputs_and_grad_outputs = payload
                future = experts[uid].backward_pool.submit_task(*inputs_and_grad_outputs)
                response = await loop.run_in_executor(pool, future.result)
            elif header == 'info':
                uid = payload
                fn = experts[uid].get_info
                response = await loop.run_in_executor(pool, fn)
            else:
                raise NotImplementedError(f"Unknown header: {header}")

            raw_response = await loop.run_in_executor(pool, PytorchSerializer.dumps, response)
            await connection.send_raw('rest', raw_response)
        except RuntimeError:
            # socket connection broken
            pass
