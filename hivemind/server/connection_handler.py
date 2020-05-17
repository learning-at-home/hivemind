from socket import socket
from typing import Tuple, Dict

from hivemind.runtime.expert_backend import ExpertBackend
from hivemind.utils import PytorchSerializer, Connection


async def handle_connection(connection_tuple: Tuple[socket, str], experts: Dict[str, ExpertBackend]):
    with Connection(*connection_tuple) as connection:
        try:
            header = connection.recv_header()
            payload = PytorchSerializer.loads(connection.recv_raw())

            if header == 'fwd_':
                uid, inputs = payload
                response = experts[uid].forward_pool.submit_task(*inputs).result()
            elif header == 'bwd_':
                uid, inputs_and_grad_outputs = payload
                response = experts[uid].backward_pool.submit_task(*inputs_and_grad_outputs).result()
            elif header == 'info':
                uid = payload
                response = experts[uid].get_info()
            else:
                raise NotImplementedError(f"Unknown header: {header}")

            connection.send_raw('rest', PytorchSerializer.dumps(response))
        except RuntimeError:
            # socket connection broken
            pass
