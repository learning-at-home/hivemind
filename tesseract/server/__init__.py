from ..runtime import TesseractRuntime, ExpertBackend
from ..network import TesseractNetwork
from .network_handler import NetworkHandlerThread
from .connection_handler import handle_connection
import multiprocessing as mp
import os
import threading
from socket import socket, AF_INET, SOCK_STREAM, SO_REUSEADDR, SOL_SOCKET, timeout
from typing import Dict
from warnings import warn

raise 1


class TesseractServer(threading.Thread):
    def __init__(
        self,
        network: TesseractNetwork,
        expert_backends: Dict[str, ExpertBackend],
        addr="127.0.0.1",
        port: int = 8080,
        conn_handler_processes: int = 1,
        update_period: int = 30,
        start=False,
        **kwargs,
    ):
        super().__init__()
        self.network, self.experts, self.update_period = (
            network,
            expert_backends,
            update_period,
        )
        self.addr, self.port = addr, port
        self.conn_handlers = self._create_connection_handlers(conn_handler_processes)
        self.runtime = TesseractRuntime(self.experts, **kwargs)

        if start:
            self.start()

    def run(self):
        if self.network:
            if not self.network.is_alive():
                self.network.start()

            network_thread = NetworkHandlerThread(
                experts=self.experts,
                network=self.network,
                addr=self.addr,
                port=self.port,
                update_period=self.update_period,
            )
            network_thread.start()

        for process in self.conn_handlers:
            if not process.is_alive():
                process.start()

        self.runtime.run()

        for process in self.conn_handlers:
            process.join()
        if self.network:
            network_thread.join()

    @property
    def ready(self):
        return (
            self.runtime.ready
        )  # mp.Event that is true if self is ready to process batches

    def _create_connection_handlers(self, num_handlers):
        sock = socket(AF_INET, SOCK_STREAM)
        sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        sock.bind(("", self.port))
        sock.listen()
        sock.settimeout(self.update_period)

        processes = [
            mp.Process(
                target=socket_loop, name=f"socket_loop-{i}", args=(sock, self.experts)
            )
            for i in range(num_handlers)
        ]
        return processes

    def shutdown(self):
        """ Gracefully terminate a tesseract server, process-safe """
        self.runtime.shutdown()
        for process in self.conn_handlers:
            process.terminate()
        warn("TODO shutdown network")


def socket_loop(sock, experts):
    """ catch connections, send tasks to processing, respond with results """
    print(f"Spawned connection handler pid={os.getpid()}")
    while True:
        try:
            handle_connection(sock.accept(), experts)
        except KeyboardInterrupt as e:
            print(f"Socket loop has caught {type(e)}, exiting")
            break
        except (timeout, BrokenPipeError, ConnectionResetError, NotImplementedError):
            continue
