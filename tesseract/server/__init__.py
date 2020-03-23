import multiprocessing as mp
import os
import threading
from socket import socket, AF_INET, SOCK_STREAM, SO_REUSEADDR, SOL_SOCKET, timeout
from typing import Dict

from .connection_handler import handle_connection
from .network_handler import NetworkHandlerThread
from ..network import TesseractNetwork
from ..runtime import TesseractRuntime, ExpertBackend


class TesseractServer(threading.Thread):
    """
    TesseractServer allows you to host "experts" - pytorch sub-networks used by Decentralized Mixture of Experts.
    After creation, a server should be started: see TesseractServer.run or TesseractServer.run_in_background.

    A working server does 3 things:
     - processes incoming forward/backward requests via TesseractRuntime (created by the server)
     - publishes updates to expert status every :update_period: seconds
     - follows orders from TesseractController - if it exists

    :type network: TesseractNetwork or None. Server with network=None will NOT be visible from DHT,
     but it will still support accessing experts directly with RemoteExpert(uid=UID, host=IPADDR, port=PORT).
    :param expert_backends: dict{expert uid (str) : ExpertBackend} for all expert hosted by this server.
    :param addr: server's network address that determines how it can be accessed. Default is local connections only.
    :param port: port to which server listens for requests such as expert forward or backward pass.
    :param conn_handler_processes: maximum number of simultaneous requests. Please note that the default value of 1
        if too small for normal functioning, we recommend 4 handlers per expert backend.
    :param update_period: how often will server attempt to publish its state (i.e. experts) to the DHT;
        if network is None, this parameter is ignored.
    :param start: if True, the server will immediately start as a background thread and returns control after server
        is ready (see .ready below)
    """

    def __init__(self, network: TesseractNetwork, expert_backends: Dict[str, ExpertBackend], addr='127.0.0.1',
                 port: int = 8080, conn_handler_processes: int = 1, update_period: int = 30, start=False,
                 **kwargs):
        super().__init__()
        self.network, self.experts, self.update_period = network, expert_backends, update_period
        self.addr, self.port = addr, port
        self.conn_handlers = self._create_connection_handlers(conn_handler_processes)
        self.runtime = TesseractRuntime(self.experts, **kwargs)

        if start:
            self.run_in_background(await_ready=True)

    def run(self):
        """
        Starts TesseractServer in the current thread. Initializes network if necessary, starts connection handlers,
        runs TesseractRuntime (self.runtime) to process incoming requests.
        """
        if self.network:
            if not self.network.is_alive():
                self.network.start()

            network_thread = NetworkHandlerThread(experts=self.experts, network=self.network,
                                                  addr=self.addr, port=self.port, update_period=self.update_period)
            network_thread.start()

        for process in self.conn_handlers:
            if not process.is_alive():
                process.start()

        self.runtime.run()

        for process in self.conn_handlers:
            process.join()
        if self.network:
            network_thread.join()

    def run_in_background(self, await_ready=True, timeout=None):
        """
        Starts TesseractServer in a background thread. if await_ready, this method will wait until background server
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError("TesseractServer didn't notify .ready in {timeout} seconds")

    @property
    def ready(self) -> mp.synchronize.Event:
        """
        An event (multiprocessing.Event) that is set when the server is ready to process requests.

        Example
        =======
        >>> server.start()
        >>> server.ready.wait(timeout=10)
        >>> print("Server ready" if server.ready.is_set() else "Server didn't start in 10 seconds")
        """
        return self.runtime.ready  # mp.Event that is true if self is ready to process batches

    def _create_connection_handlers(self, num_handlers):
        sock = socket(AF_INET, SOCK_STREAM)
        sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        sock.bind(('', self.port))
        sock.listen()
        sock.settimeout(self.update_period)

        processes = [mp.Process(target=socket_loop, name=f"socket_loop-{i}", args=(sock, self.experts))
                     for i in range(num_handlers)]
        return processes

    def shutdown(self):
        """
        Gracefully terminate a tesseract server, process-safe.
        Please note that terminating server otherwise (e.g. by killing processes) may result in zombie processes.
        If you did already cause a zombie outbreak, your only option is to kill them with -9 (SIGKILL).
        """
        self.ready.clear()
        for process in self.conn_handlers:
            process.terminate()

        if self.network is not None:
            self.network.shutdown()

        self.runtime.shutdown()


def socket_loop(sock, experts):
    """ catch connections, send tasks to processing, respond with results """
    print(f'Spawned connection handler pid={os.getpid()}')
    while True:
        try:
            handle_connection(sock.accept(), experts)
        except KeyboardInterrupt as e:
            print(f'Socket loop has caught {type(e)}, exiting')
            break
        except (timeout, BrokenPipeError, ConnectionResetError, NotImplementedError):
            continue
