import multiprocessing as mp

import threading
from collections import namedtuple
from typing import Dict, Optional

from .connection_handler import ConnectionHandler
from .dht_handler import DHTHandlerThread
from ..dht import DHT
from ..runtime import Runtime, ExpertBackend
from hivemind.runtime.task_pool import RemotePoolInterface

ExpertData = namedtuple('ExpertData', ('forward_pool', 'backward_pool', 'metadata'))


class Server(threading.Thread):
    """
    Server allows you to host "experts" - pytorch sub-networks used by Decentralized Mixture of Experts.
    After creation, a server should be started: see Server.run or Server.run_in_background.

    A working server does 3 things:
     - processes incoming forward/backward requests via Runtime (created by the server)
     - publishes updates to expert status every :update_period: seconds
     - follows orders from HivemindController - if it exists

    :type dht: DHT or None. Server with dht=None will NOT be visible from DHT,
     but it will still support accessing experts directly with RemoteExpert(uid=UID, host=IPADDR, port=PORT).
    :param expert_backends: dict{expert uid (str) : ExpertBackend} for all expert hosted by this server.
    :param addr: server's dht address that determines how it can be accessed. Default is local connections only.
    :param port: port to which server listens for requests such as expert forward or backward pass.
    :param conn_handler_processes: maximum number of simultaneous requests. Please note that the default value of 1
        if too small for normal functioning, we recommend 4 handlers per expert backend.
    :param update_period: how often will server attempt to publish its state (i.e. experts) to the DHT;
        if dht is None, this parameter is ignored.
    :param start: if True, the server will immediately start as a background thread and returns control after server
        is ready (see .ready below)
    """

    def __init__(self, dht: Optional[DHT], expert_backends: Dict[str, ExpertBackend], addr='127.0.0.1',
                 port: int = 8080, conn_handler_processes: int = 1, update_period: int = 30, start=False,
                 **kwargs):
        super().__init__()
        self.dht, self.experts, self.update_period = dht, expert_backends, update_period
        self.addr, self.port = addr, port
        self.runtime = Runtime(self.experts, **kwargs)
        self.conn_handler_processes = conn_handler_processes

        data_for_expert = {uid: ExpertData(RemotePoolInterface(expert.forward_pool),
                                           RemotePoolInterface(expert.backward_pool), expert.get_info())
                           for uid, expert in self.experts.items()}

        self.conn_handler_process = ConnectionHandler(self.port, self.conn_handler_processes, data_for_expert)

        if start:
            self.run_in_background(await_ready=True)

    def run(self):
        """
        Starts Server in the current thread. Initializes dht if necessary, starts connection handlers,
        runs Runtime (self.runtime) to process incoming requests.
        """
        if self.dht:
            if not self.dht.is_alive():
                self.dht.run_in_background(await_ready=True)

            dht_handler_thread = DHTHandlerThread(experts=self.experts, dht=self.dht,
                                                  addr=self.addr, port=self.port, update_period=self.update_period)
            dht_handler_thread.start()

        self.conn_handler_process.start()
        self.conn_handler_process.ready.wait()

        self.runtime.run()

        if self.dht:
            dht_handler_thread.join()

    def run_in_background(self, await_ready=True, timeout=None):
        """
        Starts Server in a background thread. if await_ready, this method will wait until background server
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError("Server didn't notify .ready in {timeout} seconds")

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

    def shutdown(self):
        """
        Gracefully terminate a hivemind server, process-safe.
        Please note that terminating server otherwise (e.g. by killing processes) may result in zombie processes.
        If you did already cause a zombie outbreak, your only option is to kill them with -9 (SIGKILL).
        """
        self.ready.clear()

        if self.dht is not None:
            self.dht.shutdown()

        self.conn_handler_process.terminate()
        self.conn_handler_process.join()

        self.runtime.shutdown()
