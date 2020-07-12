import ctypes
import multiprocessing as mp
from multiprocessing import synchronize, pool, context
import threading
from typing import Dict, Optional

from hivemind.dht import DHT
from hivemind.server.runtime import Runtime
from hivemind.server.task_pool import Task, TaskPool, TaskPoolBase
from hivemind.server.expert_backend import ExpertBackend
from hivemind.server.checkpoint_saver import CheckpointSaver
from hivemind.server.connection_handler import ConnectionHandler
from hivemind.server.dht_handler import DHTHandlerThread
from hivemind.utils import Endpoint, get_port, replace_port, find_open_port


class Server(threading.Thread):
    """
    Server allows you to host "experts" - pytorch sub-networks used by Decentralized Mixture of Experts.
    After creation, a server should be started: see Server.run or Server.run_in_background.

    A working server does 3 things:
     - processes incoming forward/backward requests via Runtime (created by the server)
     - publishes updates to expert status every :update_period: seconds
     - follows orders from HivemindController - if it exists

    :type dht: DHT or None. Server with dht=None will NOT be visible from DHT,
     but it will still support accessing experts directly with RemoteExpert(uid=UID, endpoint="IPADDR:PORT").
    :param expert_backends: dict{expert uid (str) : ExpertBackend} for all expert hosted by this server.
    :param listen_on: server's dht address that determines how it can be accessed. Address and (optional) port
    :param num_connection_handlers: maximum number of simultaneous requests. Please note that the default value of 1
        if too small for normal functioning, we recommend 4 handlers per expert backend.
    :param update_period: how often will server attempt to publish its state (i.e. experts) to the DHT;
        if dht is None, this parameter is ignored.
    :param start: if True, the server will immediately start as a background thread and returns control after server
        is ready (see .ready below)
    """

    def __init__(
            self, dht: Optional[DHT], expert_backends: Dict[str, ExpertBackend], listen_on: Endpoint = "0.0.0.0:*",
            num_connection_handlers: int = 1, update_period: int = 30, start=False, checkpoint_dir=None, **kwargs):
        super().__init__()
        self.dht, self.experts, self.update_period = dht, expert_backends, update_period
        if get_port(listen_on) is None:
            self.listen_on = listen_on = replace_port(listen_on, new_port=find_open_port())
        self.port = get_port(listen_on)

        self.conn_handlers = [ConnectionHandler(listen_on, self.experts) for _ in range(num_connection_handlers)]
        if checkpoint_dir is not None:
            self.checkpoint_saver = CheckpointSaver(expert_backends, checkpoint_dir, update_period)
        else:
            self.checkpoint_saver = None
        self.runtime = Runtime(self.experts, **kwargs)

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

            dht_handler_thread = DHTHandlerThread(
                experts=self.experts, dht=self.dht, endpoint=self.listen_on, update_period=self.update_period)
            dht_handler_thread.start()
        if self.checkpoint_saver is not None:
            self.checkpoint_saver.start()

        for process in self.conn_handlers:
            if not process.is_alive():
                process.start()

        for process in self.conn_handlers:
            process.ready.wait()

        self.runtime.run()

        for process in self.conn_handlers:
            process.join()
        if self.dht:
            dht_handler_thread.stop = True
            dht_handler_thread.join()
        if self.checkpoint_saver is not None:
            self.checkpoint_saver.stop = True
            self.checkpoint_saver.join()

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
        for process in self.conn_handlers:
            process.terminate()

        if self.dht is not None:
            self.dht.shutdown()

        self.runtime.shutdown()
