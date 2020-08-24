import multiprocessing as mp
import multiprocessing.synchronize
import threading
import torch
from typing import Dict, Optional

import hivemind
from hivemind.dht import DHT
from hivemind.server.runtime import Runtime
from hivemind.server.task_pool import Task, TaskPool, TaskPoolBase
from hivemind.server.expert_backend import ExpertBackend
from hivemind.server.checkpoint_saver import CheckpointSaver
from hivemind.server.connection_handler import ConnectionHandler
from hivemind.server.dht_handler import DHTHandlerThread
from hivemind.server.layers.__init__ import name_to_block, name_to_input
from hivemind.utils import Endpoint, get_port, replace_port, find_open_port, get_logger

logger = get_logger(__name__)


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
            listen_on = replace_port(listen_on, new_port=find_open_port())
        self.listen_on, self.port = listen_on, get_port(listen_on)

        self.conn_handlers = [ConnectionHandler(listen_on, self.experts) for _ in range(num_connection_handlers)]
        if checkpoint_dir is not None:
            self.checkpoint_saver = CheckpointSaver(expert_backends, checkpoint_dir, update_period)
        else:
            self.checkpoint_saver = None
        self.runtime = Runtime(self.experts, **kwargs)

        if start:
            self.run_in_background(await_ready=True)

    def __enter__(self):
        if not self.ready:
            self.run_in_background(await_ready=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    @classmethod
    def create(cls, listen_on='0.0.0.0:*', num_experts=None, expert_uids=None, expert_cls='ffn', hidden_dim=1024,
               num_handlers=None, expert_prefix='expert', expert_offset=0, max_batch_size=16384, device=None,
               no_optimizer=False, no_dht=False, initial_peers=(), dht_port=None, root_port=None, verbose=True,
               start=False, **kwargs):  # removed type specification (-> Server)
        """
        Instantiate a server with several identical experts. See argparse comments below for details
        :param listen_on: network interface with address and (optional) port, e.g. "127.0.0.1:1337" or "[::]:80"
        :param num_experts: run this many identical experts
        :param expert_prefix: all expert uids will be {expert_prefix}.{index}
        :param expert_offset: expert uid will use indices in range(expert_offset, expert_offset + num_experts)
        :param expert_uids: spawn experts with these exact uids, overrides num_experts, expert_prefix and expert_offset
        :param expert_cls: expert type from test_utils.layers, e.g. 'ffn', 'transformer', 'det_dropout' or 'nop';
        :param hidden_dim: main dimension for expert_cls
        :param num_handlers: server will use this many parallel processes to handle incoming requests
        :param max_batch_size: total num examples in the same batch will not exceed this value
        :param device: all experts will use this device in torch notation; default: cuda if available else cpu
        :param no_optimizer: if specified, all optimizers use learning rate=0
        :param no_dht: if specified, the server will not be attached to a dht
        :param initial_peers: a list of peers that will introduce this node to the dht,
          e.g. [("1.2.3.4", 1337), ("127.0.0.1", 4321)]'), default = no peers
        :param dht_port:  DHT node will listen on this port, default = find open port
        :param root_port: if this server does not have initial_peers, it will create a virtual dht node on this port.
            You can then use this node as initial peer for subsequent servers.
        :param verbose: whether to print server started / finished / terminated events
        :param start: if True, starts server right away and returns when server is ready for requests
        """
        assert (expert_uids is None) != (num_experts is None and expert_prefix == 'expert' and expert_offset == 0), \
            "Please provide either expert uids *or* (num_experts, expert_prefix and expert_offset), not both"
        if verbose and len(kwargs) != 0:
            print("Ignored kwargs:", kwargs)
        assert expert_cls in name_to_block
        num_handlers = num_handlers if num_handlers is not None else num_experts * 8
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # initialize dht
        dht = None
        if not no_dht:
            if not len(initial_peers):
                logger.info("No initial peers provided. Starting additional dht as an initial peer.")
                dht_root = hivemind.DHT(initial_peers=initial_peers, start=True,
                                        listen_on=f"{hivemind.LOCALHOST}:{root_port or hivemind.find_open_port()}")
                logger.info(f"Initializing DHT with port {dht_root.port}")
                initial_peers = [f"{hivemind.LOCALHOST}:{dht_root.port}"]
            else:
                logger.info("Bootstrapping dht with peers:", initial_peers)
                if root_port is not None:
                    logger.info(f"Warning: root_port={root_port} will not be used since we already have peers.")

            dht = hivemind.DHT(initial_peers=initial_peers, start=True,
                               listen_on=f"{hivemind.LOCALHOST}:{dht_port or hivemind.find_open_port()}")
            if verbose:
                logger.info(f"Running dht node on port {dht.port}")

        sample_input = name_to_input[expert_cls](4, hidden_dim)
        if isinstance(sample_input, tuple):
            args_schema = tuple(hivemind.BatchTensorDescriptor.from_tensor(arg) for arg in sample_input)
        else:
            args_schema = (hivemind.BatchTensorDescriptor.from_tensor(sample_input),)

        # initialize experts
        if expert_uids is None:
            num_experts = num_experts if num_experts is not None else 1
            expert_uids = [f'{expert_prefix}{hivemind.DHT.UID_DELIMITER}{i + expert_offset}'
                           for i in range(num_experts)]

        experts = {}
        for expert_uid in expert_uids:
            expert = name_to_block[expert_cls](hidden_dim)
            opt = torch.optim.SGD(expert.parameters(), 0.0 if no_optimizer else 0.05)
            experts[expert_uid] = hivemind.ExpertBackend(name=expert_uid, expert=expert, opt=opt,
                                                         args_schema=args_schema,
                                                         outputs_schema=hivemind.BatchTensorDescriptor(hidden_dim),
                                                         max_batch_size=max_batch_size,
                                                         )
        # actually start server
        server = Server(
            dht, experts, listen_on=listen_on,
            num_connection_handlers=num_handlers, device=device)

        if start:
            server.run_in_background(await_ready=True)
            if verbose:
                logger.info(f"Server started at {server.listen_on}")
                logger.info(f"Got {len(experts)} active experts of type {expert_cls}: {list(experts.keys())}")
        return server

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
            dht_handler_thread.stop.set()
            dht_handler_thread.join()
        if self.checkpoint_saver is not None:
            self.checkpoint_saver.stop.set()
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
            self.dht.join()

        self.runtime.shutdown()
