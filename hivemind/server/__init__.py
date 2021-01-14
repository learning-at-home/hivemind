from __future__ import annotations

import multiprocessing as mp
import multiprocessing.synchronize
import random
import threading
from contextlib import contextmanager
from functools import partial
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import torch

import hivemind
from hivemind.dht import DHT
from hivemind.server.checkpoints import CheckpointSaver, load_weights, dir_is_correct
from hivemind.server.connection_handler import ConnectionHandler
from hivemind.server.dht_handler import DHTHandlerThread
from hivemind.server.expert_backend import ExpertBackend
from hivemind.server.layers import name_to_block, name_to_input
from hivemind.server.runtime import Runtime
from hivemind.server.task_pool import Task, TaskPool, TaskPoolBase
from hivemind.utils import Endpoint, get_port, replace_port, find_open_port, get_logger
from hivemind.proto.runtime_pb2 import CompressionType

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

    @staticmethod
    def create(listen_on='0.0.0.0:*', num_experts: int = None, expert_uids: str = None, expert_pattern: str = None,
               expert_cls='ffn', hidden_dim=1024, optim_cls=torch.optim.Adam, num_handlers=None, max_batch_size=4096,
               device=None, no_dht=False, initial_peers=(), dht_port=None, checkpoint_dir: Optional[Path] = None,
               load_experts=False, compression=CompressionType.NONE, *, start: bool, **kwargs) -> Server:
        """
        Instantiate a server with several identical experts. See argparse comments below for details
        :param listen_on: network interface with address and (optional) port, e.g. "127.0.0.1:1337" or "[::]:80"
        :param num_experts: run this many identical experts
        :param expert_pattern: a string pattern or a list of expert uids,  example: myprefix.[0:32].[0:256]\
           means "sample random experts between myprefix.0.0 and myprefix.255.255;
        :param expert_uids: spawn experts with these exact uids, overrides num_experts and expert_pattern
        :param expert_cls: expert type from hivemind.server.layers, e.g. 'ffn', 'transformer', 'det_dropout' or 'nop';
        :param hidden_dim: main dimension for expert_cls
        :param num_handlers: server will use this many parallel processes to handle incoming requests
        :param max_batch_size: total num examples in the same batch will not exceed this value
        :param device: all experts will use this device in torch notation; default: cuda if available else cpu
        :param optim_cls: uses this optimizer to train all experts
        :param no_dht: if specified, the server will not be attached to a dht
        :param initial_peers: a list of peers that will introduce this node to the dht,\
           e.g. ('123.11.22.33:1337', '[fe80::abe2:db1c:be7d:5a85]:4567'), default = no peers

        :param dht_port:  DHT node will listen on this port, default = find open port
           You can then use this node as initial peer for subsequent servers.

        :param checkpoint_dir: directory to save expert checkpoints
        :param load_experts: whether to load expert checkpoints from checkpoint_dir

        :param compression: if specified, use this compression to pack all inputs, outputs and gradients by all experts
            hosted on this server. For a more fine-grained compression, start server in python and specify compression
            for each BatchTensorProto in ExpertBackend for the respective experts.

        :param start: if True, starts server right away and returns when server is ready for requests
        """
        if len(kwargs) != 0:
            logger.info("Ignored kwargs:", kwargs)
        assert expert_cls in name_to_block

        if no_dht:
            dht = None
        else:
            dht_endpoint = replace_port(listen_on, dht_port or hivemind.find_open_port())
            dht = hivemind.DHT(initial_peers=initial_peers, start=True, listen_on=dht_endpoint)
            logger.info(f"Running DHT node on port {dht.port}, initial peers = {initial_peers}")

        if load_experts:
            assert dir_is_correct(checkpoint_dir)
            assert expert_uids is None, "Can't both load saved experts and create new ones from given UIDs"
            expert_uids = [child.name for child in checkpoint_dir.iterdir() if (child / 'checkpoint_last.pt').exists()]
            if expert_uids:
                logger.info(f"Located checkpoints for experts {expert_uids}, ignoring UID generation options")
            else:
                logger.info(f"No expert checkpoints found in {checkpoint_dir}, generating...")

        assert (expert_pattern is None and num_experts is None) or (expert_uids is None) or (num_experts == 0), \
            "Please provide either expert_uids *or* num_experts and expert_pattern, but not both"

        # get expert uids if not loaded previously
        if expert_uids is None:
            assert num_experts is not None, "Please specify either expert_uids or num_experts [and expert_pattern]"
            logger.info(f"Generating expert uids from pattern {expert_pattern}")
            expert_uids = generate_uids_from_pattern(num_experts, expert_pattern, dht=dht)

        num_experts = len(expert_uids)
        num_handlers = num_handlers if num_handlers is not None else num_experts * 8
        optim_cls = optim_cls if optim_cls is not None else partial(torch.optim.SGD, lr=0.0)
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        sample_input = name_to_input[expert_cls](4, hidden_dim)
        if isinstance(sample_input, tuple):
            args_schema = tuple(hivemind.BatchTensorDescriptor.from_tensor(arg, compression) for arg in sample_input)
        else:
            args_schema = (hivemind.BatchTensorDescriptor.from_tensor(sample_input, compression),)

        # initialize experts
        experts = {}
        for expert_uid in expert_uids:
            expert = name_to_block[expert_cls](hidden_dim)
            experts[expert_uid] = hivemind.ExpertBackend(name=expert_uid, expert=expert,
                                                         args_schema=args_schema,
                                                         outputs_schema=hivemind.BatchTensorDescriptor(
                                                             hidden_dim, compression=compression),
                                                         opt=optim_cls(expert.parameters()),
                                                         max_batch_size=max_batch_size)

        if load_experts:
            load_weights(experts, checkpoint_dir)

        server = Server(dht, experts, listen_on=listen_on, num_connection_handlers=num_handlers, device=device,
                        start=start)
        return server

    def run(self):
        """
        Starts Server in the current thread. Initializes dht if necessary, starts connection handlers,
        runs Runtime (self.runtime) to process incoming requests.
        """
        logger.info(f"Server started at {self.listen_on}")
        logger.info(f"Got {len(self.experts)} experts:")
        for expert_name, backend in self.experts.items():
            num_parameters = sum(p.numel() for p in backend.expert.parameters() if p.requires_grad)
            logger.info(f"{expert_name}: {backend.expert.__class__.__name__}, {num_parameters} parameters")

        if self.dht:
            if not self.dht.is_alive():
                self.dht.run_in_background(await_ready=True)

            if self.experts:
                dht_handler_thread = DHTHandlerThread(
                    experts=self.experts, dht=self.dht, endpoint=self.listen_on, update_period=self.update_period)
                dht_handler_thread.start()
        if self.checkpoint_saver is not None:
            self.checkpoint_saver.start()

        for process in self.conn_handlers:
            if not process.is_alive():
                process.start()
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


@contextmanager
def background_server(*args, shutdown_timeout=5, **kwargs) -> Tuple[hivemind.Endpoint, hivemind.Endpoint]:
    """ A context manager that creates server in a background thread, awaits .ready on entry and shutdowns on exit """
    pipe, runners_pipe = mp.Pipe(duplex=True)
    runner = mp.get_context("spawn").Process(target=_server_runner, args=(runners_pipe, *args), kwargs=kwargs)

    try:
        runner.start()
        yield pipe.recv()  # once the server is ready, runner will send us a tuple(hostname, port, dht port)
        pipe.send('SHUTDOWN')  # on exit from context, send shutdown signal
    finally:
        runner.join(timeout=shutdown_timeout)
        if runner.is_alive():
            logger.info("Server failed to shutdown gracefully, terminating it the hard way...")
            runner.kill()
            logger.info("Server terminated.")


def _server_runner(pipe, *args, **kwargs):
    server = Server.create(*args, start=True, **kwargs)
    try:
        if server.dht is not None:
            dht_listen_on = hivemind.replace_port(server.dht.listen_on, server.dht.port)
        else:
            dht_listen_on = None
        pipe.send((server.listen_on, dht_listen_on))
        pipe.recv()  # wait for shutdown signal
    finally:
        logger.info("Shutting down server...")
        server.shutdown()
        server.join()
        logger.info("Server shut down.")


def generate_uids_from_pattern(num_experts: int, expert_pattern: Optional[str], dht: Optional[DHT] = None,
                               attempts_per_expert=10) -> List[str]:
    """
    Sample experts from a given pattern, remove duplicates.
    :param num_experts: sample this many unique expert uids
    :param expert_pattern: a string pattern or a list of expert uids,  example: myprefix.[0:32].[0:256]\
     means "sample random experts between myprefix.0.0 and myprefix.255.255;
    :param dht: if specified, uses this DHT to check that expert uids are not yet occupied by other peers
    :param attempts_per_expert: give up if unable to generate a new expert uid after this many attempts per uid
    :note: this method is not strictly process-safe. If several servers run it concurrently, they have
     a small chance of sampling duplicate expert uids.
    """
    remaining_attempts = attempts_per_expert * num_experts
    found_uids, attempted_uids = list(), set()

    def _generate_uid():
        if expert_pattern is None:
            return f"expert{hivemind.dht.UID_DELIMITER}{attempts_per_expert * num_experts - remaining_attempts}"

        uid = []
        for block in expert_pattern.split(hivemind.dht.UID_DELIMITER):
            try:
                if '[' not in block and ']' not in block:
                    uid.append(block)
                elif block.startswith('[') and block.endswith(']') and ':' in block:
                    slice_start, slice_end = map(int, block[1:-1].split(':'))
                    uid.append(str(random.randint(slice_start, slice_end - 1)))
                else:
                    raise ValueError("Block must be either fixed or a range [from:to]")
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                raise ValueError(f"Expert pattern {expert_pattern} has invalid block {block}, {e}")
        return hivemind.dht.UID_DELIMITER.join(uid)

    while remaining_attempts > 0 and len(found_uids) < num_experts:

        # 1. sample new expert uids at random
        new_uids = []
        while len(new_uids) + len(found_uids) < num_experts and remaining_attempts > 0:
            new_uid = _generate_uid()
            remaining_attempts -= 1
            if new_uid not in attempted_uids:
                attempted_uids.add(new_uid)
                new_uids.append(new_uid)

        # 2. look into DHT (if given) and remove duplicates
        if dht:
            existing_expert_uids = {found_expert.uid for found_expert in dht.get_experts(new_uids)
                                    if found_expert is not None}
            new_uids = [new_uid for new_uid in new_uids if new_uid not in existing_expert_uids]

        found_uids += new_uids

    if len(found_uids) != num_experts:
        logger.warning(f"Found only {len(found_uids)} out of {num_experts} free expert uids after "
                       f"{attempts_per_expert * num_experts} attempts")
    return found_uids
