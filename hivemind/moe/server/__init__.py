from __future__ import annotations

import multiprocessing as mp
import multiprocessing.synchronize
import threading
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from multiaddr import Multiaddr

import hivemind
from hivemind.dht import DHT
from hivemind.moe.server.checkpoints import CheckpointSaver, is_directory, load_experts
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.moe.server.dht_handler import DHTHandlerThread, declare_experts, get_experts
from hivemind.moe.server.expert_backend import ExpertBackend
from hivemind.moe.server.expert_uid import UID_DELIMITER, generate_uids_from_pattern
from hivemind.moe.server.layers import (
    add_custom_models_from_file,
    name_to_block,
    name_to_input,
    register_expert_class,
    schedule_name_to_scheduler,
)
from hivemind.moe.server.runtime import Runtime
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils import BatchTensorDescriptor, Endpoint, get_free_port, get_logger, get_port, replace_port

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
        self,
        dht: Optional[DHT],
        expert_backends: Dict[str, ExpertBackend],
        listen_on: Endpoint = "0.0.0.0:*",
        num_connection_handlers: int = 1,
        update_period: int = 30,
        start=False,
        checkpoint_dir=None,
        **kwargs,
    ):
        super().__init__()
        self.dht, self.experts, self.update_period = dht, expert_backends, update_period
        if get_port(listen_on) is None:
            listen_on = replace_port(listen_on, new_port=get_free_port())
        self.listen_on, self.port = listen_on, get_port(listen_on)

        self.conn_handlers = [ConnectionHandler(listen_on, self.experts) for _ in range(num_connection_handlers)]
        if checkpoint_dir is not None:
            self.checkpoint_saver = CheckpointSaver(expert_backends, checkpoint_dir, update_period)
        else:
            self.checkpoint_saver = None
        self.runtime = Runtime(self.experts, **kwargs)

        if self.dht and self.experts:
            self.dht_handler_thread = DHTHandlerThread(
                experts=self.experts,
                dht=self.dht,
                endpoint=self.listen_on,
                update_period=self.update_period,
                daemon=True,
            )

        if start:
            self.run_in_background(await_ready=True)

    @classmethod
    def create(
        cls,
        listen_on="0.0.0.0:*",
        num_experts: int = None,
        expert_uids: str = None,
        expert_pattern: str = None,
        expert_cls="ffn",
        hidden_dim=1024,
        optim_cls=torch.optim.Adam,
        scheduler: str = "none",
        num_warmup_steps=None,
        num_total_steps=None,
        clip_grad_norm=None,
        num_handlers=None,
        min_batch_size=1,
        max_batch_size=4096,
        device=None,
        no_dht=False,
        initial_peers=(),
        checkpoint_dir: Optional[Path] = None,
        compression=CompressionType.NONE,
        stats_report_interval: Optional[int] = None,
        custom_module_path=None,
        *,
        start: bool,
    ) -> Server:
        """
        Instantiate a server with several identical experts. See argparse comments below for details
        :param listen_on: network interface with address and (optional) port, e.g. "127.0.0.1:1337" or "[::]:80"
        :param num_experts: run this many identical experts
        :param expert_pattern: a string pattern or a list of expert uids,  example: myprefix.[0:32].[0:256]\
           means "sample random experts between myprefix.0.0 and myprefix.255.255;
        :param expert_uids: spawn experts with these exact uids, overrides num_experts and expert_pattern
        :param expert_cls: expert type from hivemind.moe.server.layers, e.g. 'ffn' or 'transformer';
        :param hidden_dim: main dimension for expert_cls
        :param num_handlers: server will use this many parallel processes to handle incoming requests
        :param min_batch_size: total num examples in the same batch will be greater than this value
        :param max_batch_size: total num examples in the same batch will not exceed this value
        :param device: all experts will use this device in torch notation; default: cuda if available else cpu

        :param optim_cls: uses this optimizer to train all experts
        :param scheduler: if not `none`, the name of the expert LR scheduler
        :param num_warmup_steps: the number of warmup steps for LR schedule
        :param num_total_steps: the total number of steps for LR schedule
        :param clip_grad_norm: maximum gradient norm used for clipping

        :param no_dht: if specified, the server will not be attached to a dht
        :param initial_peers: multiaddrs of one or more active DHT peers (if you want to join an existing DHT)

        :param checkpoint_dir: directory to save and load expert checkpoints

        :param compression: if specified, use this compression to pack all inputs, outputs and gradients by all experts
            hosted on this server. For a more fine-grained compression, start server in python and specify compression
            for each BatchTensorProto in ExpertBackend for the respective experts.

        :param start: if True, starts server right away and returns when server is ready for requests
        :param stats_report_interval: interval between two reports of batch processing performance statistics
        """
        if custom_module_path is not None:
            add_custom_models_from_file(custom_module_path)
        assert expert_cls in name_to_block

        if no_dht:
            dht = None
        else:
            dht = hivemind.DHT(initial_peers=initial_peers, start=True)
            visible_maddrs_str = [str(a) for a in dht.get_visible_maddrs()]
            logger.info(f"Running DHT node on {visible_maddrs_str}, initial peers = {initial_peers}")

        assert (expert_pattern is None and num_experts is None and expert_uids is not None) or (
            num_experts is not None and expert_uids is None
        ), "Please provide either expert_uids *or* num_experts (possibly with expert_pattern), but not both"

        if expert_uids is None:
            if checkpoint_dir is not None:
                assert is_directory(checkpoint_dir)
                expert_uids = [
                    child.name for child in checkpoint_dir.iterdir() if (child / "checkpoint_last.pt").exists()
                ]
                total_experts_in_checkpoint = len(expert_uids)
                logger.info(f"Located {total_experts_in_checkpoint} checkpoints for experts {expert_uids}")

                if total_experts_in_checkpoint > num_experts:
                    raise ValueError(
                        f"Found {total_experts_in_checkpoint} checkpoints, but num_experts is set to {num_experts}, "
                        f"which is smaller. Either increase num_experts or remove unneeded checkpoints."
                    )
            else:
                expert_uids = []

            uids_to_generate = num_experts - len(expert_uids)
            if uids_to_generate > 0:
                logger.info(f"Generating {uids_to_generate} expert uids from pattern {expert_pattern}")
                expert_uids.extend(generate_uids_from_pattern(uids_to_generate, expert_pattern, dht))

        num_experts = len(expert_uids)
        num_handlers = num_handlers if num_handlers is not None else num_experts * 8
        optim_cls = optim_cls if optim_cls is not None else partial(torch.optim.SGD, lr=0.0)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        sample_input = name_to_input[expert_cls](3, hidden_dim)
        if isinstance(sample_input, tuple):
            args_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in sample_input)
        else:
            args_schema = (BatchTensorDescriptor.from_tensor(sample_input, compression),)

        scheduler = schedule_name_to_scheduler[scheduler]

        # initialize experts
        experts = {}
        for expert_uid in expert_uids:
            expert = name_to_block[expert_cls](hidden_dim)
            experts[expert_uid] = hivemind.ExpertBackend(
                name=expert_uid,
                expert=expert,
                args_schema=args_schema,
                optimizer=optim_cls(expert.parameters()),
                scheduler=scheduler,
                num_warmup_steps=num_warmup_steps,
                num_total_steps=num_total_steps,
                clip_grad_norm=clip_grad_norm,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
            )

        if checkpoint_dir is not None:
            load_experts(experts, checkpoint_dir)

        return cls(
            dht,
            experts,
            listen_on=listen_on,
            num_connection_handlers=num_handlers,
            device=device,
            checkpoint_dir=checkpoint_dir,
            stats_report_interval=stats_report_interval,
            start=start,
        )

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
                self.dht_handler_thread.start()
        if self.checkpoint_saver is not None:
            self.checkpoint_saver.start()

        for process in self.conn_handlers:
            if not process.is_alive():
                process.start()
            process.ready.wait()

        try:
            self.runtime.run()
        finally:
            self.shutdown()

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
        Gracefully terminate the server, process-safe.
        Please note that terminating server otherwise (e.g. by killing processes) may result in zombie processes.
        If you did already cause a zombie outbreak, your only option is to kill them with -9 (SIGKILL).
        """
        self.ready.clear()

        for process in self.conn_handlers:
            process.terminate()
            process.join()
        logger.debug("Connection handlers terminated")

        if self.dht and self.experts:
            self.dht_handler_thread.stop.set()
            self.dht_handler_thread.join()

        if self.checkpoint_saver is not None:
            self.checkpoint_saver.stop.set()
            self.checkpoint_saver.join()

        if self.dht is not None:
            self.dht.shutdown()
            self.dht.join()

        logger.debug(f"Shutting down runtime")

        self.runtime.shutdown()
        logger.info("Server shutdown succesfully")


@contextmanager
def background_server(*args, shutdown_timeout=5, **kwargs) -> Tuple[hivemind.Endpoint, List[Multiaddr]]:
    """A context manager that creates server in a background thread, awaits .ready on entry and shutdowns on exit"""
    pipe, runners_pipe = mp.Pipe(duplex=True)
    runner = mp.Process(target=_server_runner, args=(runners_pipe, *args), kwargs=kwargs)
    try:
        runner.start()
        # once the server is ready, runner will send us
        # either (False, exception) or (True, (server.listen_on, dht_maddrs))
        start_ok, data = pipe.recv()
        if start_ok:
            yield data
            pipe.send("SHUTDOWN")  # on exit from context, send shutdown signal
        else:
            raise RuntimeError(f"Server failed to start: {data}")
    finally:
        runner.join(timeout=shutdown_timeout)
        if runner.is_alive():
            logger.info("Server failed to shutdown gracefully, terminating it the hard way...")
            runner.kill()
            logger.info("Server terminated")


def _server_runner(pipe, *args, **kwargs):
    try:
        server = Server.create(*args, start=True, **kwargs)
    except Exception as e:
        logger.exception(f"Encountered an exception when starting a server: {e}")
        pipe.send((False, f"{type(e).__name__} {e}"))
        return

    try:
        dht_maddrs = server.dht.get_visible_maddrs() if server.dht is not None else None
        pipe.send((True, (server.listen_on, dht_maddrs)))
        pipe.recv()  # wait for shutdown signal

    finally:
        logger.info("Shutting down server...")
        server.shutdown()
        server.join()
        logger.info("Server shut down")
