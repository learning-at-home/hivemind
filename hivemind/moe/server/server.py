from __future__ import annotations

import multiprocessing as mp
import random
import threading
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import torch

from hivemind.dht import DHT
from hivemind.moe.expert_uid import UID_DELIMITER
from hivemind.moe.server.checkpoints import CheckpointSaver, is_directory, load_experts
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.moe.server.dht_handler import DHTHandlerThread, get_experts
from hivemind.moe.server.layers import (
    add_custom_models_from_file,
    name_to_block,
    name_to_input,
    schedule_name_to_scheduler,
)
from hivemind.moe.server.layers.optim import ClippingWrapper
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.moe.server.runtime import Runtime
from hivemind.p2p import PeerInfo
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.logging import get_logger
from hivemind.utils.tensor_descr import DUMMY_BATCH_SIZE, BatchTensorDescriptor

logger = get_logger(__name__)


class Server(threading.Thread):
    """
    Server allows you to host "experts" - pytorch subnetworks that can be accessed remotely by peers.
    After creation, a server should be started: see Server.run or Server.run_in_background.

    A working server does two things:
     - processes incoming forward/backward requests via Runtime (created by the server)
     - publishes updates to expert status every :update_period: seconds

    :type dht: an instance of hivemind.DHT. Server will use DHT for all network interactions.
    :param module_backends: dict{expert uid (str) : ModuleBackend} for all expert hosted by this server.
    :param num_connection_handlers: maximum number of simultaneous requests. Please note that the default value of 1
        if too small for normal functioning, we recommend 4 handlers per expert backend.
    :param update_period: how often will server attempt to publish its state (i.e. experts) to the DHT;
        if dht is None, this parameter is ignored.
    :param expiration: when server declares its experts to the DHT, these entries will expire after this many seconds
    :param start: if True, the server will immediately start as a background thread and returns control after server
        is ready (see .ready below)
    """

    def __init__(
        self,
        dht: DHT,
        module_backends: Dict[str, ModuleBackend],
        num_connection_handlers: int = 1,
        update_period: float = 30,
        expiration: Optional[float] = None,
        start=False,
        checkpoint_dir=None,
        **kwargs,
    ):
        super().__init__()
        self.dht, self.module_backends, self.update_period = dht, module_backends, update_period

        self.conn_handlers = [ConnectionHandler(dht, self.module_backends) for _ in range(num_connection_handlers)]
        if checkpoint_dir is not None:
            self.checkpoint_saver = CheckpointSaver(module_backends, checkpoint_dir, update_period)
        else:
            self.checkpoint_saver = None
        self.runtime = Runtime(self.module_backends, **kwargs)

        if self.module_backends:
            self.dht_handler_thread = DHTHandlerThread(
                module_backends=self.module_backends,
                dht=self.dht,
                update_period=self.update_period,
                expiration=expiration,
                daemon=True,
            )

        if start:
            self.run_in_background(await_ready=True)

    @classmethod
    def create(
        cls,
        num_experts: int = None,
        expert_uids: str = None,
        expert_pattern: str = None,
        expert_cls="ffn",
        hidden_dim=1024,
        optim_cls=torch.optim.Adam,
        scheduler: str = "none",
        num_warmup_steps=None,
        num_training_steps=None,
        clip_grad_norm=None,
        num_handlers=None,
        min_batch_size=1,
        max_batch_size=4096,
        device=None,
        initial_peers=(),
        checkpoint_dir: Optional[Path] = None,
        compression=CompressionType.NONE,
        stats_report_interval: Optional[int] = None,
        custom_module_path=None,
        update_period: float = 30,
        expiration: Optional[float] = None,
        *,
        start: bool,
        **kwargs,
    ) -> Server:
        """
        Instantiate a server with several identical modules. See argparse comments below for details

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
        :param num_training_steps: the total number of steps for LR schedule
        :param clip_grad_norm: maximum gradient norm used for clipping

        :param initial_peers: multiaddrs of one or more active DHT peers (if you want to join an existing DHT)

        :param checkpoint_dir: directory to save and load expert checkpoints

        :param compression: if specified, use this compression to pack all inputs, outputs and gradients by all experts
            hosted on this server. For a more fine-grained compression, start server in python and specify compression
            for each BatchTensorProto in ModuleBackend for the respective experts.

        :param start: if True, starts server right away and returns when server is ready for requests
        :param stats_report_interval: interval between two reports of batch processing performance statistics
        :param kwargs: any other params will be forwarded to DHT upon creation
        """
        if custom_module_path is not None:
            add_custom_models_from_file(custom_module_path)
        assert expert_cls in name_to_block

        dht = DHT(initial_peers=initial_peers, start=True, **kwargs)
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
                expert_uids.extend(_generate_uids(uids_to_generate, expert_pattern, dht))

        num_experts = len(expert_uids)
        num_handlers = num_handlers if num_handlers is not None else num_experts * 8
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        sample_input = name_to_input[expert_cls](DUMMY_BATCH_SIZE, hidden_dim)
        if isinstance(sample_input, tuple):
            args_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in sample_input)
        else:
            args_schema = (BatchTensorDescriptor.from_tensor(sample_input, compression),)

        scheduler_cls = schedule_name_to_scheduler[scheduler]
        if scheduler_cls is not None:
            scheduler_cls = partial(
                scheduler_cls, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
            )

        # initialize experts
        experts = {}
        for expert_uid in expert_uids:
            expert = name_to_block[expert_cls](hidden_dim)
            optimizer = optim_cls(expert.parameters()) if optim_cls is not None else None
            scheduler = scheduler_cls(optimizer) if scheduler_cls is not None else None
            if clip_grad_norm is not None:
                optimizer = ClippingWrapper(optimizer, clip_grad_norm)
            experts[expert_uid] = ModuleBackend(
                name=expert_uid,
                module=expert,
                args_schema=args_schema,
                optimizer=optimizer,
                scheduler=scheduler,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
            )

        if checkpoint_dir is not None:
            load_experts(experts, checkpoint_dir)

        return cls(
            dht,
            experts,
            num_connection_handlers=num_handlers,
            device=device,
            checkpoint_dir=checkpoint_dir,
            stats_report_interval=stats_report_interval,
            update_period=update_period,
            expiration=expiration,
            start=start,
        )

    def run(self):
        """
        Starts Server in the current thread. Initializes dht if necessary, starts connection handlers,
        runs Runtime (self.runtime) to process incoming requests.
        """
        logger.info(f"Server started with {len(self.module_backends)} modules:")
        for expert_name, backend in self.module_backends.items():
            num_parameters = sum(p.numel() for p in backend.module.parameters() if p.requires_grad)
            logger.info(f"{expert_name}: {backend.module.__class__.__name__}, {num_parameters} parameters")

        if not self.dht.is_alive():
            self.dht.run_in_background(await_ready=True)

        if self.module_backends:
            self.dht_handler_thread.start()

        if self.checkpoint_saver is not None:
            self.checkpoint_saver.start()

        for handler in self.conn_handlers:
            handler.run_in_background()

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

        for handler in self.conn_handlers:
            handler.shutdown()
        logger.debug("Connection handlers terminated")

        if self.module_backends:
            self.dht_handler_thread.stop.set()
            self.dht_handler_thread.join()

        if self.checkpoint_saver is not None:
            self.checkpoint_saver.stop.set()
            self.checkpoint_saver.join()

        self.dht.shutdown()

        logger.debug(f"Shutting down runtime")
        self.runtime.shutdown()

        logger.info("Server shutdown succesfully")


@contextmanager
def background_server(*args, shutdown_timeout=5, **kwargs) -> PeerInfo:
    """A context manager that creates server in a background , awaits .ready on entry and shuts down on exit"""
    pipe, runners_pipe = mp.Pipe(duplex=True)
    runner = mp.Process(target=_server_runner, args=(runners_pipe, *args), kwargs=kwargs)
    try:
        runner.start()
        # once the server is ready, runner will send us
        # either (False, exception) or (True, PeerInfo(dht_peer_id, dht_maddrs))
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
        dht_maddrs = server.dht.get_visible_maddrs()
        pipe.send((True, PeerInfo(server.dht.peer_id, dht_maddrs)))
        pipe.recv()  # wait for shutdown signal

    finally:
        logger.info("Shutting down server...")
        server.shutdown()
        server.join()
        logger.info("Server shut down")


def _generate_uids(
    num_experts: int, expert_pattern: Optional[str], dht: Optional[DHT] = None, attempts_per_expert=10
) -> List[str]:
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
            return f"expert{UID_DELIMITER}{attempts_per_expert * num_experts - remaining_attempts}"

        uid = []
        for block in expert_pattern.split(UID_DELIMITER):
            try:
                if "[" not in block and "]" not in block:
                    uid.append(block)
                elif block.startswith("[") and block.endswith("]") and ":" in block:
                    slice_start, slice_end = map(int, block[1:-1].split(":"))
                    uid.append(str(random.randint(slice_start, slice_end - 1)))
                else:
                    raise ValueError("Block must be either fixed or a range [from:to]")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                raise ValueError(f"Expert pattern {expert_pattern} has invalid block {block}, {e}")
        return UID_DELIMITER.join(uid)

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
        if dht is not None:
            existing_expert_uids = {
                found_expert.uid for found_expert in get_experts(dht, new_uids) if found_expert is not None
            }
            new_uids = [new_uid for new_uid in new_uids if new_uid not in existing_expert_uids]

        found_uids += new_uids

    if len(found_uids) != num_experts:
        logger.warning(
            f"Found only {len(found_uids)} out of {num_experts} free expert uids after "
            f"{attempts_per_expert * num_experts} attempts"
        )
    return found_uids
