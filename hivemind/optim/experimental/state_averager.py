""" An extension of averager that supports common optimization use cases. """
import logging
from asyncio import Future
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from threading import Event
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Sequence, Tuple, Union

import torch

import hivemind
from hivemind.averaging import DecentralizedAverager
from hivemind.compression import CompressionInfo, TensorRole
from hivemind.optim.grad_scaler import GradScaler
from hivemind.utils import get_logger, nested_flatten, nested_pack

logger = get_logger(__name__)


Parameters = Iterable[torch.Tensor]
ParamGroups = Iterable[Dict[str, Any]]
TorchOptimizer = torch.optim.Optimizer
LRSchedulerBase = getattr(torch.optim.lr_scheduler, "_LRScheduler", None)
OptimizerFactory = Callable[[Union[Parameters, ParamGroups]], TorchOptimizer]
SchedulerFactory = Callable[[TorchOptimizer], LRSchedulerBase]


class TrainingStateAverager(DecentralizedAverager):
    """
    An auxiliary class that holds peer's training state, including model parameters, optimizer statistics, scheduler
    and any other variables that define the local training state (e.g. batchnorm moving averages).
    TrainingStateAveraager is intended to keep these parameters weakly synchronized across the swarm.

    The intended use is to call .step(optimizer_step=..., averaging_round=...) periodically, e.g. after every batch.
    If peer gets out of sync with the swarm, one should call state_averager.load_state_from_peers() to re-synchronize.

    Example:

    >>> avgr = TrainingStateAverager(optimizer=torch.optim.Adam, params=model.parameters(), ...)
    >>> # alternative interface: TrainingStateAverager(optimizer=torch.optim.Adam(model.parameters()), ...)
    >>> avgr.load_state_from_peers()
    >>> for i, batch in enumerate(training_dataloader):
    >>>     loss = compute_loss(model, batch)
    >>>     loss.backward()
    >>>     avgr.step(optimizer_step=i % 10 == 0, averaging_round=is_it_time_for_averaging(), delay_averaging=True)

    :note: when using delay_averaging or delay_optimizer_step, calling optimizer directly is not recommended because
      it may overlap with delayed updates from a background thread with unpredictable results. Instead, please call
      TrainingStateAverager.step(..., optimizer_step=True)

    :param optimizer: PyTorch Optimizer or a callable that creates a optimizer from param groups
    :param params: optional, a list/tuple of parameters or structured param groups for the optimizer
    :param scheduler: optional learning rate scheduler or callable that creates one from optimizer instance
    :note: if provided, scheduler will be updated based on averager.local_epoch, not the number of step cycles
    :param initialize_optimizer: if True, run a speculative optimizer step with zero gradients to initialize all
      state tensors. If False, user must make sure that all tensors are pre-initialized at init.
      By default, initialize optimizer unless it already has some state tensors to begin with.
    :param offload_optimizer: if True, create optimizer on top of averaged parameters which may save device memory.
    :param custom_gradients: if True, do *not* automatically load local gradients into the offloaded optimizer.
      This assumes that offloaded gradients will be populated externally, e.g. by the user or by hivemind.Optimizer.
    :param reuse_tensors: if True, reuse parameters and optimizer statistics as averaged_tensors for allreduce.
      For this to work, all parameters must be on CPU and have the appropriate dtype for use in DecentralizedAverager
    :param sync_epoch_when_averaging: if True, update local epoch to the latest epoch among averaging peers
    :param parameter_names: optionally provide parameter names in the same order as in params
    :param average_opt_statistics: names of optimizer statistics from state dict that should be averaged with peers
    :param extra_tensors: if specified, these extra tensors will also be averaged and shared in load_state_from_peers.
    :note: you can use extra_tensors to for any tensors not used by the optimizer (e.g. batchnorm statistics)
    :param kwargs: any additional parameters will be forwarded to DecentralizedAverager
    """

    def __init__(
        self,
        *,
        dht: hivemind.DHT,
        optimizer: Union[TorchOptimizer, OptimizerFactory],
        params: Optional[Union[Parameters, ParamGroups]] = None,
        scheduler: Optional[Union[LRSchedulerBase, SchedulerFactory]] = None,
        initialize_optimizer: Optional[bool] = None,
        offload_optimizer: bool = False,
        custom_gradients: bool = False,
        reuse_tensors: bool = False,
        sync_epoch_when_averaging: bool = False,
        parameter_names: Optional[Sequence[str]] = None,
        average_opt_statistics: Sequence[str] = (),
        extra_tensors: Sequence[torch.Tensor] = (),
        status_loglevel: int = logging.DEBUG,
        **kwargs,
    ):
        average_opt_statistics = tuple(average_opt_statistics)
        assert all(isinstance(key, str) for key in average_opt_statistics)
        if offload_optimizer and reuse_tensors:
            logger.warning("Setting offload_optimizer=True has no effect because reuse_parameters=True")
        if custom_gradients and not offload_optimizer:
            logger.warning("Setting custom_gradients=True has no effect because the optimizer is not offloaded")

        params_groups, main_parameters, parameter_names = self._check_params(optimizer, params, parameter_names)

        self.status_loglevel = status_loglevel
        self.reuse_tensors = reuse_tensors
        self.offload_optimizer = offload_optimizer
        self.custom_gradients = custom_gradients

        self.main_parameters, self.parameter_names = main_parameters, parameter_names
        self._averaged_parameters = tuple(map(self._make_host_tensor, main_parameters))
        self.optimizer, self.scheduler = self._init_components(
            params_groups, optimizer, scheduler, initialize_optimizer
        )
        self.opt_keys_for_averaging, self.extra_tensors = average_opt_statistics, extra_tensors
        self.sync_epoch_when_averaging = sync_epoch_when_averaging
        self.local_epoch = 0

        self.step_executor = ThreadPoolExecutor(max_workers=1)
        self.finished_optimizer_step = Event()
        self.finished_averaging_round = Event()
        self.pending_update = Future()
        self.pending_update.set_result(None)

        super().__init__(
            dht=dht, averaged_tensors=self._init_averaged_tensors(), tensor_infos=self._init_tensor_infos(), **kwargs
        )

    @staticmethod
    def _check_params(
        optimizer: Union[TorchOptimizer, OptimizerFactory],
        param_groups: Optional[Union[Parameters, ParamGroups]],
        parameter_names: Optional[Sequence[str]],
    ) -> Tuple[ParamGroups, Sequence[torch.Tensor], Sequence[str]]:
        """Get and verify parameters, groups and names"""
        if param_groups is None:
            assert hasattr(optimizer, "param_groups"), "Must provide param_groups or an optimizer with .param_groups"
            param_groups = optimizer.param_groups
        param_groups = tuple(param_groups)
        if all(isinstance(p, torch.Tensor) for p in param_groups):
            param_groups = (dict(params=param_groups),)
        for group in param_groups:
            assert isinstance(group, dict) and group.get("params") is not None
            assert all(isinstance(p, torch.Tensor) for p in group["params"])
        parameters = tuple(chain(*(group["params"] for group in param_groups)))
        if parameter_names is None:
            parameter_names = tuple(i for i in range(len(parameters)))
        parameter_names = tuple(nested_flatten(parameter_names))
        assert len(parameters) == len(parameter_names), f"Expected {len(parameters)} names, got {len(parameter_names)}"
        assert len(set(parameters)) == len(parameters), "Found duplicate parameters in param_groups"
        return param_groups, parameters, parameter_names

    def _make_host_tensor(self, source_tensor: torch.Tensor) -> torch.Tensor:
        """Create a new tensor for averaging or reuse the existing one"""
        if self.reuse_tensors:
            assert source_tensor.device == torch.device("cpu") and source_tensor.dtype == torch.float32
            if not source_tensor.is_shared():
                source_tensor.share_memory_()
            return source_tensor
        else:
            averaged_tensor = source_tensor.detach().to(device="cpu", dtype=torch.float32, copy=True)
            return averaged_tensor.share_memory_().requires_grad_(source_tensor.requires_grad)

    def _init_components(
        self,
        param_groups: ParamGroups,
        optimizer_or_factory: Union[TorchOptimizer, OptimizerFactory],
        scheduler_or_factory: Optional[Union[LRSchedulerBase, SchedulerFactory]],
        initialize_optimizer: Optional[bool],
    ) -> Tuple[TorchOptimizer, Optional[LRSchedulerBase]]:
        """Get optimizer and scheduler by either instantiating user-provided factory or using pre-instantiated ones"""
        assert hasattr(self, "_averaged_parameters"), "Internal error: must initialize averaged parameters first"
        optimizer_is_factory = callable(optimizer_or_factory) and not isinstance(optimizer_or_factory, TorchOptimizer)
        scheduler_is_factory = callable(scheduler_or_factory) and not isinstance(scheduler_or_factory, LRSchedulerBase)
        if optimizer_is_factory and not scheduler_is_factory and scheduler_or_factory is not None:
            raise ValueError("If optimizer is created internally, scheduler must also be initialized internally")
        if self.offload_optimizer and not optimizer_is_factory:
            raise ValueError("Using offload_optimizer requires creating optimizer inside hivemind")

        # create optimizer
        if optimizer_is_factory:
            if self.offload_optimizer:
                for param in self._averaged_parameters:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)

                next_index = 0
                param_groups_for_optimizer = []
                for param_group in param_groups:
                    num_params = len(param_group["params"])
                    averaged_params_for_group = self._averaged_parameters[next_index : next_index + num_params]
                    param_groups_for_optimizer.append(dict(param_group, params=averaged_params_for_group))
                    next_index += num_params
                assert next_index == len(self._averaged_parameters)

            else:
                param_groups_for_optimizer = param_groups
            optimizer = optimizer_or_factory(param_groups_for_optimizer)
        else:
            optimizer = optimizer_or_factory

        # optionally initialize optimizer state dict
        if initialize_optimizer is None:
            initialize_optimizer = not any(isinstance(x, torch.Tensor) for x in nested_flatten(optimizer.state_dict()))
            logger.log(
                self.status_loglevel,
                "Initializing optimizer manually since it has no tensors in state dict. "
                "To override this, please provide initialize_optimizer=False",
            )

        if initialize_optimizer:
            initialize_optimizer_state_(optimizer)  # note: this will run one optimizer step!

        # create LR scheduler
        if scheduler_is_factory:
            assert callable(scheduler_or_factory)
            scheduler = scheduler_or_factory(optimizer)
        else:
            scheduler = scheduler_or_factory

        # verify optimizer and scheduler
        assert isinstance(optimizer, TorchOptimizer) and len(optimizer.param_groups) == len(list(param_groups))
        if self.offload_optimizer or self.reuse_tensors:
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    assert param.is_shared()
        assert isinstance(scheduler, (LRSchedulerBase, type(None)))
        if scheduler is not None:
            assert scheduler.optimizer == optimizer
        return optimizer, scheduler

    def _local_tensors(self) -> Iterator[torch.Tensor]:
        """Iterate local trainer's tensors that should be averaged with peers"""
        for param_group in self.optimizer.param_groups:
            yield from param_group["params"]
        for stats in self.opt_keys_for_averaging:
            for param_group in self.optimizer.param_groups:
                for param in param_group["params"]:
                    yield self.optimizer.state[param][stats]
        yield from self.extra_tensors

    @torch.no_grad()
    def _init_averaged_tensors(self) -> Sequence[torch.Tensor]:
        """Create or reuse a tuple of all averaged tensors, including parameters, optimizer statistics and extras"""
        assert hasattr(self, "optimizer"), "Optimizer should already be initialized by this point"
        assert hasattr(self, "_averaged_parameters"), "Should initialize _averaged_parameters first"
        assert not hasattr(self, "_averaged_tensors"), "Averager is already initialized"
        assert all(isinstance(key, str) for key in self.opt_keys_for_averaging)

        local_tensors = tuple(self._local_tensors())
        local_non_parameters = local_tensors[len(self._averaged_parameters) :]
        averaged_tensors = tuple(map(torch.Tensor.detach, self._averaged_parameters))
        averaged_non_parameters = tuple(map(self._make_host_tensor, local_non_parameters))
        averaged_tensors = tuple(chain(averaged_tensors, averaged_non_parameters))

        assert len(averaged_tensors) == len(local_tensors)
        for local_tensor, averaged_tensor in zip(local_tensors, averaged_tensors):
            assert local_tensor.shape == averaged_tensor.shape
            if averaged_tensor.grad is not None:
                logger.debug(self.status_loglevel, "setting gradients for averaged tensor to None")

        return averaged_tensors

    def _init_tensor_infos(self) -> Sequence[CompressionInfo]:
        """Get CompressionInfo for each state tensor, accounting for its role and specification"""
        tensor_infos = []
        for param, param_name in zip(self.main_parameters, self.parameter_names):
            tensor_infos.append(CompressionInfo.from_tensor(param, key=param_name, role=TensorRole.PARAMETER))
        for stats_name in self.opt_keys_for_averaging:
            opt_parameters = [param for group in self.optimizer.param_groups for param in group["params"]]
            assert len(opt_parameters) == len(self.parameter_names)
            for param, param_name in zip(opt_parameters, self.parameter_names):
                tensor_infos.append(
                    CompressionInfo.from_tensor(
                        self.optimizer.state[param][stats_name],
                        key=(param_name, stats_name),
                        role=TensorRole.OPTIMIZER,
                    )
                )
        for i, extra_tensor in enumerate(self.extra_tensors):
            tensor_infos.append(CompressionInfo.from_tensor(extra_tensor, key=i, role=TensorRole.UNSPECIFIED))
        return tuple(tensor_infos)

    def step(
        self,
        wait_for_delayed_update: bool = None,
        apply_delayed_updates: bool = True,
        increment_epoch: bool = False,
        optimizer_step: bool = False,
        zero_grad: bool = False,
        delay_optimizer_step: bool = False,
        averaging_round: bool = False,
        delay_averaging: Optional[bool] = None,
        grad_scaler: Optional[GradScaler] = None,
        averaging_opts: Optional[Dict[str, Any]] = None,
    ):
        """
        Perform one or several possible actions, depending on the specified keyword args.
        The actions will be performed in the same order as specified below:

        :param wait_for_delayed_update: if there are background averaging rounds, wait for them to finish
          by default, await delayed updates when scheduling the next optimizer step, otherwise do not update
        :param apply_delayed_updates: apply any averaging rounds that have finished but were not applied yet
        :param increment_epoch: increment .local_epoch and update the learning rate scheduler (if present)
        :param optimizer_step: perform a single optimizer step and update local parameters (without changing scheduler)
        :param zero_grad: if True, reset local gradients after performing optimizer step
        :param delay_optimizer_step: if True, run optimizer step in background and apply results in a future step
        :param averaging_round: average parameters, chosen optimizer keys and extra tensors with a group of peers
        :param grad_scaler: when using hivemind.GradScaler, one must forward it to step after calling .unscale_
        :param delay_averaging: if True, perform averaging in background and apply results in a future step
          by default, delay averaging if the optimizer step is also delayed. Set to true to delay only this phase.
        :param averaging_opts: a dict of keyword arguments forwarded into averaging round
        """
        if delay_averaging is None:
            delay_averaging = delay_optimizer_step
        if wait_for_delayed_update is None:
            wait_for_delayed_update = optimizer_step or zero_grad or averaging_round
        assert not delay_optimizer_step or delay_averaging, "Delayed optimizer step requires delayed averaging"
        if optimizer_step or averaging_round or zero_grad:
            assert wait_for_delayed_update, "Must wait for background updates to finish before scheduling new ones"
        if delay_optimizer_step:
            assert self.offload_optimizer, "Delayed optimizer step is only available with offload_optimizer"
            assert not averaging_round or delay_averaging, "Averaging after delayed optimizer should also be delayed"
        if averaging_opts and not averaging_round:
            logger.warning(f"Averaging parameters not used because averaging_round=False: {averaging_opts}")
        output = None

        if wait_for_delayed_update:
            if not self.pending_update.done():
                logger.log(self.status_loglevel, "Waiting for delayed updates to finish...")
                output = self.pending_update.result()

        if self.pending_update.done() and self.pending_update.exception():
            logger.warning(f"Background update failed with {self.pending_update.exception()} and will be ignored")

        if apply_delayed_updates:
            if self.finished_averaging_round.is_set():
                if not self.reuse_tensors:
                    self._apply_averaging_results_()
                logger.log(self.status_loglevel, "Received parameters from background averaging round")
                self.finished_averaging_round.clear()

            if self.finished_optimizer_step.is_set():
                if self.offload_optimizer:
                    self._apply_optimizer_results_()
                logger.log(self.status_loglevel, "Received parameters from background optimizer step")
                self.finished_optimizer_step.clear()

        if increment_epoch:
            self.local_epoch += 1

        if optimizer_step or zero_grad or averaging_round:
            assert self.pending_update.done(), "Tried to perform a new update but previous update is still running"

            if self.offload_optimizer and not self.custom_gradients:
                self._load_local_grads_into_optimizer_()

            self.pending_update = self.step_executor.submit(
                self._do,
                optimizer_step,
                zero_grad,
                averaging_round,
                grad_scaler,
                **averaging_opts or {},
            )

            if (optimizer_step or zero_grad) and not delay_optimizer_step:
                self.finished_optimizer_step.wait()
                self.finished_optimizer_step.clear()
                if self.offload_optimizer:
                    self._apply_optimizer_results_()
                logger.log(self.status_loglevel, "Finished optimizer step")

            if averaging_round and not delay_averaging:
                self.finished_averaging_round.wait()
                self.finished_averaging_round.clear()
                if not self.reuse_tensors:
                    self._apply_averaging_results_()
                logger.log(self.status_loglevel, "Finished averaging round")

            if not delay_averaging:
                try:
                    output = self.pending_update.result()
                finally:
                    self.finished_averaging_round.clear()
                    self.finished_optimizer_step.clear()
        return output

    def _do(
        self,
        optimizer_step: bool,
        zero_grad: bool,
        averaging_round: bool,
        grad_scaler: Optional[GradScaler],
        **kwargs,
    ):
        """
        Run the optimizer step, followed by a scheduler step and an averaging round, each stage is optional.
        This method is meant to be called in the background executor.
        """
        try:
            if optimizer_step:
                logger.log(self.status_loglevel, f"Running optimizer step")
                if grad_scaler is None:
                    self.optimizer.step()
                else:
                    with grad_scaler.running_global_step():
                        assert grad_scaler.step(self.optimizer)
            self._update_scheduler()

            if zero_grad:
                logger.log(self.status_loglevel, f"Running zero grad")
                self.optimizer.zero_grad()
                if self.offload_optimizer:
                    for parameter in self.main_parameters:
                        if parameter.grad is not None:
                            parameter.grad.zero_()

            self.finished_optimizer_step.set()

            if averaging_round:
                if not self.reuse_tensors:
                    self._load_local_tensors_into_averager_()
                try:
                    gathered = super().step(gather=self.local_epoch, **kwargs)
                    logger.log(self.status_loglevel, f"Averaged parameters with {len(gathered)} peers")
                except BaseException as e:
                    logger.log(self.status_loglevel, f"Averaging failed with {type(e)}")
                    self.finished_averaging_round.set()
                    gathered = {}

                self.finished_averaging_round.set()

                if self.sync_epoch_when_averaging:
                    old_epoch = self.local_epoch
                    for peer_epoch in gathered.values():
                        self.local_epoch = max(self.local_epoch, peer_epoch)
                    if self.local_epoch != old_epoch:
                        logger.log(self.status_loglevel, f"Found peer with newer epoch ({self.local_epoch})")
                        self._update_scheduler()

        except Exception as e:
            logger.exception(e)
            self.finished_optimizer_step.set()
            self.finished_averaging_round.set()

    @torch.no_grad()
    def _load_local_grads_into_optimizer_(self):
        """Copy local gradients into the gradient buffers of the offloaded optimizer"""
        assert self.offload_optimizer, "Loading into offloaded optimizer requires using offloaded optimizer"
        opt_parameters = [param for group in self.optimizer.param_groups for param in group["params"]]
        for main_param, opt_param in zip(self.main_parameters, opt_parameters):
            if main_param.grad is not None:
                opt_param.grad.copy_(main_param.grad, non_blocking=True)

    @torch.no_grad()
    def _apply_optimizer_results_(self):
        """Copy parameters from offloaded optimizer to the main model"""
        assert self.offload_optimizer, "Applying offloaded optimizer updates requires offloaded optimizer"
        with self.lock_averaged_tensors:
            offloaded_parameters = [param for group in self.optimizer.param_groups for param in group["params"]]
            assert len(offloaded_parameters) == len(
                self.main_parameters
            ), "Optimizer parameters changed during training"
            for main_param, offloaded_param in zip(self.main_parameters, offloaded_parameters):
                main_param.copy_(offloaded_param, non_blocking=True)

    @torch.no_grad()
    def _load_local_tensors_into_averager_(self):
        """Copy local tensors into the averaging buffers"""
        assert not self.reuse_tensors, "No need to load tensors into averager: both tensors share the same memory"
        with self.get_tensors() as averaged_tensors:
            for local_tensor, averaged_tensor in zip(self._local_tensors(), averaged_tensors):
                averaged_tensor.copy_(local_tensor, non_blocking=True)

    @torch.no_grad()
    def _apply_averaging_results_(self):
        """Copy averaged tensors into their respective local tensors"""
        assert not self.reuse_tensors, "No need to update averaged tensors since they reuse the same memory"
        with self.get_tensors() as averaged_tensors:
            local_tensors = list(self._local_tensors())
            assert len(local_tensors) == len(averaged_tensors), "Tensor structure changed during training"
            for local_tensor, averaged_tensor in zip(local_tensors, averaged_tensors):
                local_tensor.copy_(averaged_tensor, non_blocking=True)

    def get_current_state(self):
        """
        Get current model/optimizer state and when requested by a newbie peer. executed in the host process.
        :returns: a tuple of (serializable_small_metadata, sequence of torch tensors)
        """
        with torch.no_grad():
            optimized_parameters = tuple(
                param.detach().cpu() for param_group in self.optimizer.param_groups for param in param_group["params"]
            )
            parameter_infos = [
                CompressionInfo.from_tensor(param, key=key, role=TensorRole.PARAMETER)
                for param, key in zip(optimized_parameters, self.parameter_names)
            ]
            extra_tensors = tuple(tensor.detach().cpu() for tensor in self.extra_tensors)
            extra_infos = [
                CompressionInfo.from_tensor(extra_tensor, key=i, role=TensorRole.UNSPECIFIED)
                for i, extra_tensor in enumerate(extra_tensors)
            ]
            optimizer_metadata, optimizer_tensors = dump_optimizer_state(self.optimizer)
            optimizer_infos = [
                CompressionInfo.from_tensor(opt_tensor, key=i, role=TensorRole.OPTIMIZER)
                for i, opt_tensor in enumerate(optimizer_tensors)
            ]

        metadata = dict(
            epoch=self.local_epoch, group_bits=self.get_group_bits(), optimizer_metadata=optimizer_metadata
        )
        all_tensors = list(chain(optimized_parameters, extra_tensors, optimizer_tensors))
        all_tensor_infos = list(chain(parameter_infos, extra_infos, optimizer_infos))
        return metadata, all_tensors, all_tensor_infos

    def load_state_from_peers(self, **kwargs):
        """
        Attempt to download the latest optimizer state from peers and update trainer parameters/statistics.
        :returns: whether or the averager succeeded in loading parameters
        """
        parameters_and_extras = tuple(chain(self.main_parameters, self.extra_tensors))
        num_parameters_and_extras = len(parameters_and_extras)

        loaded_state = super().load_state_from_peers(**kwargs)
        if loaded_state is None:
            return

        metadata, flat_tensors = loaded_state
        if (not isinstance(metadata.get("epoch"), int)) or metadata["epoch"] < self.local_epoch:
            logger.warning("Cowardly refusing to load state from peer: peer's epoch is behind our local epoch")
            return

        loaded_parameters_and_extras = flat_tensors[:num_parameters_and_extras]
        loaded_opt_tensors = flat_tensors[num_parameters_and_extras:]
        if num_parameters_and_extras != len(loaded_parameters_and_extras):
            logger.error("Failed to load state from peer, received parameters, extras or metadata.")
            return

        try:
            load_optimizer_state(self.optimizer, metadata["optimizer_metadata"], loaded_opt_tensors)
        except StopIteration:
            logger.warning("Failed to load state from peer, received inconsistent number of optimizer statistics")
            return

        with torch.no_grad():
            for local_param, loaded_param in zip(parameters_and_extras, loaded_parameters_and_extras):
                local_param.copy_(loaded_param, non_blocking=True)
        self.local_epoch = metadata["epoch"]
        self._update_scheduler()

    def _update_scheduler(self):
        """Increase the scheduler state until it becomes synchronized with local epoch"""
        if self.scheduler:
            while self.scheduler._step_count <= self.local_epoch:
                self.scheduler.step()


def initialize_optimizer_state_(opt: torch.optim.Optimizer):
    """Initialize optimizer statistics by running a virtual optimizer step with zero gradients"""
    flat_params = tuple(param for group in opt.param_groups for param in group["params"])
    old_grads = []
    for param in flat_params:
        old_grads.append(param.grad)
        param.grad = torch.zeros_like(param)
    opt.step()
    for param, old_grad in zip(flat_params, old_grads):
        param.grad = old_grad


def dump_optimizer_state(opt: torch.optim.Optimizer):
    """Convert optimizer state into a format of DecentralizedAverager's get_current_state/load_state_from_peers"""
    with torch.no_grad():
        flat_metadata, flat_tensors = [], []
        for elem in nested_flatten(opt.state_dict()):
            if isinstance(elem, torch.Tensor):
                flat_metadata.append(dict(type="tensor", index=len(flat_tensors)))
                flat_tensors.append(elem.cpu())
            else:
                flat_metadata.append(dict(type="value", value=elem))
        return flat_metadata, flat_tensors


def load_optimizer_state(optimizer: torch.optim.Optimizer, flat_metadata: Dict, flat_tensors: Sequence[torch.Tensor]):
    """Load a state obtained by dump_optimizer_state back into the optimizer"""
    flat_optimizer_state = []
    for elem in flat_metadata:
        if elem.get("type") == "tensor" and isinstance(elem.get("index"), int):
            flat_optimizer_state.append(flat_tensors[elem["index"]])
        elif elem.get("type") == "value" and "value" in elem:
            flat_optimizer_state.append(elem["value"])
    return optimizer.load_state_dict(nested_pack(flat_optimizer_state, structure=optimizer.state_dict()))
