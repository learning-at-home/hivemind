import contextlib
from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeVar

import torch

from hivemind.averaging import DecentralizedAverager
from hivemind.averaging.control import StepControl
from hivemind.dht import DHT
from hivemind.utils import DHTExpiration, get_logger

logger = get_logger(__name__)


TGradientAverager = TypeVar("TGradientAverager", bound="GradientAverager")
GradientAveragerFactory = Callable[..., TGradientAverager]


class GradientAverager(DecentralizedAverager):
    """
    An auxiliary averaging class that is responsible for accumulating gradients and aggregating them with peers.
    GradientAverager is meant to be used within hivemind.Optimizer, but it can be used standalone (see example below).

    GradientAverager manages three sets of buffers:
    (1) model gradients - the gradients associated with local model parameters by PyTorch (param.grad).
        These tensors are typically stored on device and updated by torch autograd
    (2) gradient accumulators - an [optional] set of buffers where local gradients are accumulated.
      - note: if reuse_grad_buffers is True, the averager will use gradients from parameters as local accumulators,
        which reduces RAM usage but requires the user to avoid calling zero_grad / clip_grad manually
    (3) averaged gradients - gradient buffers that are aggregated in-place with peers, always in host memory

    :param parameters: pytorch parameters for which to aggregate gradients
    :param dht: a DHT instance connected to the rest of the swarm. See hivemind.DHT docs
    :param prefix: a unique DHT key used for matchmaking. E.g. this can be your experiment name with optional suffixes
    :param reuse_grad_buffers: if True, use model's .grad buffers for accumulating gradients over multiple steps.
      This is more memory efficient, but it requires that the user does *not* call zero_grad or clip_by_whatever at all
    :param accumulate_grads_on: if specified, accumulate gradients on this device. By default, this will use the same
      device as model parameters. One can specify a different device (e.g. 'cpu' vs 'cuda') to save device memory at
      the cost of extra time per step. If reuse_grad_buffers is True, this parameter has no effect.
    :param client_mode: if False, this averager will accept incoming requests from other peers.
      if True, the averager will only join existing groups where at least one peer has client_mode=False.
      By default, this flag is copied from DHTNode inside the ``dht`` instance.
    :param warn: if True, warn when the averager did not reset accumulators after use or did not use averaging results
    :param averaged_grads: if provided, it will be used as a set of averagable gradients
    :param kwargs: see DecentralizedAverager keyword arguments for additional parameters


    Example:

    >>> model = SuchModelMuchLayers()
    >>> opt = torch.optim.Adam(model.parameters())
    >>> grad_averager = GradientAverager(model.parameters(), dht=hivemind.DHT(...))
    >>> next_step_time = hivemind.get_dht_time() + 60   # runs global steps every 60 seconds
    >>> next_step_control = None
    >>> while True:
    >>>    # accumulate as many gradients as you can before next_step_time
    >>>    loss = compute_loss(model, batch_size=32)
    >>>    loss.backward()
    >>>    grad_averager.accumulate_grads_(batch_size=32)
    >>>    # [optional] next step in 5 seconds, start looking for peers in advance
    >>>    if next_step_time - hivemind.get_dht_time() <= 5
    >>>        next_step_control = grad_averager.schedule_step(scheduled_time=next_step_time)
    >>>    # aggregate gradients and perform optimizer step
    >>>    if hivemind.get_dht_time() >= next_step_time:
    >>>        grad_averager.step(control=next_step_control)
    >>>        with grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
    >>>            opt.step()  # update model parameters using averaged gradients
    >>>        grad_averager.reset_accumulated_grads_()  # prepare for next step
    >>>        next_step_time = hivemind.get_dht_time() + 60
    >>>        next_step_control = None

    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        dht: DHT,
        prefix: str,
        reuse_grad_buffers: bool = False,
        accumulate_grads_on: Optional[torch.device] = None,
        client_mode: bool = None,
        warn: bool = True,
        averaged_grads: Sequence[torch.Tensor] = (),
        **kwargs,
    ):
        if reuse_grad_buffers and accumulate_grads_on is not None:
            logger.warning("Setting 'accumulate_grads_on' has no effect if reuse_grad_buffers=True")
        client_mode = client_mode if client_mode is not None else dht.client_mode
        self.parameters = tuple(parameters)
        self.reuse_grad_buffers = reuse_grad_buffers
        self.warn = warn
        self.local_samples_accumulated = 0
        self.local_times_accumulated = 0
        self._anchor_batch_size = None
        self._local_accumulators = None
        if not reuse_grad_buffers:
            self._local_accumulators = tuple(
                torch.zeros_like(grad, device=accumulate_grads_on) for grad in self._grads_from_parameters()
            )
        self._accumulators_used_in_step = False
        self._new_averaged_grads = False

        with torch.no_grad():
            if not averaged_grads:
                averaged_grads = tuple(
                    grad.detach().cpu().clone().share_memory_() for grad in self._grads_from_parameters()
                )
            else:
                if any(
                    param_grad.size() != grad.size()
                    for param_grad, grad in zip(self._grads_from_parameters(), averaged_grads)
                ):
                    raise ValueError("Averaged gradients don't have same shape as gradients from parameters")
        super().__init__(averaged_tensors=averaged_grads, dht=dht, prefix=prefix, client_mode=client_mode, **kwargs)

    def _grads_from_parameters(self) -> Iterator[torch.Tensor]:
        """gradient buffers associated with parameters"""
        for param in self.parameters:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            yield param.grad

    @torch.no_grad()
    def _grad_accumulators(self) -> Iterator[torch.Tensor]:
        """averager-based gradient accumulators"""
        assert (self._local_accumulators is None) == self.reuse_grad_buffers
        yield from self._grads_from_parameters() if self.reuse_grad_buffers else self._local_accumulators

    @torch.no_grad()
    def accumulate_grads_(self, batch_size: int):
        """add current gradients to local grad accumulators (if used)"""
        if self._accumulators_used_in_step and self.warn:
            logger.warning(
                "[warn=True] Gradient accumulators were not reset since the last averaging round. Please "
                "call .reset_accumulated_grads_ after every step or use .step(reset_accumulators=True)"
            )
            self._accumulators_used_in_step = False  # warn once per round
        if self._anchor_batch_size is None:
            # remember the first batch size to correctly re-scale gradients if subsequent batches have a different size
            self._anchor_batch_size = batch_size
        self.local_samples_accumulated += batch_size
        self.local_times_accumulated += 1
        if self.reuse_grad_buffers:
            pass  # user is responsible for accumulating gradients in .grad buffers
        else:
            alpha = float(batch_size) / self._anchor_batch_size
            for grad_buf, grad_acc in zip(self._grads_from_parameters(), self._grad_accumulators()):
                grad_acc.add_(grad_buf.to(grad_acc.device), alpha=alpha)

    def schedule_step(self, scheduled_time: Optional[DHTExpiration] = None, **kwargs) -> StepControl:
        """
        Begin matchmaking: look for a group of peers and prepare for averaging gradients at a specified time.

        :param scheduled_time: expected time when to perform all-reduce. Can be changed using control.scheduled_time
        :param kwargs: any additional keyword args from DecentralizedAverager.step, such as gather, allow_retries, etc
        :note: setting weight at this stage is not supported, please leave this parameter as None
        :returns: step_control - a handle that can be passed into GradientAverager.step to use the pre-scheduled group
        :note: in the current implementation, each step_control can only be used in one step.
        """
        assert kwargs.get("weight") is None, "setting weight in schedule_step is not supported"
        return super().step(scheduled_time=scheduled_time, wait=False, require_trigger=True, **kwargs)

    def step(
        self,
        weight: Optional[float] = None,
        reset_accumulators: bool = True,
        control: Optional[StepControl] = None,
        timeout: Optional[float] = None,
        wait: bool = True,
        **kwargs,
    ):
        """
        Average accumulated gradients with peers, optionally load averaged gradients and reset accumulators

        :param weight: overrides the averaging weight; by default, weight equals the number of accumulated samples
        :param reset_accumulators: by default, set local gradient accumulators to zeros after averaging succeeds
        :param control: reuse a pre-arranged group of peers (or a matchmaking in progress) from averager.schedule_step
        :param timeout: if specified, await for averaging round for at most this number of seconds (if wait=True)
        :param wait: if True, await for the step to finish (or fail), otherwise run all-reduce in background
        """
        if control is None:
            control = self.schedule_step(timeout=timeout, **kwargs)
        elif len(kwargs) > 0:
            raise RuntimeError(f"Averaging with a pre-scheduled group, parameters {kwargs} will have no effect")
        assert not control.triggered, f"This {type(control)} instance was already used"
        if self._new_averaged_grads and self.warn:
            logger.warning(
                "[warn=True] Starting new averaging round, but previous round results were not used. "
                "This may be a sign of incorrect optimizer behavior"
            )

        self.load_accumulators_into_averager_()
        self._accumulators_used_in_step = True
        self._new_averaged_grads = True

        control.weight = self.local_samples_accumulated if weight is None else weight
        if reset_accumulators:
            self.reset_accumulated_grads_()
        control.allow_allreduce()

        return control.result(timeout) if wait else control

    @torch.no_grad()
    def load_accumulators_into_averager_(self):
        """load locally accumulated gradients into the averager for aggregation"""
        # divide locally accumulated gradients by the number of times they were accumulated
        grad_scale = (1.0 / self.local_times_accumulated) if self.local_times_accumulated != 0 else 0.0
        with self.get_tensors() as averaged_grads:
            for grad_acc, averaged_grad in zip(self._grad_accumulators(), averaged_grads):
                averaged_grad.copy_(grad_acc, non_blocking=True).mul_(grad_scale)

    @torch.no_grad()
    def reset_accumulated_grads_(self):
        """reset averager-internal gradient accumulators and the denominator"""
        self._accumulators_used_in_step = False
        self.local_samples_accumulated = self.local_times_accumulated = 0
        self._anchor_batch_size = None
        for grad_buf in self._grad_accumulators():
            grad_buf.zero_()

    @contextlib.contextmanager
    @torch.no_grad()
    def use_averaged_gradients(self):
        """Substitute model's main gradients with averaged gradients (does not respect device placement)"""
        self._new_averaged_grads = False
        with self.get_tensors() as averaged_grads:
            assert len(averaged_grads) == len(self.parameters)
            try:
                old_grads = [param.grad for param in self.parameters]
                for param, new_grad in zip(self.parameters, averaged_grads):
                    param.grad = new_grad
                yield averaged_grads
            finally:
                for param, old_grad in zip(self.parameters, old_grads):
                    param.grad = old_grad

    def notify_used_averaged_gradients(self):
        """Notify averager that the results of a previous averaging round are accounted for"""
        self._new_averaged_grads = False
