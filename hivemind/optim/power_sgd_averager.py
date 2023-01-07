import asyncio
import contextlib
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

import torch

from hivemind.averaging.allreduce import AveragingMode
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.compression import CompressionInfo, TensorRole
from hivemind.dht import DHT
from hivemind.optim.grad_averager import GradientAverager
from hivemind.utils import get_logger
from hivemind.utils.asyncio import enter_asynchronously
from hivemind.utils.math import get_flatten_greedy_dims, orthogonalize_

GatheredData = Any
logger = get_logger(__name__)


class AllReducePhases(Enum):
    PHASE_P = 1
    PHASE_Q = 2


class PowerSGDGradientAverager(GradientAverager):
    """
    A gradient averager that implements PowerSGD compression: https://arxiv.org/abs/1905.13727
    For basic properties and guaranties of gradient averagers, please refer to the base class docstring.
    Put simply, this method approximates large gradient tensors (m,n) with a product of two
    smaller matrices (m,r) by (r,n), where r is a parameter chosen by the user (see averager_rank).

    As a result, PowerSGD only needs to aggregate O((m + n) * r) tensors instead of O(m * n).
    High r, e.g. sqrt(max(m, n)) typically reduce communication by 2-8x without affecting convergence.
    Low r, e.g. 1-8, further accelerate communication, but may converge worse depending on the task.

    To maintain convergence with low r, this averager uses the error feedback strategy. Put simply,
    if some part of the gradient is "lost in compression", it will be added to the next iteration.
    This has two implications: (a) it needs more RAM in order to store the "feedback buffers"
    and (b) if devices stay alive only for one step, training with small rank may converge slower.
    This is because error feedback takes multiple steps to kick in.

    Since not all gradients are matrices, PowerSGD views 3d+ tensors via tensor.flatten(1, -1).
    If a tensor has less than 2 dimensions or does not compress efficiently, it will be aggregated
    normally, i.e. without powerSGD. See min_compression_ratio for details.

    :note: due to the above rule, PowerSGD is *not* shape-invariant. For instance, a
     matrix of shape (256, 256) be compressed differently if you .reshape it to (32, 32, 32).

    :param parameters: pytorch parameters for which to aggregate gradients
    :param averager_rank: rank of compressed gradients
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
    :param min_compression_ratio: apply PowerSGD to a tensor only if it reduces communication by at least this factor,
      otherwise aggregate tensors as is
    :param averaged_grads: if provided, it will be used as a set of averagable gradients
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        averager_rank: int,
        *,
        dht: DHT,
        prefix: str,
        reuse_grad_buffers: bool = False,
        accumulate_grads_on: Optional[torch.device] = None,
        client_mode: bool = None,
        warn: bool = True,
        min_compression_ratio: float = 0.5,
        averaged_grads: Optional[Sequence[torch.Tensor]] = None,
        **kwargs,
    ):
        self.rank = averager_rank
        self.parameters = tuple(parameters)
        self._uncompressed_gradients_indexes = set(
            i
            for i, grad in enumerate(self._grads_from_parameters())
            if grad.ndim <= 1
            or (1 - self.rank * sum(get_flatten_greedy_dims(grad)) / grad.numel()) < min_compression_ratio
            # compute how much parameters are left after factorization
        )
        self._ms = [
            torch.zeros_like(grad, device="cpu").share_memory_()
            for idx, grad in enumerate(self._grads_from_parameters())
            if idx not in self._uncompressed_gradients_indexes
        ]
        self._qs = [
            torch.rand((get_flatten_greedy_dims(grad)[1], self.rank), device="cpu").share_memory_()
            for idx, grad in enumerate(self._grads_from_parameters())
            if idx not in self._uncompressed_gradients_indexes
        ]

        super().__init__(
            self.parameters,
            dht=dht,
            prefix=prefix,
            reuse_grad_buffers=reuse_grad_buffers,
            accumulate_grads_on=accumulate_grads_on,
            client_mode=client_mode,
            warn=warn,
            averaged_grads=averaged_grads,
            **kwargs,
        )

    @contextlib.contextmanager
    def _register_allreduce_group(self, group_info: GroupInfo):
        """Register a given group for one or more all-reduce rounds"""
        try:
            for phase in list(AllReducePhases):
                self._running_groups[group_info.group_id + phase.name.encode()] = asyncio.Future()
            self._pending_groups_registered.set()
            yield
        finally:
            for phase in list(AllReducePhases):
                maybe_future = self._running_groups.pop(group_info.group_id + phase.name.encode(), None)
                if maybe_future and not maybe_future.done():
                    logger.warning(f"All-reduce group {group_info.group_id + phase.name.encode()} did not finish.")
            self._pending_groups_registered.set()

    async def _aggregate_with_group(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run aggregation in a given group and update tensors in place, return gathered metadata"""
        try:
            bandwidths, mode_ids, user_gathered_bytes = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)))
            modes = tuple(map(AveragingMode, mode_ids))

            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            )

            async with enter_asynchronously(self.get_tensors()) as averaged_grads:
                averaged_grads_via_sgd = [
                    grad for idx, grad in enumerate(averaged_grads) if idx not in self._uncompressed_gradients_indexes
                ]
                for grad, m in zip(averaged_grads_via_sgd, self._ms):
                    m.add_(grad.to(m.device))

                ps = [
                    torch.zeros((get_flatten_greedy_dims(grad)[0], self.rank), device="cpu")
                    for idx, grad in enumerate(averaged_grads_via_sgd)
                ]
                for p, q, m in zip(ps, self._qs, self._ms):
                    # we use reshape for all matrixes because PowerSGD works only with 2d tensors
                    torch.matmul(m.reshape(-1, q.size(0)), q, out=p)

                p_group_id = group_info.group_id + AllReducePhases.PHASE_P.name.encode()
                q_groud_id = group_info.group_id + AllReducePhases.PHASE_Q.name.encode()

                await self._run_allreduce_inplace_(ps, group_info, p_group_id, peer_fractions=peer_fractions, **kwargs)

                for p in ps:
                    orthogonalize_(p)

                for p, q, m in zip(ps, self._qs, self._ms):
                    torch.matmul(m.reshape(-1, q.size(0)).t(), p, out=q)

                phase_q_tensors = self._qs + [
                    grad for idx, grad in enumerate(averaged_grads) if idx in self._uncompressed_gradients_indexes
                ]

                await self._run_allreduce_inplace_(
                    phase_q_tensors, group_info, q_groud_id, peer_fractions=peer_fractions, **kwargs
                )

                for p, q, m, grad in zip(ps, self._qs, self._ms, averaged_grads_via_sgd):
                    new_m = torch.matmul(p, q.t()).reshape(m.size())
                    m.sub_(new_m)
                    grad.copy_(new_m)

                return user_gathered
        except BaseException as e:
            logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")

    def get_current_state(self):
        """
        Get current gradient averager state and when requested by a newbie peer.
        """
        with torch.no_grad(), self.lock_averaged_tensors:
            grad_averager_buffers = [q for q in self._qs]
            grad_averager_buffers_infos = [
                CompressionInfo.from_tensor(buffer, key=f"buffer_q_{key}", role=TensorRole.GRADIENT)
                for buffer, key in zip(grad_averager_buffers, enumerate(grad_averager_buffers))
            ]

        metadata = dict(group_bits=self.get_group_bits())
        return metadata, grad_averager_buffers, grad_averager_buffers_infos

    def load_state_from_peers(self, **kwargs):
        """
        Attempt to download the latest optimizer state from peers and update gradient averager buffers.
        :returns: whether or the averager succeeded in loading parameters
        """
        loaded_state = super().load_state_from_peers(**kwargs)
        if loaded_state is None:
            return

        metadata, flat_tensors = loaded_state
        logger.info("Starting loading gradient averager buffers from peers")

        if len(flat_tensors) != len(self._qs):
            logger.error("Failed to load state from peer, received invalid parameters, extras or metadata")
            return

        with torch.no_grad(), self.lock_averaged_tensors:
            for local_q, loaded_q in zip(self._qs, flat_tensors):
                local_q.copy_(loaded_q, non_blocking=True)
