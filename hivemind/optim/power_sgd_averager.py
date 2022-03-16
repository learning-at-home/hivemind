import asyncio
import contextlib
import math
import multiprocessing as mp
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch

from hivemind.averaging.allreduce import AllReduceRunner, AveragingMode
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import Matchmaking, MatchmakingException
from hivemind.compression import CompressionInfo, TensorRole
from hivemind.dht import DHT
from hivemind.p2p import P2P
from hivemind.utils import get_logger
from hivemind.utils.asyncio import as_aiter, azip, enter_asynchronously
from hivemind.utils.math import get_flatten_greedy_dims, orthogonalize_
from hivemind.utils.timed_storage import get_dht_time

from .grad_averager import GradientAverager

GatheredData = Any
logger = get_logger(__name__)


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
    This is because error feedback takes multiple step to kick in.

    Since not all gradients are matrices, PowerSGD views 3d+ tensors via tensor.flatten(1, -1).
    If a tensor has less than 2 dimensions or does not compress efficiently, it will be aggregated
    normally, i.e. without powerSGD. See min_compression_ratio for details.

    :note: due to the above rule, PowerSGD is *not* shape-invariant. For instance, a
     matrix of shape (256, 256) be compressed differently if you .reshape it to (32, 32, 32).

    :param parameters: pytorch parameters for which to aggregate gradients
    :param averager_rank: compress gradient tensors
    :param min_comprasion_ratio: apply PowerSGD to a tensor only if it reduces communication by at least this factor, otherwise aggregate tensors as is
    :param dht: a DHT isntance connected to the rest of the swarm. See hivemind.DHT docs
    :param prefix: a unique DHT key used for matchmaking. E.g. this can be your experiment name with optional suffixes
    :param reuse_grad_buffers: if True, use model's .grad buffers for accumulating gradients over multiple steps.
      This is more memory efficient, but it requires that the user does *not* call zero_grad or clip_by_whatever at all
    :param accumulate_grads_on: if specified, accumulate gradients on this device. By default, this will use the same
      device as model parameters. One can specify a different device (e.g. 'cpu' vs 'cuda') to save device memory at
      the cost of extra time per step. If reuse_grad_buffers is True, this parameter has no effect.
    :param client_mode: if False, this averager will accept incoming requests from other peers.
      if True, the averager will only join existing groups where at least one peer has client_mode=False.
      By default, this flag is copied from DHTNode inside the ``dht`` instance.
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
            if len(tuple(grad.size())) <= 1
            or (
                1 - self.rank * (grad.size(0) + np.prod(grad.size()[1:])) / np.prod(grad.size())
                < min_compression_ratio
            )  # compute how much parameters can we left via factorization
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

        self.all_reduce_phases = (b".phase_p", b".phase_q")

        super().__init__(
            self.parameters,
            dht=dht,
            prefix=prefix,
            reuse_grad_buffers=reuse_grad_buffers,
            accumulate_grads_on=accumulate_grads_on,
            client_mode=client_mode,
            warn=warn,
            averaged_grads=None,
            **kwargs,
        )

    @contextlib.contextmanager
    def _register_allreduce_group(self, group_info: GroupInfo):
        """registers a given all-reduce runner to listen for incoming connections"""
        try:
            for phase in self.all_reduce_phases:
                self._running_groups[group_info.group_id + phase] = asyncio.Future()
            self._pending_groups_registered.set()
            yield
        finally:
            for phase in self.all_reduce_phases:
                maybe_future = self._running_groups.pop(group_info.group_id + phase, None)
                if maybe_future and not maybe_future.done():
                    logger.warning(f"All-reduce group {group_info.group_id + phase} did not finish.")
            self._pending_groups_registered.set()

    async def _run_allreduce(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run All-Reduce in a given group and update tensors in place, return gathered metadata"""
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
                # make this two pairs list for better mapping between m buffers and gradients
                averaged_grads_via_sgd = [
                    grad for idx, grad in enumerate(averaged_grads) if idx not in self._uncompressed_gradients_indexes
                ]
                for grad, m in zip(averaged_grads_via_sgd, self._ms):
                    m.add_(grad.to(m.device))

                ps = [
                    torch.zeros((get_flatten_greedy_dims(grad)[0], self.rank), device="cpu")
                    for idx, grad in enumerate(averaged_grad_via_sgd)
                ]
                for p, q, m in zip(ps, self._qs, self._ms):
                    # we use reshape for all matrixes because sgd works only with 2d tensors
                    torch.matmul(m.reshape(-1, q.size(0)), q, out=p)

                allreduce_p_phase = AllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id + self.all_reduce_phases[0],
                    tensors=ps,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    gathered=user_gathered,
                    modes=modes,
                    **kwargs,
                )
                self._running_groups[group_info.group_id + self.all_reduce_phases[0]].set_result(allreduce_p_phase)

                if modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX:
                    async for tensor, update in azip(as_aiter(*first_all_reduced), allreduce_p_phase):
                        # all-reduce is performed asynchronously while iterating
                        tensor.add_(update, alpha=self._averaging_alpha)
                else:
                    async for _ in allreduce_p_phase:  # trigger all-reduce by iterating
                        raise ValueError("aux peers should not receive averaged tensors")

                for p in ps:
                    orthogonalize_(p)

                for p, q, m in zip(ps, self._qs, self._ms):
                    torch.matmul(m.reshape(-1, q.size(0)).t(), p, out=q)

                averaged_grad_wo_sgd = [
                    grad for idx, grad in enumerate(averaged_grads) if idx in self._uncompressed_gradients_indexes
                ]

                allreduce_q_phase = AllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id + self.all_reduce_phases[1],
                    tensors=self._qs + averaged_grad_wo_sgd,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    gathered=user_gathered,
                    modes=modes,
                    **kwargs,
                )
                self._running_groups[group_info.group_id + self.all_reduce_phases[1]].set_result(allreduce_q_phase)

                if modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX:
                    async for tensor, update in azip(as_aiter(*self._qs), allreduce_q_phase):
                        # all-reduce is performed asynchronously while iterating
                        tensor.add_(update, alpha=self._averaging_alpha)
                        self.last_updated = get_dht_time()
                        self._state_updated.set()
                else:
                    async for _ in allreduce_q_phase:  # trigger all-reduce by iterating
                        raise ValueError("aux peers should not receive averaged tensors")

                for p, q, m, grad in zip(ps, self._qs, self._ms, averaged_grad_via_sgd):
                    new_m = torch.matmul(p, q.t()).reshape(m.size())
                    m.sub_(new_m)
                    grad.copy_(new_m)

                return allreduce1.gathered
        except BaseException as e:
            logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")

    def get_current_state(self):
        with torch.no_grad(), self.lock_averaged_tensors:
            grad_averager_buffers = [q for q in self._qs]
            grad_averager_buffers_infos = [
                CompressionInfo.from_tensor(buffer, key=f"buffer_q_{key}", role=TensorRole.GRADIENT)
                for buffer, key in zip(grad_averager_buffers, enumerate(grad_averager_buffers))
            ]

        metadata = dict(group_bits=self.get_group_bits())
        return metadata, grad_averager_buffers, grad_averager_buffers_infos

    def load_state_from_peers(self, **kwargs):
        loaded_state = super().load_state_from_peers(**kwargs)
        if loaded_state is None:
            return

        metadata, flat_tensors = loaded_state
        logger.info("Starting loading gradient averager buffers from peers")

        if len(flat_tensors) != len(self._qs):
            logger.error("Failed to load state from peer, received parameters, extras or metadata")
            return

        with torch.no_grad(), self.lock_averaged_tensors:
            for local_q, loaded_q in zip(self._qs, flat_tensors):
                local_q.copy_(loaded_q, non_blocking=True)
