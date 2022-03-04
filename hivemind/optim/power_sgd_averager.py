import asyncio
import contextlib
import faulthandler
import math
import multiprocessing as mp
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch

import hivemind
from hivemind.averaging.allreduce import AllreduceException, AllReduceRunner, AveragingMode, GroupID
from hivemind.averaging.control import AveragingStage, StepControl
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import Matchmaking, MatchmakingException
from hivemind.averaging.partition import DEFAULT_PART_SIZE_BYTES
from hivemind.compression import (
    CompressionBase,
    CompressionInfo,
    NoCompression,
    deserialize_torch_tensor,
    serialize_torch_tensor,
    TensorRole,
)
from hivemind.dht import DHT, DHTID
from hivemind.p2p import P2P, P2PContext, P2PHandlerError, PeerID, ServicerBase
from hivemind.proto import averaging_pb2
from hivemind.utils import MPFuture, TensorDescriptor, get_logger
from hivemind.utils.asyncio import (
    achain,
    aiter_with_timeout,
    anext,
    as_aiter,
    azip,
    enter_asynchronously,
    switch_to_uvloop,
)
from hivemind.utils.grpc import combine_from_streaming, split_for_streaming
from hivemind.utils.serializer import MSGPackSerializer, SerializerBase
from hivemind.utils.timed_storage import DHTExpiration, ValueWithExpiration, get_dht_time

from .grad_averager import GradientAverager

GatheredData = Any
logger = get_logger(__name__)


class PowerSGDGradientAverager(GradientAverager):
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        averager_rank: int,
        *,
        dht: hivemind.DHT,
        prefix: str,
        reuse_grad_buffers: bool = False,
        accumulate_grads_on: Optional[torch.device] = None,
        client_mode: bool = None,
        warn: bool = True,
        min_comprasion_ratio: float = 0.5,
        averaged_grads: Optional[Sequence[torch.Tensor]] = None,
        **kwargs,
    ):
        self.rank = averager_rank
        self.parameters = tuple(parameters)
        self._uncompressed_gradients = set(
            i
            for i, grad in enumerate(self._grads_from_parameters())
            if len(tuple(grad.size())) == 1
            or (
                self.rank * (grad.size(0) + np.prod(grad.size()[1:])) / np.prod(grad.size()) > 1 - min_comprasion_ratio
            )
        )
        self._ms = list(torch.zeros_like(grad, device="cpu").share_memory_() for grad in self._grads_from_parameters())
        self._qs = list(
            torch.rand((grad.reshape((grad.size(0), -1)).size(1), self.rank), device="cpu").share_memory_()
            for idx, grad in enumerate(self._grads_from_parameters())
            if idx not in self._uncompressed_gradients
        )

        self.all_reduce_phases = (b".phase1", b".phase2")

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
                for grad, m in zip(averaged_grads, self._ms):
                    m.add_(grad.to(m.device))

                averaged_sgd_ms = [
                    m for idx, m in enumerate(self._ms) if idx not in self._uncompressed_gradients
                ]
                averaged_sgd_grad = [
                    grad for idx, grad in enumerate(averaged_grads) if idx not in self._uncompressed_gradients
                ]
                ps = [
                    torch.zeros((grad.size(0), self.rank), device="cpu")
                    for idx, grad in enumerate(averaged_grads)
                    if idx not in self._uncompressed_gradients
                ]
                for p, q, m in zip(ps, self._qs, averaged_sgd_ms):
                    torch.matmul(m.reshape(-1, q.size(0)), q, out=p)
                first_all_reduced = ps + [
                    m for idx, m in enumerate(self._ms) if idx in self._uncompressed_gradients
                ]
                allreduce1 = AllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id + self.all_reduce_phases[0],
                    tensors=first_all_reduced,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    gathered=user_gathered,
                    modes=modes,
                    **kwargs,
                )
                self._running_groups[group_info.group_id + self.all_reduce_phases[0]].set_result(allreduce1)

                if modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX:
                    async for tensor, update in azip(as_aiter(*first_all_reduced), allreduce1):
                        # all-reduce is performed asynchronously while iterating
                        tensor.add_(update, alpha=self._averaging_alpha)
                else:
                    async for _ in allreduce1:  # trigger all-reduce by iterating
                        raise ValueError("aux peers should not receive averaged tensors")

                # orth ps
                for p in ps:
                    orthogonalize(p)

                # compute qs
                for p, q, m in zip(ps, self._qs, averaged_sgd_ms):
                    torch.matmul(m.reshape(-1, q.size(0)).t(), p, out=q)

                allreduce2 = AllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id + self.all_reduce_phases[1],
                    tensors=self._qs,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    gathered=user_gathered,
                    modes=modes,
                    **kwargs,
                )
                self._running_groups[group_info.group_id + self.all_reduce_phases[1]].set_result(allreduce2)

                if modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX:
                    async for tensor, update in azip(as_aiter(*self._qs), allreduce2):
                        # all-reduce is performed asynchronously while iterating
                        tensor.add_(update, alpha=self._averaging_alpha)
                        self.last_updated = get_dht_time()
                        self._state_updated.set()
                else:
                    async for _ in allreduce2:  # trigger all-reduce by iterating
                        raise ValueError("aux peers should not receive averaged tensors")

                # recompute grads
                for p, q, m, grad in zip(ps, self._qs, averaged_sgd_ms, averaged_sgd_grad):
                    new_m = torch.matmul(p, q.t())
                    m.sub_(new_m.reshape(m.size()))
                    grad.copy_(new_m.reshape(grad.size()))

                for idx, (m, grad) in enumerate(zip(self._ms, averaged_grads)):
                    if idx in self._uncompressed_gradients:
                        grad.copy_(m)
                        m.data[...] = 0

                return allreduce1.gathered
        except BaseException as e:
            logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")
        finally:
            pass

    def get_current_state(self):
        with torch.no_grad(), self.lock_averaged_tensors:
            grad_averager_buffers = list(q for q in self._qs)
            grad_averager_buffers_infos = [
                CompressionInfo.from_tensor(buffer, key=f"buffer_q_{key}", role=TensorRole.GRADIENT)
                for buffer, key in zip(grad_averager_buffers, range(len(grad_averager_buffers)))
            ]

        metadata = dict(group_bits=self.get_group_bits())
        return metadata, grad_averager_buffers, grad_averager_buffers_infos

    def load_state_from_peers(self, **kwargs):
        loaded_state = super().load_state_from_peers(**kwargs)
        if loaded_state is None:
            return

        metadata, flat_tensors = loaded_state
        logger.info("Starting loading gradient averager buffers from peers")

        if num_parameters_and_extras != len(self._qs):
            logger.error("Failed to load state from peer, received parameters, extras or metadata")
            return

        with torch.no_grad(), self.lock_averaged_tensors:
            for local_q, loaded_q in zip(self._qs, flat_tensors):
                local_q.copy_(loaded_q, non_blocking=True)


@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            rest -= torch.sum(col * rest, dim=0) * col
