import asyncio
import contextlib
import faulthandler
import math
import torch
import multiprocessing as mp

from typing import Any, Iterable, Optional, Sequence

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


class PowerEFGradientAverager(GradientAverager):
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
        **kwargs,
    ):
        self.rank = averager_rank
        self.parameters = tuple(parameters)
        self._uncompressed_gradients = set(i for i, grad in enumerate(self._grads_from_parameters()) if len(tuple(grad.size())) == 1)
        self._gs = list(
            torch.zeros_like(grad, device=accumulate_grads_on)
            for idx, grad in enumerate(self._grads_from_parameters()) if idx not in self._uncompressed_gradients
        )
        self._qs = list(
            torch.rand((grad.reshape((grad.size(0), -1)).size(1), self.rank), device=accumulate_grads_on)
            for idx, grad in enumerate(self._grads_from_parameters()) if idx not in self._uncompressed_gradients
        )
        for tensor in (self._qs + self._gs):
            if tensor is not None:
                assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
                tensor.share_memory_()

        super().__init__(
            self.parameters,
            dht=dht,
            prefix=prefix,
            reuse_grad_buffers=reuse_grad_buffers,
            accumulate_grads_on=accumulate_grads_on,
            client_mode=client_mode,
            warn=warn,
            **kwargs
        )

    @contextlib.contextmanager
    def _register_allreduce_group(self, group_info: GroupInfo):
        """registers a given all-reduce runner to listen for incoming connections"""
        try:
            self._running_groups[group_info.group_id + b'.phase1'] = asyncio.Future()
            self._running_groups[group_info.group_id + b'.phase2'] = asyncio.Future()
            self._pending_groups_registered.set()
            yield
        finally:
            maybe_future = self._running_groups.pop(group_info.group_id + b'.phase1', None)
            if maybe_future and not maybe_future.done():
                logger.warning(f"All-reduce group {group_info.group_id + b'.phase1'} did not finish.")
            maybe_future = self._running_groups.pop(group_info.group_id + b'.phase2', None)
            if maybe_future and not maybe_future.done():
                logger.warning(f"All-reduce group {group_info.group_id + b'.phase2'} did not finish.")
            self._pending_groups_registered.set()

    async def _run_allreduce(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run All-Reduce in a given group and update tensors in place, return gathered metadata"""
        async def _dump_later():
            await asyncio.sleep(15.0)
            print([*map(asyncio.Task.print_stack, asyncio.Task.all_tasks())])
        # task = asyncio.create_task(_dump_later())
        try:
            bandwidths, mode_ids, user_gathered_bytes = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)))
            modes = tuple(map(AveragingMode, mode_ids))

            # compute optimal part sizes from peer bandwidths; TODO: replace with proper load balancing
            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            )

            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                compressed_tensors = [lt for idx, lt in enumerate(local_tensors) if idx not in self._uncompressed_gradients]
                cs = [torch.zeros_like(grad, device="cpu") for grad in compressed_tensors]
                for c, g, cg in zip(cs, self._gs, compressed_tensors):
                    torch.sub(cg, g, out=c)

                ps = [torch.zeros((grad.size(0), self.rank), device="cpu") for grad in compressed_tensors]
                for p, q, c in zip(ps, self._qs, cs):
                    torch.matmul(c.reshape(-1, q.size(0)), q, out=p)
                first_all_reduced = ps + [lt for idx, lt in enumerate(local_tensors) if idx in self._uncompressed_gradients]
                allreduce1 = AllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id + b'.phase1',
                    tensors=first_all_reduced,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    gathered=user_gathered,
                    modes=modes,
                    **kwargs,
                )
                self._running_groups[group_info.group_id + b'.phase1'].set_result(allreduce1)

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
                for p, q, c in zip(ps, self._qs, cs):
                    torch.matmul(c.reshape(-1, q.size(0)).t(), p, out=q)

                allreduce2 = AllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id + b'.phase2',
                    tensors=self._qs,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    gathered=user_gathered,
                    modes=modes,
                    **kwargs,
                )
                self._running_groups[group_info.group_id + b'.phase2'].set_result(allreduce2)

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
                for p, q, c in zip(ps, self._qs, cs):
                    new_c = torch.matmul(p, q.t())
                    c.copy_(new_c.reshape(c.size()))

                for c, g in zip(cs, self._gs):
                    torch.add(g, c * 0.9, out=g)

                return allreduce1.gathered
        except BaseException as e:
            logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")
        finally:
            pass

    @contextlib.contextmanager
    @torch.no_grad()
    def use_averaged_gradients(self):
        self._new_averaged_grads = False
        with self.get_tensors() as averaged_grads:
            compressed_tensors = [lt for idx, lt in enumerate(averaged_grads) if idx not in self._uncompressed_gradients]
            old_averaged = [torch.zeros_like(lt) for lt in compressed_tensors]
            for g, cg, oag in zip(self._gs, compressed_tensors, old_averaged):
                oag.copy_(cg)
                cg.copy_(g)
            try:
                assert len(averaged_grads) == len(self.parameters)
                old_grads = [param.grad for param in self.parameters]
                for param, new_grad in zip(self.parameters, averaged_grads):
                    param.grad = new_grad
                yield
            finally:
                for param, old_grad in zip(self.parameters, old_grads):
                    param.grad = old_grad
            for cg, oag in zip(compressed_tensors, old_averaged):
                cg.copy_(oag)


@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0) * col
