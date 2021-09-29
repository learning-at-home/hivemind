""" An extension of averager that runs PowerSGD decomposition """
from itertools import chain
from typing import Iterator, Optional, Sequence

import torch
from torch import nn

from hivemind import get_dht_time
from hivemind.averaging import DecentralizedAverager
from hivemind.averaging.allreduce import AveragingMode, AllReduceRunner
from hivemind.averaging.averager import GatheredData
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.compression import CompressionInfo, TensorRole
from hivemind.utils import get_logger, asyncio

logger = get_logger(__name__)


class PowerSGDAverager(DecentralizedAverager):
    """
    A decentralized averager that communicates low-rank tensors as described in https://arxiv.org/pdf/2106.10207.pdf
    The implementation is inspired by:
    - https://github.com/epfml/powersgd/blob/master/gradient_reducers.py#L665
    - https://pytorch.org/docs/stable/_modules/torch/distributed/algorithms/ddp_comm_hooks/powerSGD_hook.html

    :param tensors: a list/tuple of pytorch tensors that will used as reference to create gradient accumulators
    :param rank: a decomposition rank from PowerSGD; higher rank means slower but more accurate approximation
    :param allow_none_grads: if False, using PowerSGD with None gradients will raise an error, if True - treat as zeros
    :param skip_1d: if True, PowerSGD will not apply to 1d tensors (these tensors will be averaged as is)
    :param min_compression_rate: any tensors that will have less than this compression rate
    :note: compression rate is defined as original_tensor_size / (p_size + q_size)
    :note: tensors with more than 2 dimensions will be flattened to (first_dim, product_of_subsequent_dims)
    :param parameter_names: optionally provide a list of names for every parameter, used as keys for CompressionInfo
    :param kwargs: any additional parameters will be forwarded to DecentralizedAverager
    """

    def __init__(
            self,
            parameters: Sequence[nn.Parameter],
            rank: int = 4,
            allow_none_grads: bool = False,
            skip_1d: bool = True,
            min_compression_rate: float = 2.0,
            parameter_names: Optional[Sequence[str]] = None,
            **kwargs
    ):
        if parameter_names is None:
            parameter_names = tuple(map("parameter{}".format, range(len(parameters))))
        assert len(parameter_names) == len(parameters)

        ps, qs, compression_mask, compressed_names, uncompressed_names = [], [], [], [], []
        for parameter, name in zip(parameters, parameter_names):
            matrix = to_matrix(parameter)
            compressed_size = (matrix.shape[0] + matrix.shape[1]) * rank
            if (skip_1d and parameter.ndim < 2) or matrix.numel() / compressed_size < min_compression_rate:
                compression_mask.append(False)
                uncompressed_names.append(name)
            else:
                compression_mask.append(True)
                compressed_names.append(name)
                ps.append(torch.zeros(matrix.shape[0], rank, dtype=matrix.dtype).share_memory_())
                qs.append(torch.randn(matrix.shape[1], rank, dtype=matrix.dtype).share_memory_())

        self.parameters, self.ps, self.qs, self.compression_mask = map(tuple, (parameters, ps, qs, compression_mask))
        self._grad_accumulators = tuple(torch.zeros(*x.shape, dtype=x.dtype).share_memory_() for x in parameters)
        averaged_tensors = tuple(chain(self.accumulators(compressed=False), self.ps, self.qs))
        tensor_infos = []
        for tensor, name in zip(self.accumulators(compressed=False), uncompressed_names):
            tensor_infos.append(CompressionInfo.from_tensor(tensor, key=name, role=TensorRole.GRADIENT))
        for tensor, name in zip(self.ps, compressed_names):
            tensor_infos.append(CompressionInfo.from_tensor(tensor, key=name + ".P", role=TensorRole.GRADIENT))
        for tensor, name in zip(self.qs, compressed_names):
            tensor_infos.append(CompressionInfo.from_tensor(tensor, key=name + ".Q", role=TensorRole.GRADIENT))
        super().__init__(averaged_tensors, tensor_infos=tuple(tensor_infos), **kwargs)
        self.allow_none_grads = allow_none_grads

    def accumulators(self, compressed: bool) -> Iterator[torch.Tensor]:
        for grad, mask in zip(self._grad_accumulators, self.compression_mask):
            if compressed == mask:
                yield grad

    def step(self, *args, **kwargs):
        """
        Collect gradients from parameters, apply PowerSGD step and set .grad buffers to averaged PowerSGD approximations
        :param kwargs: any additional arguments are forwarded to DecentralizedAverager.step
        """
        # add local updates to gradient accumulators
        for param, accumulator in zip(self.parameters, self._grad_accumulators):
            if param.grad is None and self.allow_none_grads:
                param.grad = torch.zeros_like(param)
            assert param.grad is not None, "One of the parameters did not have .grad. Set allow_none_grads to override"
            accumulator.add_(param.grad)

        # run PowerSGD
        result = super().step(*args, wait=True, **kwargs)  # wait=False is not implemented yet

        # set local gradients and update accumulators
        compressed_pointer = 0
        for param, is_compressed, accumulator in zip(self.parameters, self.compression_mask, self._grad_accumulators):
            assert param.grad is not None, "grad was set to None while step was in progress"
            if is_compressed:
                torch.matmul(self.ps[compressed_pointer], self.qs[compressed_pointer].t(), out=param.grad)
                accumulator.sub_(param.grad)
                compressed_pointer += 1
            else:
                param.grad.copy_(accumulator)
                accumulator.zero_()
        return result

    def _run_allreduce(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        try:
            bandwidths, mode_ids, user_gathered = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered)))
            modes = tuple(map(AveragingMode, mode_ids))

            download_bandwidths = [thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            )
            is_aux = modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX

            async with self.get_tensors_async() as local_tensors:
                num_uncompressed = sum(self.compression_mask)
                num_compressed = len(self.parameters) - num_uncompressed
                uncompressed_accumulators = local_tensors[:num_uncompressed]
                ps = local_tensors[num_uncompressed: num_uncompressed + num_compressed]
                qs = local_tensors[num_uncompressed + num_compressed:]

                # update local Ps
                for m, p, q in zip(self.accumulators(compressed=True), ps, qs):
                    torch.matmul(m, q, out=p)

                # aggregate and normalize Ps
                self._allreduce_mean_inplace_(
                    local_tensors=uncompressed_accumulators + ps,
                    group_id=group_info.group_id + b'.phase1',
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    gathered=user_gathered,
                    is_aux=is_aux,
                    modes=modes,
                )

                for p in ps:
                    orthonormalize_inplace_(p)

                # compute and aggregate Qs
                for m, p, q in zip(self.accumulators(compressed=True), ps, qs):
                    torch.matmul(m.t(), p, out=q)

                self._allreduce_mean_inplace_(
                    local_tensors=qs,
                    group_id=group_info.group_id + b'.phase2',
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    gathered=user_gathered,
                    is_aux=is_aux,
                    modes=modes,
                )

                # note: gradient accumulators will be updated later in .step
                self.last_updated = get_dht_time()
                return user_gathered
        except BaseException as e:
            logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")

    def _allreduce_mean_inplace_(self, *, local_tensors, group_id, is_aux: bool, **kwargs):
        allreduce = AllReduceRunner(
            p2p=self._p2p,
            servicer_type=type(self),
            prefix=self.prefix,
            group_id=group_id,
            tensors=local_tensors,
            **kwargs,
        )

        with self.register_allreduce_group(group_id, allreduce):
            # actually run all-reduce
            averaging_outputs = [output async for output in allreduce]
            if not is_aux:
                for tensor, update in zip(local_tensors, averaging_outputs):
                    tensor.add_(update, alpha=self._averaging_alpha)

        return allreduce.gathered


@torch.jit.script
def orthonormalize_inplace_(matrix, eps=torch.tensor(1e-8)):
    """Ortho-normalize matrix columns in-place"""
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            rest.sub_(torch.sum(col * rest, dim=0) * col)
    return matrix


def to_matrix(tensor):
    if tensor.ndim < 1:
        return tensor.view(1, -1)
    elif tensor.ndim > 2:
        return tensor.flatten(1)
    else:
        return tensor
