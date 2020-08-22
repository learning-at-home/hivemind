from __future__ import annotations

import asyncio
import time
from typing import Tuple, List, Optional, Awaitable, Set, Dict

import grpc.experimental.aio
import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

import hivemind
from hivemind.client.expert import RemoteExpert, DUMMY, _get_expert_stub
from hivemind.proto import runtime_pb2, runtime_pb2_grpc as runtime_grpc
from hivemind.utils import nested_pack, nested_flatten, serialize_torch_tensor, deserialize_torch_tensor
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


class RemoteMixtureOfExperts(nn.Module):
    """
    A torch module that performs mixture of experts inference with a local gating function and multiple remote experts.
    Natively supports pytorch autograd.

    :note: By default, not all experts are guaranteed to perform forward pass. Moreover, not all of those who ran
     forward pass are guaranteed to perform backward pass. In the latter case, gradient will be averaged without
     the missing experts

    :param in_features: common input size for experts and gating function
    :param grid_size: hivemind dimensions that form expert uid (see below)
    :param uid_prefix: common prefix for all expert uids
     expert uid follows the pattern {uid_prefix}.{0...grid_size[0]}.{0...grid_size[1]}...{0...grid_size[-1]}
    :param dht: DHT where the experts reside
    :param k_best: queries this many experts with highest scores
    :param k_min: makes sure at least this many experts returned output
    :param timeout_after_k_min: waits for this many seconds after k_min experts returned results.
     Any expert that didn't manage to return output after that delay is considered unavailable
    :param allow_broadcasting: if RemoteMixtureOfExperts if fed with input dimension above 2,
     allow_broadcasting=True will flatten first d-1 input dimensions, apply RemoteMixtureOfExperts and un-flatten again
     allow_broadcasting=False will raise an error
    """

    def __init__(self, *, in_features, grid_size: Tuple[int, ...], dht: hivemind.DHT, k_best: int, k_min: int = 1,
                 forward_timeout: Optional[float] = None, timeout_after_k_min: Optional[float] = None,
                 backward_k_min: int = 1, backward_timeout: Optional[float] = None, uid_prefix='',
                 allow_broadcasting=True, loop: asyncio.BaseEventLoop = None):
        super().__init__()
        self.dht, self.grid_size, self.uid_prefix = dht, grid_size, uid_prefix
        self.loop = loop or asyncio.new_event_loop()
        # fmt:off
        assert not self.loop.is_running(), "Event loop is already running. If in jupyter, please apply nest_asyncio " \
            "(pip install nest_asyncio , https://pypi.org/project/nest-asyncio ) and send loop=asyncio.new_event_loop()"
        # fmt:on
        self.k_best, self.k_min, self.backward_k_min = k_best, k_min, backward_k_min
        self.forward_timeout, self.backward_timeout = forward_timeout, backward_timeout
        self.timeout_after_k_min = timeout_after_k_min
        self.allow_broadcasting = allow_broadcasting

        self.proj = nn.Linear(in_features, sum(grid_size))  # jointly predict logits for all grid dimensions
        self._outputs_schema = None  # expert['info'][outputs_schema] from one of experts in the grid

    def forward(self, input: torch.Tensor, *args: torch.Tensor, **kwargs: torch.Tensor):
        """
        Choose k best experts with beam search, then call chosen experts and average their outputs. Input tensor is averaged over all
        dimensions except first and last (we assume that extra dimensions represent sequence length or image dimensions)

        :param input: a tensor of values that are used to estimate gating function, batch-first.
        :param args: extra positional parameters that will be passed to each expert after input, batch-first
        :param kwargs: extra keyword parameters that will be passed to each expert, batch-first
        :returns: averaged predictions of all experts that delivered result on time, nested structure of batch-first
        """
        if input.ndim != 2:
            input_for_gating = input.mean(dim=tuple(range(1, input.ndim - 1)))
        else:
            input_for_gating = input

        # 1. compute scores and find most appropriate experts with beam search
        grid_scores = self.proj(input_for_gating).split_with_sizes(self.grid_size, dim=-1)

        async def _search():
            coroutines = [asyncio.create_task(self.beam_search(
                [dim_scores[i] for dim_scores in grid_scores], self.k_best), name=f'beam_search_{i}')
                for i in range(len(input))]
            return list(await asyncio.gather(*coroutines))

        chosen_experts: List[List[RemoteExpert]] = self.loop.run_until_complete(_search())
        # ^-- List[batch_size] of List[RemoteExpert] chosen for every input in batch

        expert_mask, *expert_outputs = _RemoteCallMany.apply(
            DUMMY, chosen_experts, self.k_min, self.backward_k_min, self.timeout_after_k_min,
            self.forward_timeout, self.backward_timeout, self.loop, *nested_flatten(((input, *args), kwargs)))
        # ^-- multiple tensors of shape [batch_size, max_experts, ...output_shape]

        expert_logits = self.compute_expert_scores(grid_scores, chosen_experts)
        masked_logits = torch.full((1,), float('-inf'), device=expert_logits.device, dtype=expert_logits.dtype)
        expert_logits = torch.where(expert_mask, expert_logits, masked_logits)
        expert_weights = torch.softmax(expert_logits, dim=1)
        averaged_outputs_flat = [
            (expert_weights[..., None] * tensor.flatten(start_dim=2)).view(tensor.shape).sum(dim=1)
            for tensor in expert_outputs]  # ^-- multiply by softmax weights along first 2 axes
        return nested_pack(averaged_outputs_flat, self.outputs_schema)

    async def beam_search(self, grid_scores: List[torch.Tensor], k_best: int, **kwargs) -> List[RemoteExpert]:
        """
        Find and return k best experts in the grid using (exact) beam search of the product space

        :param grid_scores: scores predicted for each dimension in the grid,
        :type grid_scores: a sequence of tensors of shape[batch_size, self.grid_size[i]]
        :param k_best: how many of the top experts participate in the computation
        :param kwargs: extra keyword parameters passed to self.dht.first_k_active
        :returns: a list of *batch_size* lists that contain chosen experts for one sample each inner list contains \
         RemoteExpert instances for *up to* k_best experts
        """
        assert len(grid_scores) == len(self.grid_size)
        assert all(dim_scores.shape == (self.grid_size[dim_index],) for dim_index, dim_scores in enumerate(grid_scores))
        grid_scores = [dim_scores.cpu().detach() for dim_scores in grid_scores]

        beam_experts: List[RemoteExpert] = []
        beam: List[str] = [self.uid_prefix]
        beam_scores = torch.zeros(1)

        for dim_index, dim_scores in enumerate(grid_scores):
            # create all possible successors from current beam and sort them by total score
            expanded_scores = beam_scores[:, None] + dim_scores[None, :]
            sorted_indices = [(flat_i // len(dim_scores), flat_i % len(dim_scores))
                              for flat_i in (-expanded_scores).flatten().argsort().numpy()]

            sorted_candidates = [f"{beam[row]}{self.dht.UID_DELIMITER}{col}" for row, col in sorted_indices]
            candidate_to_indices = dict(zip(sorted_candidates, sorted_indices))

            # select k best candidates according to scores but only those that are still active
            best_alive_prefixes: Dict[str, RemoteExpert] = await self.dht.first_k_active(
                uid_prefixes=sorted_candidates, k=k_best, return_future=True, **kwargs)
            if not best_alive_prefixes:
                logger.warning(f"Grid is empty: found neither of {sorted_candidates}")
                break
            beam = list(best_alive_prefixes.keys())
            beam_scores = expanded_scores[tuple(zip(*map(candidate_to_indices.get, beam)))]
            beam_experts = list(best_alive_prefixes.values())

        if self._outputs_schema is None:
            try:
                self._outputs_schema = beam_experts[0].info['outputs_schema']
            except grpc.RpcError as e:
                logger.warning(f"Failed to get RemoteMixtureOfExperts.output_shape: {e}")

        return beam_experts

    def compute_expert_scores(
            self, grid_scores: List[torch.Tensor], batch_experts: List[List[RemoteExpert]]) -> torch.Tensor:
        """
        Compute scores for each expert by adding up grid scores, autograd-friendly
        :param grid_scores: list of torch tensors, i-th tensor contains scores for i-th grid dimension
        :param batch_experts: list(batch) of lists(k) of up to k experts selected for this batch
        :returns: a tensor of scores, float32[batch_size, k]
        :note: if some rows in batch have less than max number of experts, their scores will be padded with -inf
        """
        expert_counts = list(map(len, batch_experts))
        batch_size = len(batch_experts)
        max_num_experts = max(expert_counts)
        total_num_experts = sum(expert_counts)
        expert_index_in_batch = torch.arange(total_num_experts, device=grid_scores[0].device)
        expert_strides = torch.cumsum(torch.as_tensor([0] + expert_counts, device=grid_scores[0].device), dim=-1)[:-1]
        flat_batch_indices = (expert_index_in_batch >= expert_strides[:, None]).to(torch.int32).sum(0) - 1
        flat_local_indices = expert_index_in_batch - expert_strides[flat_batch_indices]
        flat_experts = [expert for row in batch_experts for expert in row]

        grid_indices = torch.zeros([len(flat_experts), len(grid_scores)], dtype=torch.int64)
        for i, expert in enumerate(flat_experts):
            expert_indices = expert.uid[len(self.uid_prefix) + len(self.dht.UID_DELIMITER):]
            expert_indices = list(map(int, expert_indices.split(self.dht.UID_DELIMITER)))
            grid_indices[i] = torch.as_tensor(expert_indices, dtype=grid_indices.dtype)

        scores_per_dim = [
            dim_scores[flat_batch_indices, dim_indices] if len(flat_batch_indices) else torch.zeros(0)
            for dim_scores, dim_indices in zip(grid_scores, grid_indices.T)]
        flat_scores = torch.sum(torch.stack(scores_per_dim, dim=0), dim=0)

        scores = torch.full((batch_size, max_num_experts), fill_value=-float('inf'), device=grid_scores[0].device)
        scores[flat_batch_indices, flat_local_indices] = flat_scores  # backprop-able w.r.t. flat_scores
        return scores

    @property
    def outputs_schema(self):
        if self._outputs_schema is None:
            # grab some expert to set ensemble output shape
            proj_device = self.proj.weight.device
            dummy_scores_concat = self.proj(torch.randn(1, self.proj.in_features, device=proj_device))
            dummy_scores = dummy_scores_concat.cpu().split_with_sizes(self.grid_size, dim=-1)
            dummy_experts = self.loop.run_until_complete(self.beam_search(dummy_scores, k_best=1))
            self._outputs_schema = dummy_experts[0].info['outputs_schema']
        return self._outputs_schema


class _RemoteCallMany(torch.autograd.Function):
    """
    Internal autograd-friendly function that calls multiple experts on a batch of inputs and awaits responses
    This function that can recover from individual failures during forward and/or backward pass as long as at least
    one expert succeeds for each input. For user-friendly version of this function, use RemoteMixtureOfExperts module.

    Note: experts that failed during forward will be assigned zero outputs and marked as mask[i, j] = 0,
          experts that failed during backward will be treated as constants (i.e. gradients of through them are zeros)
    """

    @classmethod
    def forward(cls, ctx, dummy, experts_per_sample: List[List[RemoteExpert]], k_min: int, backward_k_min: int,
                timeout_after_k_min: float, forward_timeout: Optional[float], backward_timeout: Optional[float],
                loop: asyncio.base_events.BaseEventLoop, *flat_inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        assert not torch.is_grad_enabled()
        num_samples, max_experts = len(experts_per_sample), max(map(len, experts_per_sample))
        flat_inputs_per_sample: List[Tuple[torch.Tensor, ...]] = list(zip(*(x.split(1, dim=0) for x in flat_inputs)))
        assert len(experts_per_sample) == len(flat_inputs_per_sample) == num_samples

        async def _forward():
            # dispatch tasks to all remote experts, await responses
            pending_tasks = {
                asyncio.create_task(cls._forward_one_expert((i, j), expert, flat_inputs_per_sample[i]), name=f'forward_expert_{j}_for_{i}')
                for i in range(num_samples) for j, expert in enumerate(experts_per_sample[i])
            }
            alive_grid_indices, alive_flat_outputs = await cls._wait_for_responses(
                pending_tasks, num_samples, k_min, forward_timeout, timeout_after_k_min)

            # assemble responses
            alive_ii, alive_jj = map(torch.as_tensor, zip(*alive_grid_indices))
            mask = torch.zeros([num_samples, max_experts], dtype=torch.bool, device=flat_inputs[0].device)
            mask[alive_ii, alive_jj] = True

            alive_flat_outputs_stacked = list(map(torch.cat, zip(*alive_flat_outputs)))
            # list of torch tensors, where i-th tensor is of shape [num_responded, *expert_outputs[i].shape]

            outputs = []
            for response_stacked in alive_flat_outputs_stacked:
                output = torch.zeros(
                    [num_samples, max_experts, *response_stacked.shape[1:]], device=response_stacked.device,
                    dtype=response_stacked.dtype, requires_grad=response_stacked.requires_grad)
                output[alive_ii, alive_jj] = response_stacked
                outputs.append(output)

            # save individual outputs for backward pass
            ctx.save_for_backward(alive_ii, alive_jj, *flat_inputs)
            ctx._saved_non_tensors = loop, backward_k_min, backward_timeout, timeout_after_k_min, experts_per_sample
            return (mask,) + tuple(outputs)

        return loop.run_until_complete(_forward())

    @classmethod
    @once_differentiable
    def backward(cls, ctx, *raw_grads: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        assert not torch.is_grad_enabled()
        loop, backward_k_min, backward_timeout, timeout_after_k_min, expert_per_sample = ctx._saved_non_tensors
        alive_ii, alive_jj, *flat_inputs = ctx.saved_tensors
        dummy_grad_mask, *flat_grad_outputs = raw_grads
        num_samples, max_experts = dummy_grad_mask.shape

        inputs_per_expert = zip(*(tensor[alive_ii].split(1, dim=0) for tensor in flat_inputs))
        grad_outputs_per_expert = zip(*(tensor[alive_ii, alive_jj].split(1, dim=0) for tensor in flat_grad_outputs))

        async def _backward():
            # dispatch tasks to all remote experts, await responses
            pending_tasks = set()
            for i, j, inputs_ij, grad_outputs_ij in zip(alive_ii.cpu().numpy(), alive_jj.cpu().numpy(),
                                                        inputs_per_expert, grad_outputs_per_expert):
                pending_tasks.add(asyncio.create_task(
                    cls._backward_one_expert((i, j), expert_per_sample[i.item()][j.item()], inputs_ij, grad_outputs_ij),
                    name=f'backward_expert_{j}_for_{i}'))

            backward_survivor_indices, survivor_grad_inputs = await cls._wait_for_responses(
                pending_tasks, num_samples, backward_k_min, backward_timeout, timeout_after_k_min)

            # assemble responses
            backward_survivor_ii, backward_survivor_jj = map(torch.as_tensor, zip(*backward_survivor_indices))
            survivor_grad_inputs_stacked = list(map(torch.cat, zip(*survivor_grad_inputs)))
            # list of torch tensors, where i-th tensor is of shape [num_backward_survivors, *flat_inputs[i].shape]

            grad_inputs = []
            for i, survivor_grad_stacked in enumerate(survivor_grad_inputs_stacked):
                grad_input_per_expert = torch.zeros(  # gradient tensor with individual contributions from each expert
                    (num_samples, max_experts, *flat_inputs[i].shape[1:]),
                    device=survivor_grad_stacked.device, dtype=survivor_grad_stacked.dtype)
                grad_input_per_expert[backward_survivor_ii, backward_survivor_jj] = survivor_grad_stacked

                grad_inputs.append(grad_input_per_expert.sum(dim=1))  # add up gradients from each expert

            return (DUMMY, None, None, None, None, None, None, None, *grad_inputs)

        return loop.run_until_complete(_backward())

    @staticmethod
    async def _forward_one_expert(grid_indices: Tuple[int, ...], expert: RemoteExpert, inputs: Tuple[torch.Tensor]):
        stub: runtime_grpc.ConnectionHandlerStub = _get_expert_stub(expert.endpoint, aio=True)
        try:
            outputs = await stub.forward(runtime_pb2.ExpertRequest(
                uid=expert.uid, tensors=[serialize_torch_tensor(tensor) for tensor in inputs]))
            return grid_indices, tuple(deserialize_torch_tensor(tensor) for tensor in outputs.tensors)
        except grpc.experimental.aio.AioRpcError as error:
            logger.warning(f"RemoteExpert {expert} failed forward: {error.code()} (inputs: {inputs})")

    @staticmethod
    async def _backward_one_expert(grid_indices: Tuple[int, ...], expert: RemoteExpert,
                                   inputs: Tuple[torch.Tensor], grad_outputs: Tuple[torch.Tensor]):
        stub: runtime_grpc.ConnectionHandlerStub = _get_expert_stub(expert.endpoint, aio=True)
        payload = tuple(nested_flatten((inputs, grad_outputs)))
        try:
            grad_inputs = await stub.backward(runtime_pb2.ExpertRequest(
                uid=expert.uid, tensors=[serialize_torch_tensor(tensor) for tensor in payload]))
            return grid_indices, tuple(deserialize_torch_tensor(tensor) for tensor in grad_inputs.tensors)
        except grpc.experimental.aio.AioRpcError as error:
            logger.warning(f"RemoteExpert {expert} failed backward: {error.code()} ({inputs}, {grad_outputs})")

    @staticmethod
    async def _wait_for_responses(
            pending_tasks: Set[Awaitable[Tuple[Tuple[int, int], Tuple[torch.Tensor, ...]]]],
            num_samples: int, k_min: int, timeout_total: Optional[float], timeout_after_k_min: Optional[float]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[torch.Tensor, ...]]]:
        """ await up to k_min results and any result submitted within timeout_after_k_min, cancel stragglers """
        timeout_total = float('inf') if timeout_total is None else timeout_total
        timeout_after_k_min = float('inf') if timeout_after_k_min is None else timeout_after_k_min
        num_successful_tasks = [0 for _ in range(num_samples)]
        pending_samples = num_samples  # samples for which we have less than k_min results
        finished_indices, finished_outputs = [], []
        t_finish = time.perf_counter() + timeout_total

        while pending_tasks and time.perf_counter() <= t_finish:
            finished_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED,
                                                               timeout=t_finish - time.perf_counter())
            for task in finished_tasks:
                if not task.result():
                    continue
                task_indices, task_flat_outputs = await task
                finished_indices.append(task_indices)
                finished_outputs.append(task_flat_outputs)

                sample_index = task_indices[0]
                num_successful_tasks[sample_index] += 1
                if num_successful_tasks[sample_index] == k_min:
                    pending_samples -= 1
                    if pending_samples <= 0:  # all tasks finished, await stragglers for at most timeout_after_k_min
                        t_finish = min(t_finish, time.perf_counter() + timeout_after_k_min)

        for task in pending_tasks:
            task.cancel()
        return finished_indices, finished_outputs
