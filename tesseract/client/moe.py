import multiprocessing as mp
import multiprocessing.pool
from functools import partial
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from .expert import RemoteExpert, _RemoteModuleCall
from ..utils import nested_map, check_numpy, run_and_await_k, nested_pack, nested_flatten, DUMMY
from ..utils import run_isolated_forward, EmulatedAutogradContext, run_isolated_backward, map_with_parallel_backward


class RemoteMixtureOfExperts(nn.Module):
    """
    A torch module that performs mixture of experts inference with a local gating function and multiple remote experts.
    Natively supports pytorch autograd.

    :note: By default, not all experts are guaranteed to perform forward pass. Moreover, not all of those who ran
     forward pass are guaranteed to perform backward pass. In the latter case, gradient will be averaged without
     the missing experts

    :param in_features: common input size for experts and gating function
    :param grid_size: tesseract dimensions that form expert uid (see below)
    :param uid_prefix: common prefix for all expert uids
     expert uid follows the pattern {uid_prefix}.{0...grid_size[0]}.{0...grid_size[1]}...{0...grid_size[-1]}
    :param network: TesseractNetwork where the experts reside
    :param num_workers: number of threads for parallel network operation
    :param k_best: queries this many experts with highest scores
    :param k_min: makes sure at least this many experts returned output
    :param timeout_after_k_min: waits for this many seconds after k_min experts returned results.
     Any expert that didn't manage to return output after that delay is considered unavailable
    :param expert_padding: internal value used to denote "absent expert". Should not coincide with any expert uid.
    """
    def __init__(self, *, in_features, grid_size: Tuple[int], network, num_workers=None,
                 k_best, k_min=1, timeout_after_k_min=1.0, backward_k_min=1, backward_timeout=None,
                 uid_prefix='', expert_padding=None):
        super().__init__()
        self.network, self.grid_size = network, grid_size
        self.uid_prefix, self.expert_padding = uid_prefix, expert_padding
        self.k_best, self.k_min, self.backward_k_min = k_best, k_min, backward_k_min
        self.timeout_after_k_min, self.backward_timeout = timeout_after_k_min, backward_timeout

        self.thread_pool = mp.pool.ThreadPool(num_workers or k_best * 2)
        self.proj = nn.Linear(in_features, sum(grid_size))  # jointly predict logits for all grid dimensions

        # grab some expert to set ensemble output shape
        dummy_scores = self.proj(torch.randn(1, self.proj.in_features)).split_with_sizes(grid_size, dim=-1)
        self.output_schema = self.beam_search(dummy_scores, k_best=1)[0][0].info['output_schema']

    def forward(self, input: torch.Tensor, *args: torch.Tensor, **kwargs: torch.Tensor):
        """
        Choose k best experts with beam search, then call chosen experts and average their outputs.
        :param input: a tensor of values that are used to estimate gating function, batch-first
        :param args: extra positional parameters that will be passed to each expert after input, batch-first
        :param kwargs: extra keyword parameters that will be passed to each expert, batch-first
        :returns: averaged predictions of all experts that delivered result on time, nested structure of batch-first
        """
        if self.allow_broadcasting and input.shape != 2:
            # flatten extra dimensions, apply the function and then un-flatten them back to normal like nn.Linear does
            flattened_dims = input.shape[:-1]
            input_flat = input.view(-1, input.shape[-1])
            args_flat = [tensor.view(-1, tensor.shape[len(flattened_dims):]) for tensor in args]
            kwargs_flat = {key: tensor.view(-1, tensor.shape[len(flattened_dims):]) for key, tensor in kwargs.items()}
            out_flat = self.forward(input_flat, *args_flat, **kwargs_flat)
            return nested_map(lambda tensor: tensor.view(flattened_dims, tensor.shape[len(flattened_dims):]), out_flat)

        # 1. compute scores and find most appropriate experts with beam search
        grid_scores = self.proj(input).split_with_sizes(self.grid_size, dim=-1)
        chosen_experts = self.beam_search(grid_scores, self.k_best)
        # ^-- List[batch_size] of List[RemoteExpert] chosen for every input in batch

        expert_logits = self._score_experts(grid_scores, chosen_experts)

        expert_inputs = ((input, *args), kwargs)
        input_schema = nested_map(lambda x: None, expert_inputs)
        flat_inputs_per_expert = tuple(zip(*[tensor.split(1, dim=0) for tensor in nested_flatten(expert_inputs)]))

        batch_jobs_args = tuple(
            (expert_logits[i], chosen_experts[i], self.k_min, self.timeout_after_k_min,
             self.forward_timeout, self.backward_timeout, input_schema, *flat_inputs_per_expert[i])
            for i in range(len(input))
        )

        averaged_outputs_flat = map_with_parallel_backward(_RemoteMoECall, *batch_jobs_args)
        return nested_pack(averaged_outputs_flat, self.outputs_schema)

    def beam_search(self, grid_scores: List[torch.Tensor], k_best: int, **kwargs) -> List[List[RemoteExpert]]:
        """
        Find and return k best experts in the grid using (exact) beam search of the product space

        :param grid_scores: scores predicted for each dimension in the grid,
        :type grid_scores: a sequence of tensors of shape[batch_size, self.grid_size[i]]
        :param k_best: how many of the top experts participate in the computation
        :param kwargs: extra keyword parameters passed to self.network.first_k_active
        :returns: a list of *batch_size* lists that contain chosen experts for one sample each inner list contains \
         RemoteExpert instances for *up to* k_best experts
        """
        assert len(grid_scores) == len(self.grid_size)
        assert all(len(dim_scores.shape) == 2 for dim_scores in grid_scores)
        batch_size = len(grid_scores[0])
        beam = np.array([[self.uid_prefix]] * batch_size, dtype=object)  # [batch_size, up_to_beam_size]
        scores = np.zeros([batch_size, 1], dtype=np.float64)

        delimeters = np.array(self.network.UID_DELIMETER)[None, None, None]  # pre-compute numpy array for fast concat

        for dim_index, dim_scores in enumerate(grid_scores):
            dim_scores = check_numpy(dim_scores)
            assert dim_scores.shape[-1] == self.grid_size[dim_index]

            # create all possible successsors from current beam
            dim_indices = np.arange(dim_scores.shape[1]).astype(str)
            new_candidates = beam[:, :, None] + delimeters + dim_indices[None, None, :]
            new_candidates = new_candidates.reshape([batch_size, -1])

            new_scores = scores[:, :, None] + dim_scores[:, None, :]
            new_scores = new_scores.reshape([batch_size, -1])

            # select k best candidates according to scores but only those that are still active
            new_order = np.argsort(- new_scores, axis=-1)
            top_alive_lookups = [
                self.thread_pool.apply_async(self.network.first_k_active, args=(cands[order], k_best), kwds=kwargs)
                for cands, order in zip(new_candidates, new_order)]

            batch_cand_to_score = [
                dict(zip(cands, cand_scores)) for cands, cand_scores in zip(new_candidates, new_scores)]

            top_alive_prefixes = [result.get() for result in top_alive_lookups]
            top_alive_scores = [list(map(cand_to_score.get, top_cands))
                                for cand_to_score, top_cands in zip(batch_cand_to_score, top_alive_prefixes)]

            # pad up to beam size
            beam = np.array([row + [self.expert_padding] * (k_best - len(row))
                             for row in top_alive_prefixes], dtype='object')
            scores = np.array([row + [-float('inf')] * (k_best - len(row))
                               for row in top_alive_scores], dtype='float32')

        unique_experts = self.network.get_experts(list(set(
            uid for row in beam for uid in row if uid != self.expert_padding)))
        unique_experts_by_uid = {expert.uid: expert for expert in unique_experts if expert != self.expert_padding}

        return [[unique_experts_by_uid[uid] for uid in row if uid in unique_experts_by_uid] for row in beam]

    def _score_experts(self, grid_scores: List[torch.Tensor],
                       experts: List[List[RemoteExpert]]) -> List[Dict[RemoteExpert, torch.Tensor]]:
        flat_experts = [expert for row in experts for expert in row]
        flat_batch_indices = torch.tensor([i for i, row in enumerate(experts) for uid in range(len(row))])

        grid_indices = np.zeros([len(flat_experts), len(grid_scores)], dtype=np.int64)
        for i, expert in enumerate(flat_experts):
            expert_indices = expert.uid[len(self.uid_prefix) + len(self.network.UID_DELIMETER):]
            expert_indices = list(map(int, expert_indices.split(self.network.UID_DELIMETER)))
            grid_indices[i] = expert_indices

        scores_per_dim = [
            dim_scores[flat_batch_indices, dim_indices] if len(flat_batch_indices) else torch.zeros(0)
            for dim_scores, dim_indices in zip(grid_scores, grid_indices.T)]
        flat_scores = torch.sum(torch.stack(scores_per_dim, dim=0), dim=0)

        output_dicts = [dict() for _ in range(len(experts))]
        for batch_i, expert, score in zip(check_numpy(flat_batch_indices),
                                          flat_experts, flat_scores):
            output_dicts[batch_i][expert] = score

        return output_dicts


class _RemoteMoECall(torch.autograd.Function):
    """
    Internal autograd-friendly function that calls multiple experts on the same input and averages their outputs.
    This function that can recover from individual failures during forward and/or backward passes.
    For user-friendly version of this function, use RemoteMixtureOfExperts module.
    """
    @classmethod
    def forward(cls, ctx, expert_logits: torch.Tensor, experts: List[RemoteExpert],
                k_min: int, timeout_after_k_min: float, backward_k_min: int, timeout_total: Optional[float],
                backward_timeout: Optional[float], input_schema, *flat_inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        expert_args, expert_kwargs = nested_pack(flat_inputs, structure=input_schema)
        assert expert_logits.ndim == 1 and len(expert_logits) == len(experts)

        # 1. call experts and await results
        jobs = [partial(cls._run_expert_forward, expert, *expert_args, **expert_kwargs) for expert in experts]
        results = run_and_await_k(jobs, k=k_min, timeout_after_k=timeout_after_k_min, timeout_total=timeout_total)

        alive_contexts, alive_outputs, alive_ix = zip(*[(result[0], result[1], ix) for ix, result in enumerate(results)
                                                        if not isinstance(result, BaseException)])
        #     ^               ^            ^-- a list of indices of experts that returned outputs in time
        #      \               \-- list of outputs of every expert that didn't die on us
        #       \-- a list of autograd contexts, used for parallel backward

        # 2. compute softmax weights for alive experts and average outputs
        alive_ix = torch.as_tensor(alive_ix, device=expert_logits.device)
        alive_expert_probs = torch.softmax(expert_logits[alive_ix], dim=0)

        stacked_alive_outputs = tuple(map(torch.stack, zip(*alive_outputs)))

        flat_average_outputs = tuple((alive_expert_probs @ stacked_out.flatten(1)).view(*stacked_out.shape[1:])
                                     for stacked_out in stacked_alive_outputs)

        # 3. save individual outputs for backward pass
        ctx.save_for_backward(expert_logits, alive_ix, alive_expert_probs, *stacked_alive_outputs)
        ctx._saved_non_tensors = alive_contexts, backward_k_min, backward_timeout
        return tuple(map(torch.Tensor.detach, flat_average_outputs))

    @classmethod
    @once_differentiable
    def backward(cls, ctx, *grad_outputs_flat: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """ Like normal backward, but we ignore any experts that failed during backward pass """
        expert_logits, alive_ix, alive_expert_probas, *stacked_alive_outputs = ctx.saved_tensors
        alive_contexts, backward_k_min, backward_timeout = ctx._saved_non_tensors

        jobs = [partial(cls._run_expert_backward, ctx, prob, *grad_outputs_flat)
                for ctx, prob in zip(alive_contexts, alive_expert_probas.split(1))]
        results = run_and_await_k(jobs, k=backward_k_min, timeout_after_k=None, timeout_total=backward_timeout)
        backward_survivors_in_alive_ix, survived_grad_inputs = zip(*((i, grads) for i, grads in enumerate(results)))
        backward_survivors_in_alive_ix = torch.as_tensor(backward_survivors_in_alive_ix, device=expert_logits.device)
        backward_survivors_ix = alive_ix[backward_survivors_in_alive_ix]
        survived_probas = torch.softmax(expert_logits[backward_survivors_ix], dim=0)
        weight_ratios = survived_probas / alive_expert_probas[backward_survivors_in_alive_ix]
        flat_grad_inputs = tuple((weight_ratios @ stacked_grad_inp.flatten(1)).view(stacked_grad_inp.shape[1:])
                                 for stacked_grad_inp in map(torch.stack, zip(*survived_grad_inputs)))

        # compute grad w.r.t. logits
        grad_wrt_probs = sum(tuple(
            torch.sum(grad_out[None, ...] * stacked_avive_out[backward_survivors_in_alive_ix],
                      dim=tuple(range(1, stacked_avive_out.ndim)))
            for grad_out, stacked_avive_out in zip(grad_outputs_flat, stacked_alive_outputs)
        ))
        softmax_jacobian = torch.diagflat(survived_probas) - torch.ger(survived_probas, survived_probas)
        grad_wrt_logits = grad_wrt_probs @ softmax_jacobian

        return grad_wrt_logits, None, None, None, None, None, None, None, *flat_grad_inputs

    @staticmethod
    def _run_expert_forward(expert: RemoteExpert, *args: torch.Tensor, **kwargs: torch.Tensor):
        """ Call remote expert and return flattened outputs. Compatible with concurrent autograd. """
        flat_inputs = nested_flatten((args, kwargs))
        return run_isolated_forward(_RemoteModuleCall, DUMMY, expert.uid, expert.host, expert.port, *flat_inputs)

    @staticmethod
    def _run_expert_backward(ctx: EmulatedAutogradContext, weight: torch.Tensor, *grad_outputs: torch.Tensor):
        backward_result = run_isolated_backward(_RemoteModuleCall, ctx, *(grad * weight for grad in grad_outputs))
        grad_dummy, no_grad_uid, no_grad_hostname, no_grad_port, *grad_inputs = backward_result
        return grad_inputs
