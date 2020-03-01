import multiprocessing as mp
import multiprocessing.pool
from functools import partial
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from .remote_expert import RemoteExpert
from ..utils import nested_map, check_numpy, run_and_await_k


class GatingFunction(nn.Module):
    def __init__(self, *, in_features, grid_size: Tuple[int], network, num_workers=None,
                 k_best, k_min=1, timeout_after_k_min=1.0, uid_prefix='', expert_padding=None):
        super().__init__()
        self.network, self.grid_size = network, grid_size
        self.uid_prefix, self.expert_padding = uid_prefix, expert_padding
        self.k_best, self.k_min, self.timeout_after_k_min = k_best, k_min, timeout_after_k_min

        self.thread_pool = mp.pool.ThreadPool(num_workers or k_best * 2)
        self.proj = nn.Linear(in_features, sum(grid_size))  # jointly predict logits for all grid dimensions

    def forward(self, input: torch.Tensor, *args, **kwargs) -> Tuple[List[List[RemoteExpert]], torch.Tensor]:
        """
        Choose k best experts with beam search, then call chosen experts and average their outputs.
        :param batch: named tensors, each tensor has 0-th axis dedicated to batch (aka batch-first
        :return: averaged predictions of all experts that delivered on time
        """
        assert len(input.shape) == 2

        # 1. compute scores and find most appropriate experts with beam search
        grid_scores = self.proj(input).split_with_sizes(self.grid_size, dim=-1)
        batch_experts = self.beam_search(grid_scores, self.k_best)
        # ^-- List[batch_size] of List[RemoteExpert] chosen for every input in batch

        # 2.1 call chosen experts (run them in background to save time)
        batch_outputs_async = [
            self.thread_pool.apply_async(self._run_experts,
                                         args=[chosen_experts, input[i: i + 1], *(tensor[i: i + 1] for tensor in args)],
                                         kwds={key: tensor[i: i + 1] for key, tensor in kwargs.items()})
            for i, chosen_experts in enumerate(batch_experts)
        ]

        # 2.2 compute *differentiable* logits for each expert
        batch_expert_logits = self._score_experts(grid_scores, batch_experts)
        # ^-- List[batch_size] of Dict[RemoteExpert, logit] before softmax for each active expert

        batch_outputs = []
        for output_async, expert_logits in zip(batch_outputs_async, batch_expert_logits):
            expert_outputs: Dict[RemoteExpert, Any] = output_async.get()
            flat_experts, flat_outputs = zip(*expert_outputs.items())

            # 3.1. normalize logits over only those experts that DID return output
            flat_logits = torch.stack([expert_logits[expert] for expert in flat_experts])
            flat_weights = torch.softmax(flat_logits, dim=-1)

            # 3.2. average each output across experts
            average_outputs = nested_map(
                lambda *tensors: sum(x * weight for x, weight in zip(tensors, flat_weights)), *flat_outputs)

            batch_outputs.append(average_outputs)

        # 4. concatenate mixture outputs from individual experts
        return nested_map(lambda *tensors: torch.cat(tensors, dim=0), *batch_outputs)

    def beam_search(self, grid_scores: List[torch.Tensor], k_best: int, **kwargs) -> List[List[RemoteExpert]]:
        """
        Find and return k best experts in the grid using (exact) beam search of the product space
        :param grid_scores: scores predicted for each dimension in the grid,
        :type grid_scores: a sequence of tensors of shape[batch_size, self.grid_size[i]]
        :param k_best: how many of the top experts participate in the computation
        :param kwargs: extra keyword parameters passed to self.network.first_k_active
        :returns: a list of *batch_size* lists that contain chosen experts for one sample
            each inner list contains RemoteExpert instances for *up to* k_best experts
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

        return [
            [unique_experts_by_uid[uid] for uid in row if uid in unique_experts_by_uid]
            for row in beam]

    def _run_experts(self, experts: List[RemoteExpert], *args, **kwargs) -> Dict[RemoteExpert, torch.Tensor]:
        outputs = run_and_await_k([partial(expert, *args, **kwargs) for expert in experts],
                                  k=self.k_min, timeout_after_k=self.timeout_after_k_min)
        return {expert: output for expert, output in zip(experts, outputs)
                if not isinstance(output, BaseException)}

    def _score_experts(self, grid_scores: List[torch.Tensor],
                       experts: List[List[RemoteExpert]]) -> List[Dict[RemoteExpert, torch.Tensor]]:
        flat_experts = [expert for row in experts for expert in row]
        flat_batch_indices = torch.tensor([i for i, row in enumerate(experts)
                                           for uid in range(len(row))])

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
