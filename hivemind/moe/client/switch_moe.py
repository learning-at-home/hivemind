from __future__ import annotations

from typing import List, Tuple

import torch

from hivemind.moe.client.expert import DUMMY, RemoteExpert
from hivemind.moe.client.moe import RemoteMixtureOfExperts, _RemoteCallMany
from hivemind.moe.expert_uid import UID_DELIMITER
from hivemind.p2p.p2p_daemon_bindings.control import P2PDaemonError
from hivemind.utils import nested_flatten, nested_pack
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


class RemoteSwitchMixtureOfExperts(RemoteMixtureOfExperts):
    """
    A module implementing Switch Transformers [1] Mixture-of-Experts inference with remote experts.

    [1] Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.
     William Fedus, Barret Zoph, Noam Shazeer. https://arxiv.org/abs/2101.03961

    :note: By default, not all experts are guaranteed to perform forward pass. Moreover, not all of those who ran
     forward pass are guaranteed to perform backward pass. In the latter case, gradient will be averaged without
     the missing experts

    :param in_features: common input size for experts and gating function
    :param grid_size: dimensions that form expert uid (see below)
    :param uid_prefix: common prefix for all expert uids (must end with '.')
    :note: expert uid follows the pattern {uid_prefix}.{0...grid_size[0]}.{0...grid_size[1]}...{0...grid_size[-1]}
    :param dht: a DHT instance used to search for best experts
    :param k_best: average this many highest-scoring experts to compute activations
    :param k_min: make sure at least this many experts returned output (i.e. didn't fail)
    :param timeout_after_k_min: wait for this many seconds after k_min experts returned results.
     Any expert that didn't manage to return output after that delay is considered unavailable
    :param detect_anomalies: whether to check input/output tensors for NaN and infinity values
    :param allow_zero_outputs: whether to return just the input if no experts respond on forward pass
    """

    def __init__(
        self,
        *,
        grid_size: Tuple[int, ...],
        utilization_alpha: float = 0.9,
        grid_dropout: float = 1.0,
        jitter_eps: float = 1e-2,
        k_best=1,
        k_min=0,
        backward_k_min=0,
        allow_zero_outputs=True,
        **kwargs,
    ):
        super().__init__(
            grid_size=grid_size,
            k_best=k_best,
            k_min=k_min,
            backward_k_min=backward_k_min,
            allow_zero_outputs=allow_zero_outputs,
            **kwargs,
        )

        initial_utilization = torch.cat(
            [torch.tensor([1 / dim_size for _ in range(dim_size)], dtype=torch.float) for dim_size in grid_size],
        )
        self.register_buffer("grid_utilization", initial_utilization)
        self.utilization_alpha = utilization_alpha
        self.grid_dropout = grid_dropout
        self.jitter_eps = jitter_eps

    def forward(self, input: torch.Tensor, *args: torch.Tensor, **kwargs: torch.Tensor):
        if input.ndim != 2:
            input_for_gating = input.mean(dim=tuple(range(1, input.ndim - 1)))
        else:
            input_for_gating = input

        # Multiplicative jitter for regularized routing
        jitter_noise = torch.empty_like(input_for_gating).uniform_(1 - self.jitter_eps, 1 + self.jitter_eps)
        input_for_gating *= jitter_noise

        # Compute scores, find most appropriate experts with beam search
        grid_scores = self.proj(input_for_gating).split_with_sizes(self.beam_search.grid_size, dim=-1)

        grid_dropout_masks = (
            (
                torch.rand(size=(dim_size,), dtype=input_for_gating.dtype, device=input_for_gating.device)
                < self.grid_dropout
            )
            for dim_size in self.beam_search.grid_size
        )
        grid_scores_dropout = [
            torch.where(
                dropout_mask,
                grid_score,
                torch.full((1,), float("-inf"), device=grid_score.device, dtype=grid_score.dtype),
            )
            for grid_score, dropout_mask in zip(grid_scores, grid_dropout_masks)
        ]

        grid_softmax = [torch.softmax(grid_score, dim=-1) for grid_score in grid_scores_dropout]
        chosen_experts: List[List[RemoteExpert]] = self.beam_search.batch_find_best_experts(
            [scores.detach().cpu() for scores in grid_scores_dropout], self.k_best
        )

        if self._expert_info is None:
            try:
                self._expert_info = next((expert.info for experts_i in chosen_experts for expert in experts_i))
            except StopIteration:
                raise RuntimeError(
                    "No responding experts found during beam search. Check that UID prefixes and "
                    "the grid size are consistent with running Server instances."
                )
            except P2PDaemonError as e:
                logger.warning(f"Failed to get RemoteSwitchMixtureOfExperts.output_shape: {e}")

        expert_mask, *expert_outputs = _RemoteCallMany.apply(
            DUMMY,
            chosen_experts,
            self.k_min,
            self.backward_k_min,
            self.timeout_after_k_min,
            self.forward_timeout,
            self.backward_timeout,
            self.detect_anomalies,
            self.allow_zero_outputs,
            self.info,
            *nested_flatten(((input, *args), kwargs)),
        )
        # ^-- multiple tensors of shape [batch_size, max_experts, ...output_shape]

        batch_utilization = self._compute_batch_utilization(chosen_experts, expert_mask)
        self.grid_utilization = (
            self.utilization_alpha * self.grid_utilization + (1 - self.utilization_alpha) * batch_utilization
        )

        # compute expert probabilities as product across grid dimensions
        expert_probs = self.compute_expert_scores(grid_softmax, chosen_experts)
        masked_probs = torch.zeros((1,), device=expert_probs.device, dtype=expert_probs.dtype)
        expert_probs = torch.where(expert_mask, expert_probs, masked_probs)

        # multiply outputs by expert probabilities
        averaged_outputs_flat = [
            (expert_probs[..., None] * tensor.flatten(start_dim=2)).view(tensor.shape).sum(dim=1)
            for tensor in expert_outputs
        ]  # ^-- multiply by softmax weights along first 2 axes

        packed_outputs = nested_pack(averaged_outputs_flat, self.info["outputs_schema"])

        # Load balancing loss: multiply fractions of probability mass and fractions of routed examples
        # for each grid dimension, sum across all indices for a dimension. Optimizing this leads to uniform allocation
        balancing_loss = torch.stack(
            [
                torch.mean(dim_softmax.mean(0) * dim_utilization) * dim_size**2
                for dim_softmax, dim_utilization, dim_size in zip(
                    grid_softmax, self.grid_utilization, self.beam_search.grid_size
                )
            ]
        ).sum()

        # residual connection
        if isinstance(packed_outputs, torch.Tensor):
            packed_outputs = packed_outputs + input
        else:
            packed_outputs[0] = packed_outputs[0] + input

        return packed_outputs, balancing_loss

    @torch.no_grad()
    def _compute_batch_utilization(self, batch_experts, expert_mask):
        batch_utilization = [
            torch.zeros((dim_size,), dtype=self.grid_utilization.dtype, device=self.grid_utilization.device)
            for dim_size in self.beam_search.grid_size
        ]

        # out of chosen_experts, select those for which expert_mask is True
        for (sample_idx, expert_idx) in expert_mask.nonzero().cpu().numpy():
            expert = batch_experts[sample_idx][expert_idx]
            expert_indices = expert.uid[len(self.beam_search.uid_prefix) :]
            expert_indices = list(map(int, expert_indices.split(UID_DELIMITER)))

            for dim_index, dim_utilization in zip(expert_indices, batch_utilization):
                dim_utilization[dim_index] += 1

        return torch.cat(
            [torch.nn.functional.normalize(dim_utilization, p=1, dim=0) for dim_utilization in batch_utilization]
        )

    def compute_expert_scores(
        self, grid_probs: List[torch.Tensor], batch_experts: List[List[RemoteExpert]]
    ) -> torch.Tensor:
        """
        Compute scores for each expert by multiplying grid probabilities, autograd-friendly
        :param grid_probs: list of torch tensors, i-th tensor contains scores for i-th grid dimension
        :param batch_experts: list(batch) of lists(k) of up to k experts selected for this batch
        :returns: a tensor of scores, float32[batch_size, k]
        :note: if some rows in batch have less than max number of experts, their scores will be padded with -inf
        """
        expert_counts = list(map(len, batch_experts))
        batch_size = len(batch_experts)
        max_num_experts = max(expert_counts)
        total_num_experts = sum(expert_counts)

        device = grid_probs[0].device

        expert_index_in_batch = torch.arange(total_num_experts, device=device)
        expert_strides = torch.cumsum(torch.as_tensor([0] + expert_counts, device=device), dim=-1)[:-1]
        flat_batch_indices = (expert_index_in_batch >= expert_strides[:, None]).to(torch.int32).sum(0) - 1
        flat_local_indices = expert_index_in_batch - expert_strides[flat_batch_indices]
        flat_experts = [expert for row in batch_experts for expert in row]

        grid_indices = torch.zeros([len(flat_experts), len(grid_probs)], dtype=torch.int64)
        for i, expert in enumerate(flat_experts):
            expert_indices = expert.uid[len(self.beam_search.uid_prefix) :]
            expert_indices = list(map(int, expert_indices.split(UID_DELIMITER)))
            grid_indices[i] = torch.as_tensor(expert_indices, dtype=grid_indices.dtype)

        scores_per_dim = [
            dim_scores[flat_batch_indices, dim_indices] if len(flat_batch_indices) else torch.zeros(0, device=device)
            for dim_scores, dim_indices in zip(grid_probs, grid_indices.T)
        ]
        flat_scores = torch.prod(torch.stack(scores_per_dim, dim=0), dim=0)

        scores = torch.full((batch_size, max_num_experts), fill_value=-float("inf"), device=device)
        scores[flat_batch_indices, flat_local_indices] = flat_scores  # backprop-able w.r.t. flat_scores
        return scores
