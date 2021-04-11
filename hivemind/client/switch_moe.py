from __future__ import annotations

import time
from queue import Queue, Empty
from typing import Tuple, List, Optional, Dict, Any

import grpc

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

import hivemind
from hivemind.client.expert import RemoteExpert, DUMMY, _get_expert_stub
from hivemind.server.expert_uid import UID_DELIMITER
from hivemind.client.beam_search import MoEBeamSearcher
from hivemind.proto import runtime_pb2, runtime_pb2_grpc as runtime_grpc
from hivemind.utils import nested_pack, nested_flatten, serialize_torch_tensor, deserialize_torch_tensor
from hivemind.utils.logging import get_logger
from hivemind.client.moe import RemoteMixtureOfExperts, _RemoteCallMany

logger = get_logger(__name__)


class RemoteSwitchMixtureOfExperts(RemoteMixtureOfExperts):
    """
    A torch module that performs mixture of experts inference with a local gating function and multiple remote experts.
    Natively supports pytorch autograd.

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
    :param detect_anomalies: whether to check input/output tensors for NaN and infinity values
     Any expert that didn't manage to return output after that delay is considered unavailable
    """

    def __init__(self, *, grid_size: Tuple[int, ...], utilization_alpha: float = 0.9, grid_dropout: float = 1.0,
                 jitter_eps: float = 1e-2, **kwargs):
        super().__init__(grid_size=grid_size, **kwargs)
        initial_utilization = torch.tensor([[1 / dim_size for _ in range(dim_size)] for dim_size in grid_size],
                                           dtype=torch.float, requires_grad=False)
        self.register_buffer('grid_utilization', initial_utilization)
        self.utilization_alpha = utilization_alpha
        self.grid_dropout = grid_dropout
        self.jitter_eps = jitter_eps

    def forward(self, input: torch.Tensor, *args: torch.Tensor, **kwargs: torch.Tensor):
        """
        Choose k best experts with beam search, then call chosen experts and average their outputs.
        Input tensor is averaged over all dimensions except for first and last
        (we assume that extra dimensions represent sequence length or image height/width)

        :param input: a tensor of values that are used to estimate gating function, batch-first.
        :param args: extra positional parameters that will be passed to each expert after input, batch-first
        :param kwargs: extra keyword parameters that will be passed to each expert, batch-first
        :returns: averaged predictions of all experts that delivered result on time, nested structure of batch-first
        """
        if input.ndim != 2:
            input_for_gating = input.mean(dim=tuple(range(1, input.ndim - 1)))
        else:
            input_for_gating = input

        # Multiplicative jitter for regularized expert routing
        jitter_noise = torch.empty_like(input_for_gating).uniform_(1 - self.jitter_eps, 1 + self.jitter_eps)
        input_for_gating *= jitter_noise

        # compute scores and find most appropriate experts with beam search
        grid_scores = self.proj(input_for_gating).split_with_sizes(self.beam_search.grid_size, dim=-1)

        chosen_experts: List[List[RemoteExpert]] = self.beam_search.batch_find_best_experts(
            [scores.detach().cpu().numpy() for scores in grid_scores], self.k_best)

        if self._expert_info is None:
            try:
                self._expert_info = next((expert.info for experts_i in chosen_experts for expert in experts_i))
            except grpc.RpcError as e:
                logger.warning(f"Failed to get RemoteMixtureOfExperts.output_shape: {e}")

        # compute softmax among top-k
        # pick top-1, route to them

        # return outputs, multiply with their probs, add up

        # route to top-1 highest-priority

        expert_mask, *expert_outputs = _RemoteCallMany.apply(
            DUMMY, chosen_experts, self.k_min, self.backward_k_min, self.timeout_after_k_min, self.forward_timeout,
            self.backward_timeout, self.detect_anomalies, self.info, *nested_flatten(((input, *args), kwargs)))
        # ^-- multiple tensors of shape [batch_size, max_experts, ...output_shape]

        expert_logits = self.compute_expert_scores(grid_scores, chosen_experts)
        masked_logits = torch.full((1,), float('-inf'), device=expert_logits.device, dtype=expert_logits.dtype)
        expert_logits = torch.where(expert_mask, expert_logits, masked_logits)
        expert_weights = torch.softmax(expert_logits, dim=1)
        averaged_outputs_flat = [
            (expert_weights[..., None] * tensor.flatten(start_dim=2)).view(tensor.shape).sum(dim=1)
            for tensor in expert_outputs]  # ^-- multiply by softmax weights along first 2 axes

        packed_outputs = nested_pack(averaged_outputs_flat, self.info['outputs_schema'])

        # balancing loss: fraction of examples routed to each expert,
        # fraction of probability assigned to each expert (both across all experts)
        # want both to have equal allocation
        return input + packed_outputs[0] #, balancing_loss
