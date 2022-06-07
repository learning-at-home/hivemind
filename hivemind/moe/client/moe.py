from __future__ import annotations

import time
from concurrent.futures import Future
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from hivemind.compression import serialize_torch_tensor
from hivemind.dht import DHT
from hivemind.moe.client.beam_search import MoEBeamSearcher
from hivemind.moe.client.expert import DUMMY, RemoteExpert, expert_backward, expert_forward, get_server_stub
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.moe.expert_uid import UID_DELIMITER
from hivemind.p2p.p2p_daemon_bindings.control import P2PDaemonError
from hivemind.utils import nested_flatten, nested_map, nested_pack
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


class RemoteMixtureOfExperts(nn.Module):
    """
    A torch module that performs Mixture-of-Experts inference with a local gating function and multiple remote experts.
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
     Any expert that didn't manage to return output after that delay is considered unavailable
    :param detect_anomalies: whether to check input/output tensors for NaN and infinity values
    :param allow_zero_outputs: whether to return zeros if no experts respond on forward pass
    """

    def __init__(
        self,
        *,
        in_features,
        grid_size: Tuple[int, ...],
        dht: DHT,
        uid_prefix: str,
        k_best: int,
        k_min: int = 1,
        forward_timeout: Optional[float] = None,
        timeout_after_k_min: Optional[float] = None,
        backward_k_min: int = 1,
        backward_timeout: Optional[float] = None,
        detect_anomalies: bool = False,
        allow_zero_outputs: bool = False,
        **dht_kwargs,
    ):
        super().__init__()
        self.dht = dht
        self.beam_search = MoEBeamSearcher(dht, uid_prefix, grid_size, **dht_kwargs)
        self.k_best, self.k_min, self.backward_k_min = k_best, k_min, backward_k_min
        self.forward_timeout, self.backward_timeout = forward_timeout, backward_timeout
        self.timeout_after_k_min = timeout_after_k_min
        self.detect_anomalies = detect_anomalies
        self.allow_zero_outputs = allow_zero_outputs

        # jointly predict logits for all grid dimensions
        self.proj = nn.Linear(in_features, self.beam_search.total_grid_size)
        self._expert_info = None  # expert['info'] from one of experts in the grid

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

        # 1. compute scores and find most appropriate experts with beam search
        grid_scores = self.proj(input_for_gating).split_with_sizes(self.beam_search.grid_size, dim=-1)

        chosen_experts: List[List[RemoteExpert]] = self.beam_search.batch_find_best_experts(
            [scores.detach().cpu().numpy() for scores in grid_scores], self.k_best
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
                logger.warning(f"Failed to get RemoteMixtureOfExperts.output_shape: {e}")

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

        expert_logits = self.compute_expert_scores(grid_scores, chosen_experts)
        masked_logits = torch.full((1,), float("-inf"), device=expert_logits.device, dtype=expert_logits.dtype)
        expert_logits = torch.where(expert_mask, expert_logits, masked_logits)
        expert_weights = torch.softmax(expert_logits, dim=1)
        averaged_outputs_flat = [
            (expert_weights[..., None] * tensor.flatten(start_dim=2)).view(tensor.shape).sum(dim=1)
            for tensor in expert_outputs
        ]  # ^-- multiply by softmax weights along first 2 axes

        return nested_pack(averaged_outputs_flat, self.info["outputs_schema"])

    def compute_expert_scores(
        self, grid_scores: List[torch.Tensor], batch_experts: List[List[RemoteExpert]]
    ) -> torch.Tensor:
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

        device = grid_scores[0].device

        expert_index_in_batch = torch.arange(total_num_experts, device=device)
        expert_strides = torch.cumsum(torch.as_tensor([0] + expert_counts, device=device), dim=-1)[:-1]
        flat_batch_indices = (expert_index_in_batch >= expert_strides[:, None]).to(torch.int32).sum(0) - 1
        flat_local_indices = expert_index_in_batch - expert_strides[flat_batch_indices]
        flat_experts = [expert for row in batch_experts for expert in row]

        grid_indices = torch.zeros([len(flat_experts), len(grid_scores)], dtype=torch.int64)
        for i, expert in enumerate(flat_experts):
            expert_indices = expert.uid[len(self.beam_search.uid_prefix) :]
            expert_indices = list(map(int, expert_indices.split(UID_DELIMITER)))
            grid_indices[i] = torch.as_tensor(expert_indices, dtype=grid_indices.dtype)

        scores_per_dim = [
            dim_scores[flat_batch_indices, dim_indices] if len(flat_batch_indices) else torch.zeros(0, device=device)
            for dim_scores, dim_indices in zip(grid_scores, grid_indices.T)
        ]
        flat_scores = torch.sum(torch.stack(scores_per_dim, dim=0), dim=0)

        scores = torch.full((batch_size, max_num_experts), fill_value=-float("inf"), device=device)
        scores[flat_batch_indices, flat_local_indices] = flat_scores  # backprop-able w.r.t. flat_scores
        return scores

    @property
    def info(self):
        if self._expert_info is None:
            # grab some expert to set ensemble output shape
            proj_device = self.proj.weight.device
            dummy_scores_concat = self.proj(torch.randn(1, self.proj.in_features, device=proj_device))
            dummy_scores = dummy_scores_concat.cpu().detach().split_with_sizes(self.beam_search.grid_size, dim=-1)
            dummy_experts = self.beam_search.find_best_experts(dummy_scores, beam_size=1)
            self._expert_info = dummy_experts[0].info
        return self._expert_info


class _RemoteCallMany(torch.autograd.Function):
    """
    Internal autograd-friendly function that calls multiple experts on a batch of inputs and awaits responses
    This function that can recover from individual failures during forward and/or backward pass as long as at least
    one expert succeeds for each input. For user-friendly version of this function, use RemoteMixtureOfExperts module.

    Note: experts that failed during forward will be assigned zero outputs and marked as mask[i, j] = 0,
          experts that failed during backward will be treated as constants (i.e. gradients through them are zeros)
    """

    @classmethod
    def forward(
        cls,
        ctx,
        dummy,
        experts_per_sample: List[List[RemoteExpert]],
        k_min: int,
        backward_k_min: int,
        timeout_after_k_min: float,
        forward_timeout: Optional[float],
        backward_timeout: Optional[float],
        detect_anomalies: bool,
        allow_zero_outputs: bool,
        info: Dict[str, Any],
        *flat_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        assert not torch.is_grad_enabled()
        num_samples, max_experts = len(experts_per_sample), max(map(len, experts_per_sample))

        flat_inputs_cpu = []
        for tensor in flat_inputs:
            if detect_anomalies and not tensor.isfinite().all():
                raise ValueError("One of inputs has nan/inf values")
            flat_inputs_cpu.append(tensor.cpu())

        flat_inputs_per_sample = list(zip(*(x.split(1, dim=0) for x in flat_inputs_cpu)))
        assert len(experts_per_sample) == len(flat_inputs_per_sample) == num_samples

        # dispatch tasks to all remote experts collect responses
        pending_tasks: Dict[Future, Tuple[int, int]] = {}
        for i in range(num_samples):
            for j, expert in enumerate(experts_per_sample[i]):
                stub = get_server_stub(expert.p2p, expert.peer_id)
                serialized_tensors = (
                    serialize_torch_tensor(tensor, proto.compression)
                    for tensor, proto in zip(flat_inputs_per_sample[i], nested_flatten(info["forward_schema"]))
                )
                new_task = RemoteExpertWorker.run_coroutine(
                    expert_forward(expert.uid, flat_inputs_per_sample[i], serialized_tensors, stub),
                    return_future=True,
                )
                pending_tasks[new_task] = (i, j)

        responded_inds, alive_flat_outputs = cls._collect_responses(
            pending_tasks, num_samples, k_min, forward_timeout, timeout_after_k_min, detect_anomalies
        )
        if len(responded_inds) < k_min:
            raise TimeoutError(f"Forward pass: less than {k_min} responded within timeout")

        if not isinstance(info["outputs_schema"], tuple):
            outputs_schema = (info["outputs_schema"],)
        else:
            outputs_schema = info["outputs_schema"]
        outputs = nested_map(
            lambda descriptor: descriptor.make_zeros(num_samples, max_experts, device=flat_inputs[0].device),
            outputs_schema,
        )

        # assemble responses
        if len(responded_inds) > 0 or allow_zero_outputs:
            batch_inds, expert_inds = map(
                lambda x: torch.as_tensor(x, device=flat_inputs[0].device, dtype=torch.long),
                list(zip(*responded_inds)) or ([], []),
            )

            alive_flat_outputs_stacked = (torch.cat(outputs) for outputs in zip(*alive_flat_outputs))
            # torch tensors, i-th tensor is of shape [num_responded, *expert_outputs[i].shape]

            for output, response_stacked in zip(outputs, alive_flat_outputs_stacked):
                output[batch_inds, expert_inds] = response_stacked.to(output.device)

        else:
            raise RuntimeError("Forward pass: 0 experts responded within timeout and allow_zero_outputs is False")

        mask = torch.zeros([num_samples, max_experts], dtype=torch.bool, device=flat_inputs[0].device)
        mask[batch_inds, expert_inds] = True

        # save individual outputs for backward pass
        ctx.save_for_backward(batch_inds, expert_inds, *flat_inputs_cpu)
        ctx._saved_non_tensors = (
            info,
            backward_k_min,
            backward_timeout,
            timeout_after_k_min,
            experts_per_sample,
            detect_anomalies,
        )

        return (mask,) + outputs

    @classmethod
    @once_differentiable
    def backward(cls, ctx, *raw_grads: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        assert not torch.is_grad_enabled()
        (
            info,
            backward_k_min,
            backward_timeout,
            timeout_after_k_min,
            expert_per_sample,
            detect_anomalies,
        ) = ctx._saved_non_tensors
        alive_ii, alive_jj, *flat_inputs_cpu = ctx.saved_tensors

        dummy_grad_mask, *flat_grad_outputs = raw_grads

        flat_grad_outputs_cpu = []
        for tensor in flat_grad_outputs:
            if detect_anomalies and not tensor.isfinite().all():
                raise ValueError("One of gradients has nan/inf values")
            flat_grad_outputs_cpu.append(tensor.cpu())

        num_samples, max_experts = dummy_grad_mask.shape

        inputs_per_expert = zip(*(tensor[alive_ii].split(1, dim=0) for tensor in flat_inputs_cpu))
        grad_outputs_per_expert = zip(
            *(tensor[alive_ii, alive_jj].split(1, dim=0) for tensor in flat_grad_outputs_cpu)
        )
        backward_schema = tuple(nested_flatten((info["forward_schema"], info["outputs_schema"])))

        # dispatch tasks to all remote experts, collect responses
        pending_tasks = {}
        for i, j, inputs_ij, grad_outputs_ij in zip(
            alive_ii.cpu().numpy(), alive_jj.cpu().numpy(), inputs_per_expert, grad_outputs_per_expert
        ):
            expert: RemoteExpert = expert_per_sample[i.item()][j.item()]
            stub = get_server_stub(expert.p2p, expert.peer_id)
            inputs_and_grad_outputs = tuple(nested_flatten((inputs_ij, grad_outputs_ij)))
            serialized_tensors = (
                serialize_torch_tensor(tensor, proto.compression)
                for tensor, proto in zip(inputs_and_grad_outputs, backward_schema)
            )
            new_task = RemoteExpertWorker.run_coroutine(
                expert_backward(expert.uid, inputs_and_grad_outputs, serialized_tensors, stub), return_future=True
            )
            pending_tasks[new_task] = (i, j)

        survivor_inds, survivor_grad_inputs = cls._collect_responses(
            pending_tasks, num_samples, backward_k_min, backward_timeout, timeout_after_k_min, detect_anomalies
        )
        if len(survivor_inds) < backward_k_min:
            raise TimeoutError(f"Backward pass: less than {backward_k_min} experts responded within timeout")

        # assemble responses
        batch_inds, expert_inds = map(
            lambda x: torch.as_tensor(x, dtype=torch.long), list(zip(*survivor_inds)) or ([], [])
        )

        survivor_grad_inputs_stacked = (torch.cat(grad_inputs) for grad_inputs in zip(*survivor_grad_inputs))
        # torch tensors, i-th tensor is of shape [num_backward_survivors, *flat_inputs_cpu[i].shape]

        grad_inputs = nested_map(
            lambda descr: descr.make_zeros(num_samples, device=flat_grad_outputs[0].device),
            list(nested_flatten(info["forward_schema"])),
        )

        for grad_input, survivor_grad_stacked in zip(grad_inputs, survivor_grad_inputs_stacked):
            grad_input_per_expert = torch.zeros(  # gradient tensor with individual contributions from each expert
                (num_samples, max_experts, *grad_input.shape[1:]),
                device=survivor_grad_stacked.device,
                dtype=survivor_grad_stacked.dtype,
            )
            grad_input_per_expert[batch_inds, expert_inds] = survivor_grad_stacked
            grad_input.copy_(grad_input_per_expert.to(flat_grad_outputs[0].device).sum(dim=1))

        return (DUMMY, None, None, None, None, None, None, None, None, None, *grad_inputs)

    @staticmethod
    def _collect_responses(
        task_to_indices: Dict[Future, Tuple[int, int]],
        num_samples: int,
        k_min: int,
        timeout_total: Optional[float],
        timeout_after_k_min: Optional[float],
        detect_anomalies: bool,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[torch.Tensor, ...]]]:
        """await up to k_min results and any result submitted within timeout_after_k_min, cancel stragglers"""
        timeout_total = float("inf") if timeout_total is None else timeout_total
        timeout_after_k_min = float("inf") if timeout_after_k_min is None else timeout_after_k_min
        num_successful_tasks = [0 for _ in range(num_samples)]
        pending_samples = num_samples  # samples for which we have less than k_min results
        finished_indices, finished_outputs = [], []
        t_finish = time.perf_counter() + timeout_total
        pending_tasks = set(task_to_indices.keys())
        finished_tasks = Queue()

        try:
            # the algorithm below is essentially futures.as_completed, but for grpc.Future
            for task in pending_tasks:
                task.add_done_callback(finished_tasks.put)

            for _ in range(len(task_to_indices)):
                timeout = max(0.0, t_finish - time.perf_counter()) if t_finish != float("inf") else None
                task = finished_tasks.get(timeout=timeout)
                pending_tasks.discard(task)

                task_output = _process_dispatched_task(task, detect_anomalies)
                if task_output is not None:
                    finished_indices.append(task_to_indices[task])
                    finished_outputs.append(task_output)

                    # count how many successes we have for each input sample
                    sample_index = task_to_indices[task][0]
                    num_successful_tasks[sample_index] += 1
                    if num_successful_tasks[sample_index] == k_min:
                        pending_samples -= 1
                        if (
                            pending_samples <= 0
                        ):  # all tasks finished, await stragglers for at most timeout_after_k_min
                            t_finish = min(t_finish, time.perf_counter() + timeout_after_k_min)

        except Empty:
            pass  # we reached t_finish, this is normal behavior
        finally:
            for task in pending_tasks:
                task.cancel()
        return finished_indices, finished_outputs


def _process_dispatched_task(task: Future, detect_anomalies: bool) -> Optional[Tuple[torch.Tensor]]:
    if task.exception() or task.cancelled():
        logger.warning(f"Task {task} failed: {type(task.exception())}")
        return None

    outputs = task.result()
    for tensor in outputs:
        if detect_anomalies and not tensor.isfinite().all():
            logger.error(f"Task {task} failed: output tensor contains nan/inf values")
            return None

    return outputs
