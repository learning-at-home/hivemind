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

    def __init__(self, *, in_features, grid_size: Tuple[int, ...], dht: hivemind.DHT, uid_prefix: str, k_best: int,
                 k_min: int = 1, forward_timeout: Optional[float] = None, timeout_after_k_min: Optional[float] = None,
                 backward_k_min: int = 1, backward_timeout: Optional[float] = None, detect_anomalies: bool = False,
                 **dht_kwargs):
        super().__init__()
        if not uid_prefix.endswith(hivemind.dht.UID_DELIMITER):
            uid_prefix += hivemind.dht.UID_DELIMITER
            logger.info(f"Prefix must end with '{hivemind.dht.UID_DELIMITER}'. New prefix: '{uid_prefix}' .")
        assert hivemind.dht.is_valid_prefix(uid_prefix), f"Prefix '{uid_prefix}' is invalid."
        self.dht, self.grid_size, self.uid_prefix, self.dht_kwargs = dht, grid_size, uid_prefix, dht_kwargs
        self.k_best, self.k_min, self.backward_k_min = k_best, k_min, backward_k_min
        self.forward_timeout, self.backward_timeout = forward_timeout, backward_timeout
        self.timeout_after_k_min = timeout_after_k_min
        self.detect_anomalies = detect_anomalies

        self.proj = nn.Linear(in_features, sum(grid_size))  # jointly predict logits for all grid dimensions
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
        grid_scores = self.proj(input_for_gating).split_with_sizes(self.grid_size, dim=-1)

        chosen_experts: List[List[RemoteExpert]] = self.dht.batch_find_best_experts(
            self.uid_prefix, [scores.detach().cpu().numpy() for scores in grid_scores], self.k_best, **self.dht_kwargs)

        if self._expert_info is None:
            try:
                self._expert_info = next((expert.info for experts_i in chosen_experts for expert in experts_i))
            except grpc.RpcError as e:
                logger.warning(f"Failed to get RemoteMixtureOfExperts.output_shape: {e}")

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
        return nested_pack(averaged_outputs_flat, self.info['outputs_schema'])

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
            expert_indices = expert.uid[len(self.uid_prefix):]
            expert_indices = list(map(int, expert_indices.split(hivemind.dht.UID_DELIMITER)))
            grid_indices[i] = torch.as_tensor(expert_indices, dtype=grid_indices.dtype)

        scores_per_dim = [
            dim_scores[flat_batch_indices, dim_indices] if len(flat_batch_indices) else torch.zeros(0)
            for dim_scores, dim_indices in zip(grid_scores, grid_indices.T)]
        flat_scores = torch.sum(torch.stack(scores_per_dim, dim=0), dim=0)

        scores = torch.full((batch_size, max_num_experts), fill_value=-float('inf'), device=grid_scores[0].device)
        scores[flat_batch_indices, flat_local_indices] = flat_scores  # backprop-able w.r.t. flat_scores
        return scores

    @property
    def info(self):
        if self._expert_info is None:
            # grab some expert to set ensemble output shape
            proj_device = self.proj.weight.device
            dummy_scores_concat = self.proj(torch.randn(1, self.proj.in_features, device=proj_device))
            dummy_scores = dummy_scores_concat.cpu().split_with_sizes(self.grid_size, dim=-1)
            dummy_experts = self.loop.run_until_complete(self.beam_search(dummy_scores, k_best=1))
            self._expert_info = dummy_experts[0].info
        return self._expert_info


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
                detect_anomalies: bool, info: Dict[str, Any], *flat_inputs: torch.Tensor) -> Tuple[torch.Tensor]:
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
        pending_tasks: Dict[grpc.Future, Tuple[int, int]] = {}
        for i in range(num_samples):
            for j, expert in enumerate(experts_per_sample[i]):
                input_tensors = [serialize_torch_tensor(tensor, proto.compression) for tensor, proto in zip(
                    flat_inputs_per_sample[i], nested_flatten(info['forward_schema']))]
                stub: runtime_grpc.ConnectionHandlerStub = _get_expert_stub(expert.endpoint)
                new_task = stub.forward.future(runtime_pb2.ExpertRequest(uid=expert.uid, tensors=input_tensors))
                pending_tasks[new_task] = (i, j)

        alive_grid_indices, alive_flat_outputs = cls._collect_responses(
            pending_tasks, num_samples, k_min, forward_timeout, timeout_after_k_min, detect_anomalies)
        if len(alive_grid_indices) == 0:
            raise TimeoutError("Forward pass: no alive experts responded within timeout.")

        # assemble responses
        alive_ii, alive_jj = map(torch.as_tensor, zip(*alive_grid_indices))
        mask = torch.zeros([num_samples, max_experts], dtype=torch.bool, device=flat_inputs[0].device)
        mask[alive_ii, alive_jj] = True

        alive_flat_outputs_stacked = (torch.cat(outputs) for outputs in zip(*alive_flat_outputs))
        # torch tensors, i-th tensor is of shape [num_responded, *expert_outputs[i].shape]

        outputs = []
        for response_stacked in alive_flat_outputs_stacked:
            output = torch.zeros(
                [num_samples, max_experts, *response_stacked.shape[1:]], device=response_stacked.device,
                dtype=response_stacked.dtype, requires_grad=response_stacked.requires_grad)
            output[alive_ii, alive_jj] = response_stacked
            outputs.append(output.to(flat_inputs[0].device))

        # save individual outputs for backward pass
        ctx.save_for_backward(alive_ii, alive_jj, *flat_inputs_cpu)
        ctx._saved_non_tensors = (info, backward_k_min, backward_timeout, timeout_after_k_min, experts_per_sample,
                                  detect_anomalies)
        return (mask,) + tuple(outputs)

    @classmethod
    @once_differentiable
    def backward(cls, ctx, *raw_grads: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        assert not torch.is_grad_enabled()
        (info, backward_k_min, backward_timeout, timeout_after_k_min, expert_per_sample,
         detect_anomalies) = ctx._saved_non_tensors
        alive_ii, alive_jj, *flat_inputs_cpu = ctx.saved_tensors

        dummy_grad_mask, *flat_grad_outputs = raw_grads

        flat_grad_outputs_cpu = []
        for tensor in flat_grad_outputs:
            if detect_anomalies and not tensor.isfinite().all():
                raise ValueError("One of gradients has nan/inf values")
            flat_grad_outputs_cpu.append(tensor.cpu())

        num_samples, max_experts = dummy_grad_mask.shape

        inputs_per_expert = zip(*(tensor[alive_ii].split(1, dim=0) for tensor in flat_inputs_cpu))
        grad_outputs_per_expert = zip(*(tensor[alive_ii, alive_jj].split(1, dim=0) for tensor in flat_grad_outputs_cpu))
        backward_schema = tuple(nested_flatten((info["forward_schema"], info["outputs_schema"])))

        # dispatch tasks to all remote experts, collect responses
        pending_tasks = {}
        for i, j, inputs_ij, grad_outputs_ij in zip(alive_ii.cpu().numpy(), alive_jj.cpu().numpy(),
                                                    inputs_per_expert, grad_outputs_per_expert):
            expert = expert_per_sample[i.item()][j.item()]
            stub: runtime_grpc.ConnectionHandlerStub = _get_expert_stub(expert.endpoint)
            inputs_and_grad_outputs = tuple(nested_flatten((inputs_ij, grad_outputs_ij)))
            tensors_serialized = [serialize_torch_tensor(tensor, proto.compression)
                                  for tensor, proto in zip(inputs_and_grad_outputs, backward_schema)]
            new_task = stub.backward.future(runtime_pb2.ExpertRequest(uid=expert.uid, tensors=tensors_serialized))
            pending_tasks[new_task] = (i, j)

        backward_survivor_indices, survivor_grad_inputs = cls._collect_responses(
            pending_tasks, num_samples, backward_k_min, backward_timeout, timeout_after_k_min, detect_anomalies)
        if len(backward_survivor_indices) == 0:
            raise TimeoutError("Backward pass: no alive experts responded within timeout.")

        # assemble responses
        backward_survivor_ii, backward_survivor_jj = map(torch.as_tensor, zip(*backward_survivor_indices) or ([], []))

        survivor_grad_inputs_stacked = (torch.cat(grad_inputs) for grad_inputs in zip(*survivor_grad_inputs))
        # torch tensors, i-th tensor is of shape [num_backward_survivors, *flat_inputs_cpu[i].shape]

        grad_inputs = []
        for i, survivor_grad_stacked in enumerate(survivor_grad_inputs_stacked):
            grad_input_per_expert = torch.zeros(  # gradient tensor with individual contributions from each expert
                (num_samples, max_experts, *flat_inputs_cpu[i].shape[1:]),
                device=survivor_grad_stacked.device, dtype=survivor_grad_stacked.dtype)
            grad_input_per_expert[backward_survivor_ii, backward_survivor_jj] = survivor_grad_stacked

            # sum gradients from each expert
            grad_inputs.append(grad_input_per_expert.to(flat_grad_outputs[0].device).sum(dim=1))

        return (DUMMY, None, None, None, None, None, None, None, None, *grad_inputs)

    @staticmethod
    def _collect_responses(task_to_indices: Dict[grpc.Future, Tuple[int, int]], num_samples: int, k_min: int,
                           timeout_total: Optional[float], timeout_after_k_min: Optional[float], detect_anomalies: bool
                           ) -> Tuple[List[Tuple[int, int]], List[Tuple[torch.Tensor, ...]]]:
        """ await up to k_min results and any result submitted within timeout_after_k_min, cancel stragglers """
        timeout_total = float('inf') if timeout_total is None else timeout_total
        timeout_after_k_min = float('inf') if timeout_after_k_min is None else timeout_after_k_min
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
                timeout = max(0.0, t_finish - time.perf_counter()) if t_finish != float('inf') else None
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
                        if pending_samples <= 0:  # all tasks finished, await stragglers for at most timeout_after_k_min
                            t_finish = min(t_finish, time.perf_counter() + timeout_after_k_min)

        except Empty:
            pass  # we reached t_finish, this is normal behavior
        finally:
            for task in pending_tasks:
                task.cancel()
        return finished_indices, finished_outputs


def _process_dispatched_task(task: grpc.Future, detect_anomalies: bool) -> Optional[Tuple[torch.Tensor]]:
    if task.exception() or task.cancelled():
        logger.warning(f"Task {task} failed: {type(task.exception())}")
        return None

    deserialized_outputs = []
    for tensor in task.result().tensors:
        deserialized_tensor = deserialize_torch_tensor(tensor)
        if detect_anomalies and not deserialized_tensor.isfinite().all():
            logger.error(f"Task {task} failed: output tensor contains nan/inf values")
            return None
        deserialized_outputs.append(deserialized_tensor)

    return tuple(deserialized_outputs)
