import heapq
import random
import threading
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn

import hivemind
from hivemind import nested_compare, nested_flatten, get_dht_time
from hivemind.moe.client.expert import _RemoteModuleCall, DUMMY
from hivemind.moe.server.expert_uid import ExpertUID
from hivemind.utils import DHTExpiration


class LoadBalancedExpert(nn.Module):
    """
    A torch module that dynamically assigns weights to one RemoteExpert from a pool.
    ToDo docstring, similar to RemoteMixtureOfExperts
    """

    def __init__(
        self,
        *,
        dht: hivemind.DHT,
        uid_prefix: str,
        grid_size: Tuple[int, ...],
        forward_timeout: Optional[float] = None,
        backward_timeout: Optional[float] = None,
        detect_anomalies: bool = False,
        refresh_period: float = 30.,
        **dht_kwargs,
    ):
        super().__init__()
        assert len(grid_size) == 1, "only 1d grids are supported for now"
        self.dht, self.dht_kwargs, self.uid_prefix, self.grid_size = dht, dht_kwargs, uid_prefix, grid_size
        self.forward_timeout, self.backward_timeout = forward_timeout, backward_timeout
        self.detect_anomalies = detect_anomalies

        self.active_experts: Dict[ExpertUID, DHTExpiration] = {}
        self.banned_experts: Dict[ExpertUID, DHTExpiration] = {}
        self.expert_queue: List[Tuple[float, float, ExpertUID]] = []
        self._expert_info = None  # expert['info'] from one of experts in the grid
        self.refresh_period, self.last_refresh = refresh_period, 0.0
        self.should_refresh_experts, self.refresh_complete = threading.Event(), threading.Event()

    def fetch_experts_in_background(self):
        while True:
            time_to_next_update = max(0.0, self.last_update + self.refresh_period - get_dht_time())
            try:
                self.should_refresh_experts.wait(timeout=time_to_next_update)
                # update triggered by main thread
            except TimeoutError:
                pass  # update triggered by refresh_period

            TODO_FETCH_MORE_EXPERTS_HERE

            # state:
            # * available_experts: Dict[uid -> (EMA, expiration)] - experts that take part in load balancing
            # * maintain blacklist: Dict[uid -> expiration] - experts banned until expiration for a non-response
            # * maintain a min-heap queue of (load, rng, expert) tuples
            # * update_triggered, update_finished: threading.Event
            #
            # update experts in background, while True:
            # * wait for 30s or for update_triggered, whichever comes first
            # * for expert, expiration_time in fetch_experts_from_dht():
            # * * if expert in banned and expiration_time <= self.blacklist[expert]:
            # * * * continue # expert is still banned
            # * * else: add expert to min-heap, intitialize throughput
            # * update_complete.set()
            #
            # on forward/backward:
            # pass (queue, blacklist, update_triggered, update_finished) to the autograd function
            #
            # forward/backward autograd function
            # while True:
            # * while len(available experts) == 0:
            # * * update_finished.clear()
            # * * update_triggered.set()
            # * * update_finished.wait()
            # * with threading.lock:
            # * * load, _, expert = queue.heappop_min()
            # * * expert_throughput_ema, expert_expiration_time = get ema from dict
            # * * task_complexity = batch_size * 1.5 if forward else 2.5 # if backward
            # * * queue.heappush (load + task_complexity / expert_throughput_ema, new_rng, expert)
            # * * try:
            # * * * with measure_ema(start=now, batch_size=batch_size) as measured_ema:
            # * * * * outputs = call_forward_or_backward()
            # * * * expert_throughput_ema.update(measured_ema)
            # * * * return outputs      # <--------- this is the desired exit point
            # * * except DidNotRespondCorrectly:
            # * * * banned_experts[expert] = expert_expiration_time
            # * * continue # try again

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor):
        """
        Call one of the RemoteExperts for the specified inputs and return output. Compatible with pytorch.autograd.

        :param args: input tensors that will be passed to each expert after input, batch-first
        :param kwargs: extra keyword tensors that will be passed to each expert, batch-first
        :returns: averaged predictions of all experts that delivered result on time, nested structure of batch-first
        """
        assert len(kwargs) == len(self.info["keyword_names"]), f"Keyword args should be {self.info['keyword_names']}"
        kwargs = {key: kwargs[key] for key in self.info["keyword_names"]}



        if self._expert_info is None:
            raise NotImplementedError()
        # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors

        forward_inputs = (args, kwargs)

        if not nested_compare(forward_inputs, self.info["forward_schema"]):
            raise TypeError(f"Inputs do not match expert input schema. Did you pass the right number of parameters?")

        flat_outputs = _RemoteModuleCall.apply(DUMMY, self.uid, self.stub, self.info, *nested_flatten(forward_inputs))
        # Note: we send DUMMY to prevent torch from excluding expert from backward if no other inputs require grad
        return nested_pack(flat_outputs, structure=self.info["outputs_schema"])

    @property
    def info(self):
        if self._expert_info is None:
            # grab some expert to set ensemble output shape
            proj_device = self.proj.weight.device
            dummy_scores_concat = self.proj(torch.randn(1, self.proj.in_features, device=proj_device))
            dummy_scores = dummy_scores_concat.cpu().split_with_sizes(self.beam_search.grid_size, dim=-1)
            dummy_experts = self.beam_search.find_best_experts(dummy_scores, beam_size=1)
            self._expert_info = dummy_experts[0].info
        return self._expert_info
