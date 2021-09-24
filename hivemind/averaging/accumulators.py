import dataclasses
from abc import ABC
from typing import Callable, Optional

import torch


class AccumulatorBase(ABC):
    def accumulate_part(self, tensor: torch.Tensor, weight: float) -> None:
        ...

    def reduce(self) -> torch.Tensor:
        ...


AccumulatorFactory = Callable[[torch.Size, int], AccumulatorBase]


class MeanAccumulator(AccumulatorBase):
    def __init__(self, part_shape: torch.Size, _n_peers: int):
        self._accumulator = torch.zeros(part_shape)
        self._denominator = 0.0

    def accumulate_part(self, tensor_part: torch.Tensor, weight: float) -> None:
        self._accumulator.add_(tensor_part, alpha=weight)
        self._denominator += weight

    def reduce(self) -> torch.Tensor:
        return self._accumulator.div_(self._denominator)


class CenteredClipAccumulator(AccumulatorBase):
    def __init__(self, part_shape: torch.Size, n_peers: int, **kwargs):
        self._kwargs = kwargs

        self._tensors = torch.empty([n_peers] + part_shape)
        self._weights = torch.empty(n_peers)
        self._index = 0

    def accumulate_part(self, tensor_part: torch.Tensor, weight: float) -> None:
        self._tensors[self._index] = tensor_part
        self._weights[self._index] = weight
        self._index += 1

    def reduce(self) -> torch.Tensor:
        clipped = centered_clip(self._tensors, self._weights, **self._kwargs)
        return clipped.result


@dataclasses.dataclass(frozen=True)
class CenteredClipResult:
    result: torch.Tensor
    n_clipped: torch.Tensor
    last_step_delta: torch.Tensor


def centered_clip(
    input_tensors: torch.Tensor,
    weights: torch.Tensor,
    tau: float = 1.0,
    n_iters: int = 20,
    stop_delta: Optional[float] = None,
) -> CenteredClipResult:
    """
    Optimized implementation of CenteredClip from [Karimireddy, 2021].
    Intended to be used in a decentralized fashion as in [Gorbunov, 2021].

    :stop_delta: Stop iterations early if the ``L_inf`` norm of the last step is less than ``stop_delta``.
                 Note: if this option is used, the step norm calculations may increase the time per iteration by ~25%.

    References:

    [Karimireddy, 2021] Karimireddy, Sai Praneeth, Lie He, and Martin Jaggi. "Learning from history for byzantine
    robust optimization." International Conference on Machine Learning. PMLR, 2021.

    [Gorbunov, 2021] Gorbunov, Eduard, Alexander Borzunov, Michael Diskin, and Max Ryabinin.
    "Secure Distributed Training at Scale." arXiv preprint arXiv:2106.11257 (2021).
    """

    with torch.no_grad():
        n_peers = input_tensors.shape[0]
        result_shape = input_tensors.shape[1:]

        input_tensors = input_tensors.flatten(start_dim=1)
        weights /= weights.sum()

        # This finds medians faster than torch.median() and torch.quantile(q=0.5),
        # see https://github.com/pytorch/pytorch/issues/51450
        sorted_tensors = input_tensors.sort(dim=0).values
        result = sorted_tensors[n_peers // 2].clone()
        delta = None

        diff = torch.sub(input_tensors, result, out=sorted_tensors)  # Reuse memory from `sorted_tensors`
        for _ in range(n_iters):
            norms = diff.norm(dim=1)
            coeffs = weights * torch.minimum(torch.tensor(1.0), tau / norms)

            if stop_delta is not None:
                prev_diff = result[...] = diff[0]  # Reuse memory from `result`

            # We only need to update `diff` (not `result`) between iterations
            diff.addmm_(-coeffs.repeat(n_peers, 1), diff)

            if stop_delta is not None:
                delta = prev_diff.sub_(diff[0]).max()
                if delta < stop_delta:
                    break
        torch.sub(input_tensors[0], diff[0], out=result)

        return CenteredClipResult(
            result=result.reshape(result_shape), n_clipped=(tau < norms).sum(), last_step_delta=delta
        )
