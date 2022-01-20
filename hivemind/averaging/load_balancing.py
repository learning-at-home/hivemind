from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.optimize

from hivemind.utils.logging import get_logger

logger = get_logger(__name__)

LOAD_BALANCING_LP_DECIMALS = 9


def load_balance_peers(vector_size, bandwidths: Sequence[Optional[float]], min_size: int = 0) -> Tuple[int, ...]:
    """
    Find an optimal partitioning of weights for butterfly all-reduce given peer bandwidths.
    :param vector_size: total size of the averaged vector (in elements, not bytes)
    :param bandwidths: 1d array of non-negative bandwidths for each peer capable of averaging
      zeros stand for client-only participants, None represents "not specified" (resolved as mean of other pears)
    :param min_size: peers that can aggregate less than this many elements will be assigned nothing
    :returns: an integer array where i-th element is the number of weights assigned to i-th peer
    """
    specified_bandwidth = [item for item in bandwidths if item is not None and item > 0]

    if specified_bandwidth:
        default_bandwidth = np.mean(specified_bandwidth)
        bandwidths = [item if item is not None else default_bandwidth for item in bandwidths]
        scores = optimize_parts_lp(vector_size, np.asarray(bandwidths), min_size)
    else:
        assert not all(item == 0 for item in bandwidths), "Must have at least one nonzero bandwidth"
        scores = np.asarray([1.0 if item is None else 0.0 for item in bandwidths])

    # TODO(jheuristic) we no longer need hagenbach-bishoff with new AllReduceRunner
    return tuple(hagenbach_bishoff(vector_size, scores))


def optimize_parts_lp(vector_size: int, bandwidths: np.ndarray, min_size: int = 0) -> np.ndarray:
    """
    This method solves an optimization problem to minimize the total allreduce time.
    In butterfly all-reduce, each peer acts both as a "client" and as an "aggregator":
    * a "client" splits his local vector into shards and sends each shard to one peer, then downloads the average
    * an "aggregator" receives a certain part of vector components from all peers, aggregates and returns the average

    Peer i network load as a "client" = vector_size * (1 - fraction_assigned_to_peer_i)
    Peer i network load as an "aggregator" = vector_size * (group_size - 1) * fraction_assigned_to_peer_i
    Peer i total communication = vector_size * [1 + (group_size - 2) * fraction_assigned_to_peer_i]
    Total time = max_i (total_communication_for_peer_i / bandwidths[i])

    We solve this optimization problem by reducing it to linear programming with a minimax reduction
    (see lecture notes: https://www.usna.edu/Users/math/dphillip/sa305.s15/phillips/lessons/32/32.pdf )

    :returns: a vector of "scores", i-th score is proportional to the fraction of weights assigned to i-th peer
    """
    assert np.all(bandwidths >= 0) and np.any(bandwidths > 0)
    bandwidths = np.asarray(bandwidths, dtype=np.float64)
    permutation = np.argsort(-bandwidths)
    bandwidths = bandwidths[permutation]
    is_nonzero = bandwidths != 0

    group_size = len(bandwidths)
    num_variables = group_size + 1  # [w_1, ..., w_N, xi]

    c = np.zeros(num_variables, dtype=np.float64)
    c[-1] = 1.0  # optimize w.r.t. xi

    # the constraints below are tuples (A, b) such that Ax <= b
    nonnegative_weights = -np.eye(group_size, num_variables, dtype=c.dtype), np.zeros(group_size, c.dtype)
    weights_sum_to_one = c[None, :] - 1.0, np.array([-1.0])
    coeff_per_variable = (group_size - 2.0) / np.maximum(bandwidths, 10 ** -LOAD_BALANCING_LP_DECIMALS)
    coeff_matrix_minus_xi = np.hstack([np.diag(coeff_per_variable), -np.ones((group_size, 1), c.dtype)])
    xi_is_maximum = coeff_matrix_minus_xi[is_nonzero], -1.0 / bandwidths[is_nonzero]
    force_max_weights = np.eye(group_size, M=num_variables, dtype=c.dtype), is_nonzero.astype(c.dtype)

    A, b = list(map(np.concatenate, zip(nonnegative_weights, weights_sum_to_one, xi_is_maximum, force_max_weights)))

    solution = scipy.optimize.linprog(c, A_ub=A, b_ub=b, method="interior-point")
    if solution.success:
        peer_scores = solution.x[:group_size]
        # if some peers have less than min_size elements, transfer their share to other peers (if any)
        if np.max(peer_scores) >= min_size / float(vector_size):
            peer_scores[peer_scores < min_size / float(vector_size)] = 0.0
        peer_scores = np.round(peer_scores, LOAD_BALANCING_LP_DECIMALS)
    else:
        logger.error(f"Failed to solve load-balancing for bandwidths {bandwidths}")
        peer_scores = np.ones(group_size, c.dtype)

    return peer_scores[np.argsort(permutation)]


def hagenbach_bishoff(vector_size: int, scores: Sequence[float]) -> Sequence[int]:
    """
    Split a vector between participants based on continuous fractions.
    https://en.wikipedia.org/wiki/Hagenbach-Bischoff_system
    The code is based on https://github.com/crflynn/voting

    :param vector_size: the total number of elements to be split
    :param scores: real-valued vector fractions for each peer
    :returns: integer-valued partitions assigned to every peer
    """
    total_score = sum(scores)
    allocated = [int(vector_size * score_i / total_score) for score_i in scores]
    while sum(allocated) < vector_size:
        quotients = [score / (allocated[idx] + 1) for idx, score in enumerate(scores)]
        idx_max = quotients.index(max(quotients))
        allocated[idx_max] += 1
    return allocated
