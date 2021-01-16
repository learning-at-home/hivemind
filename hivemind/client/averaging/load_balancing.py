from typing import Sequence
import numpy as np
import scipy.optimize

from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


def load_balance_peers(vector_size: int, throughputs: np.ndarray, min_size: int = 0) -> np.ndarray:
    """
    Find an optimal partitioning of weights for butterfly all-reduce given peer throughputs.

    This method solves an optimization problem to minimize the total allreduce time.
    In butterfly all-reduce, each peer acts both as a "client" and as an "aggregator":
    * a "client" splits his local vector into shards and sends each shard to one peer, then downloads the average
    * an "aggregator" receives a certain part of vector components from all peers, aggregates and returns the average

    Peer i network load as a "client" = vector_size * (1 - fraction_assigned_to_peer_i)
    Peer i network load as an "aggregator" = vector_size * (group_size - 1) * fraction_assigned_to_peer_i
    Peer i total communication = vector_size * [1 + (group_size - 2) * fraction_assigned_to_peer_i]
    Total time = max_i (total_communication_for_peer_i / throughputs[i])

    We find optimal vector fractions by minimizing the total time (using https://tinyurl.com/minimax-to-lp )
    Then, we use Hagenbach-Bishoff apportionment to split the finite vector based on the optimal fractions.

    :param vector_size: total size of the averaged vector (in elements, not bytes)
    :param throughputs: 1d array of throughputs for each peer, typically min(upload speed, download speed)
    :param min_size: peers that can aggregate less than this many elements will be assigned nothing
    :returns: an integer array where i-th element is the number of weights assigned to i-th peer
    """
    assert np.min(throughputs) > 0
    permutation = np.argsort(-throughputs)
    throughputs = throughputs[permutation]

    group_size = len(throughputs)
    num_variables = group_size + 1  # [w_1, ..., w_N, ksi]

    c = np.zeros(num_variables)
    c[-1] = 1.0  # optimize w.r.t. ksi

    # the constraints below are tuples (A, b) such that Ax <= b
    nonnegative_weights = -np.eye(group_size, M=num_variables), np.zeros(group_size)
    weights_sum_to_one = c[None, :] - 1.0, np.array([-1.0])
    coeff_per_variable = (group_size - 2.0) / throughputs
    coeff_matrix_minus_ksi = np.hstack([np.diag(coeff_per_variable), -np.ones((group_size, 1))])
    ksi_is_maximum = coeff_matrix_minus_ksi, -1.0 / throughputs

    A, b = list(map(np.concatenate, zip(nonnegative_weights, weights_sum_to_one, ksi_is_maximum)))

    solution = scipy.optimize.linprog(c, A_ub=A, b_ub=b)
    if solution.success:
        peer_fractions = solution.x[:group_size]
        if np.max(peer_fractions) >= min_size / float(vector_size):
            peer_fractions[peer_fractions < min_size / float(vector_size)] = 0.0
    else:
        logger.error(f"Failed to solve load-balancing for bandwidths {throughputs}.")
        peer_fractions = np.ones(group_size) / group_size

    return np.asarray(hagenbach_bishoff(vector_size, peer_fractions))[np.argsort(permutation)]


def hagenbach_bishoff(vector_size: int, scores: Sequence[float]) -> Sequence[int]:
    """
    Split a vector between participants based on continuous fractions.
    https://en.wikipedia.org/wiki/Hagenbach-Bischoff_system

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
