import argparse
import time

import torch

from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


def benchmark_compression(tensor: torch.Tensor, compression_type: CompressionType) -> float:
    t = time.time()
    deserialize_torch_tensor(serialize_torch_tensor(tensor, compression_type))
    return time.time() - t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10000000, required=False)
    parser.add_argument("--seed", type=int, default=7348, required=False)
    parser.add_argument("--num_iters", type=int, default=30, required=False)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    X = torch.randn(args.size)

    for name, compression_type in CompressionType.items():
        tm = 0
        for i in range(args.num_iters):
            tm += benchmark_compression(X, compression_type)
        tm /= args.num_iters
        logger.info(f"Compression type: {name}, time: {tm}")
