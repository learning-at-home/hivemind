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
    serialized = serialize_torch_tensor(tensor, compression_type)
    result = deserialize_torch_tensor(serialized)
    return time.time() - t, (tensor - result).square().mean(), serialized.ByteSize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10_000_000, required=False)
    parser.add_argument("--seed", type=int, default=7348, required=False)
    parser.add_argument("--num_iters", type=int, default=30, required=False)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    X = torch.randn(args.size)

    for name, compression_type in CompressionType.items():
        tm = 0
        distortion = 0
        bytes = 0
        for i in range(args.num_iters):
            iter_time, iter_distortion, size = benchmark_compression(X, compression_type)
            tm += iter_time
            distortion += iter_distortion
            bytes += size
        tm /= args.num_iters
        distortion /= args.num_iters
        bytes /= args.num_iters
        logger.info(f"Compression type: {name}, time: {tm:.5f}, distortion: {distortion:.5f}, size: {int(bytes):d}")
