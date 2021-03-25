import time
import argparse
import torch

from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils import serialize_torch_tensor, deserialize_torch_tensor


def benchmark_compression(tensor: torch.Tensor, compression_type: CompressionType) -> float:
    t = time.time()
    deserialize_torch_tensor(serialize_torch_tensor(tensor, compression_type))
    return time.time() - t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=50000000, required=False)
    parser.add_argument('--seed', type=int, default=7348, required=False)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    X = torch.randn(args.size)

    for name, compression_type in CompressionType.items():
        print(f"compression type: {name}; time: {benchmark_compression(X, compression_type)}")
