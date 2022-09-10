"""
Compression strategies that reduce the network communication in .averaging, .optim and .moe
"""

from hivemind.compression.adaptive import PerTensorCompression, RoleAdaptiveCompression, SizeAdaptiveCompression
from hivemind.compression.base import CompressionBase, CompressionInfo, NoCompression, TensorRole
from hivemind.compression.floating import Float16Compression, ScaledFloat16Compression
from hivemind.compression.quantization import BlockwiseQuantization, Quantile8BitQuantization, Uniform8BitQuantization
from hivemind.compression.serialization import (
    deserialize_tensor_stream,
    deserialize_torch_tensor,
    serialize_torch_tensor,
)
