import time

import torch
from torch import nn as nn

from hivemind.moe.server.layers.custom_experts import register_expert_class


# https://github.com/huggingface/transformers/blob/master/src/transformers/activations.py
@torch.jit.script
def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


ffn_sample_input = lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim))


@register_expert_class("ffn", ffn_sample_input)
class FeedforwardBlock(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.ffn = nn.Linear(hid_dim, 4 * hid_dim)
        self.ffn_output = nn.Linear(4 * hid_dim, hid_dim)
        self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-12)

    def forward(self, x):
        ffn_output = self.ffn(x)
        ffn_output = gelu_fast(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        return self.layer_norm(x + ffn_output)


class TransformerEncoderLayer(nn.Module):
    """
    A slight modification of torch.nn.TransformerEncoderLayer which allows for torch.jit scripting
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = gelu_fast

    def forward(self, src, src_key_padding_mask=None):
        # (N, S, E) -> (S, N, E)
        src = src.transpose(0, 1)

        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # (S, N, E) -> (N, S, E)
        src = src.transpose(0, 1)
        return src


transformer_sample_input = lambda batch_size, hid_dim: (
    torch.empty((batch_size, 128, hid_dim)),
    torch.empty((batch_size, 128), dtype=torch.bool),
)


@register_expert_class("transformer", transformer_sample_input)
class TunedTransformer(TransformerEncoderLayer):
    def __init__(self, hid_dim):
        super().__init__(hid_dim, dim_feedforward=4 * hid_dim, nhead=16)


nop_sample_input = lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim))


@register_expert_class("nop", nop_sample_input)
class NopExpert(nn.Sequential):
    def __init__(self, hid_dim):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(0), requires_grad=True)

    def forward(self, x):
        return x.clone()


@register_expert_class("nop_delay", nop_sample_input)
class DelayedNopExpert(nn.Sequential):
    def __init__(self, hid_dim, delay=0.5):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(0), requires_grad=True)
        self.delay = delay

    def forward(self, x):
        time.sleep(self.delay)
        return x.clone()
