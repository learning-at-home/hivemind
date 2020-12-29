import torch
from torch import nn as nn


class FeedforwardBlock(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hid_dim, 4 * hid_dim),
            nn.LayerNorm(4 * hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * hid_dim, 4 * hid_dim),
            nn.LayerNorm(4 * hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * hid_dim, hid_dim),
        )

    def forward(self, x):
        return x + self.layers(x)


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

        self.activation = torch.nn.GELU()

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


class NopExpert(nn.Sequential):
    def __init__(self, hid_dim):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(0), requires_grad=True)

    def forward(self, x):
        return x.clone()
