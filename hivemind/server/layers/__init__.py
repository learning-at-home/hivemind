import torch
import torch.nn as nn

from hivemind.server.layers.dropout import DeterministicDropout


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

    def forward(self, src):
        src.transpose_(0, 1)
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src.transpose_(0, 1)
        return src


class NopExpert(nn.Sequential):
    def __init__(self, hid_dim):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(0), requires_grad=True)

    def forward(self, x):
        return x.clone()


class DeterministicDropoutNetwork(nn.Module):
    def __init__(self, hid_dim, dropout_prob):
        super().__init__()
        self.linear_in = nn.Linear(hid_dim, 2 * hid_dim)
        self.activation = nn.ReLU()
        self.dropout = DeterministicDropout(dropout_prob)
        self.linear_out = nn.Linear(2 * hid_dim, hid_dim)

    def forward(self, x, mask):
        x = self.linear_in(self.dropout(x, mask))
        return self.linear_out(self.activation(x))


name_to_block = {'ffn': lambda hid_dim: FeedforwardBlock(hid_dim),
                 'transformer': lambda hid_dim: TransformerEncoderLayer(hid_dim, nhead=16),
                 'nop': lambda hid_dim: NopExpert(hid_dim),
                 'det_dropout': lambda hid_dim: DeterministicDropoutNetwork(hid_dim, dropout_prob=0.2)}
name_to_input = {'ffn': lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim)),
                 'transformer': lambda batch_size, hid_dim: torch.empty((batch_size, 512, hid_dim)),
                 'nop': lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim)),
                 'det_dropout': lambda batch_size, hid_dim:
                 (torch.empty((batch_size, hid_dim)), torch.randint(0, 1, (batch_size, hid_dim)))}
