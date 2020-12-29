import torch

from hivemind.server.layers.common import FeedforwardBlock, TransformerEncoderLayer, NopExpert
from hivemind.server.layers.dropout import DeterministicDropout, DeterministicDropoutNetwork

name_to_block = {'ffn': lambda hid_dim: FeedforwardBlock(hid_dim),
                 'transformer': lambda hid_dim: TransformerEncoderLayer(hid_dim, dim_feedforward=4 * hid_dim, nhead=16),
                 'nop': lambda hid_dim: NopExpert(hid_dim),
                 'det_dropout': lambda hid_dim: DeterministicDropoutNetwork(hid_dim, dropout_prob=0.2)}

name_to_input = {'ffn': lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim)),
                 'transformer': lambda batch_size, hid_dim:
                 (torch.empty((batch_size, 128, hid_dim)), torch.empty((batch_size, hid_dim), dtype=torch.bool)),
                 'nop': lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim)),
                 'det_dropout': lambda batch_size, hid_dim:
                 (torch.empty((batch_size, hid_dim)), torch.randint(0, 1, (batch_size, hid_dim)))}
