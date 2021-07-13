import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.moe import register_expert_class

sample_input = lambda batch_size, hidden_dim: torch.empty((batch_size, hidden_dim))


@register_expert_class("perceptron", sample_input)
class MultilayerPerceptron(nn.Module):
    def __init__(self, hidden_dim, num_classes=10):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.layer2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.layer3 = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


multihead_sample_input = lambda batch_size, hidden_dim: (
    torch.empty((batch_size, hidden_dim)),
    torch.empty((batch_size, 2 * hidden_dim)),
    torch.empty((batch_size, 3 * hidden_dim)),
)


@register_expert_class("multihead", multihead_sample_input)
class MultiheadNetwork(nn.Module):
    def __init__(self, hidden_dim, num_classes=10):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, num_classes)
        self.layer2 = nn.Linear(2 * hidden_dim, num_classes)
        self.layer3 = nn.Linear(3 * hidden_dim, num_classes)

    def forward(self, x1, x2, x3):
        x = self.layer1(x1) + self.layer2(x2) + self.layer3(x3)
        return x
