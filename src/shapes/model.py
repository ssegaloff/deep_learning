"""The classifier, shared so training saves it and the backend rebuilds it from the same definition."""

import torch
from torch import nn


class VoxelMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            # the one hidden layer: the only structural difference from logistic regression
            nn.Linear(in_features, hidden),
            # without a nonlinearity, two linear layers collapse into one, and this would just be logistic regression again
            nn.ReLU(),
            # about 1M weights vs. 3,392 training objects, so it needs regularization; dropout is active only in train mode
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits, not probabilities: CrossEntropyLoss applies log-softmax itself, so a softmax here would be applied twice
        # the backend applies softmax when it needs probabilities to display
        return self.net(x)