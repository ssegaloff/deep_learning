"""The classifier, shared so training saves it and the backend rebuilds it from the same definition."""

import torch
from torch import nn


# names, not classes, so checkpoints store a plain string: torch.load's weights_only mode refuses pickled classes
# sigmoid is the classic contrast: its gradient is at most 0.25 and near 0 when saturated, so deeper stacks learn slowly
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "sigmoid": nn.Sigmoid}


class VoxelMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden: int | list[int],
        n_classes: int,
        dropout: float = 0.5,
        activation: str = "relu",
    ):
        super().__init__()
        # an int is the level-1 form (one hidden layer); its checkpoint stores hidden=256, so both forms must load
        sizes = [hidden] if isinstance(hidden, int) else list(hidden)
        # hw2 requires at least one hidden layer; with none this would be logistic regression
        if not sizes or any(h < 1 for h in sizes):
            raise ValueError(f"need at least one hidden layer of positive size, got {hidden!r}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}")
        layers, width = [], in_features
        for h in sizes:
            # without a nonlinearity, stacked linear layers collapse into one, and this would be logistic regression again
            # dropout after every hidden layer: about 1M weights vs. 3,392 training objects; active only in train mode
            layers += [nn.Linear(width, h), ACTIVATIONS[activation](), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, n_classes))
        # one hidden layer gives linear, activation, dropout, linear: indices 0 and 3, the same keys as the level-1 checkpoint
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits, not probabilities: CrossEntropyLoss applies log-softmax itself, so a softmax here would be applied twice
        # the backend applies softmax when it needs probabilities to display
        return self.net(x)


class PointNet(nn.Module):
    """vanilla pointnet classifier (qi et al. 2017, fig. 2 without the t-nets)."""

    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()

        def shared(c_in: int, c_out: int) -> list[nn.Module]:
            # kernel_size=1 transforms each point on its own with the same weights: a shared mlp, not a spatial convolution
            # biases are redundant before batch norm, but kept so the parameter count matches the paper's
            return [nn.Conv1d(c_in, c_out, kernel_size=1), nn.BatchNorm1d(c_out), nn.ReLU()]

        # two blocks, as in fig. 2: the full model puts the feature t-net between them,
        # and the segmentation network reuses the 64-dim per-point features that come out of the first
        self.point_mlp1 = nn.Sequential(*shared(3, 64), *shared(64, 64))
        self.point_mlp2 = nn.Sequential(*shared(64, 64), *shared(64, 128), *shared(128, 1024))
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            # the paper's keep probability 0.7, on the 256-unit layer only
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # loader gives (batch, points, xyz); conv1d and batchnorm1d want channels in dim 1
        x = x.transpose(1, 2)
        # (batch, 64, points): local per-point features
        x = self.point_mlp1(x)
        # (batch, 1024, points)
        x = self.point_mlp2(x)
        # max over points is a symmetric function, so the global feature can't depend on point order
        x = x.amax(dim=2)
        # logits, as in VoxelMLP: CrossEntropyLoss applies log-softmax itself
        return self.head(x)
    

# one short name per architecture, shared by train.py and the backend, so a checkpoint's "arch" can only rebuild the class that made it
MODELS = {"mlp": VoxelMLP, "pointnet": PointNet}