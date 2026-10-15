"""Train VoxelMLP on voxel occupancy features; select on val, evaluate test once, save weights and metrics."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shapes.model import VoxelMLP
from shapes.pointcloud import GRID, N_POINTS, load_features


def set_seed(seed: int) -> None:
    # seed every generator a run touches, so the same seed reproduces the same init, dropout masks, and batch order
    random.seed(seed)
    np.random.seed(seed)
    # also seeds cuda on every device
    torch.manual_seed(seed)


def to_tensors(path: Path, device: torch.device) -> TensorDataset:
    X, y = load_features(path)
    # convert and move once, up front: train is ~56 MB of float32, so it all fits on the gpu and nothing gets cast inside the loop
    # labels are already int64 from build_dataset; .long() keeps cross-entropy's dtype requirement visible here
    return TensorDataset(torch.from_numpy(X).to(device), torch.from_numpy(y).long().to(device))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> tuple[dict, np.ndarray]:
    # eval turns dropout off; no_grad skips building the autograd graph, since nothing here is backpropagated
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for X, y in loader:
        logits = model(X)
        # loss_fn returns the batch mean; weighting by batch size keeps a short last batch from counting as much as a full one
        total_loss += loss_fn(logits, y).item() * len(y)
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    metrics = {
        "loss": total_loss / len(targets),
        "accuracy": float((preds == targets).mean()),
        # same metric and convention as baselines.py, so the numbers are directly comparable
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }
    # predictions come back separately, for the confusion matrix and for finding a misclassified object for the video
    return metrics, preds

