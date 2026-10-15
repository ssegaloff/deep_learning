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


def train_one_epoch(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[float, float]:
    # train mode turns dropout on (and, for pointnet later, makes batch norm use batch statistics)
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        # gradients accumulate across backward() calls by default, so clear last batch's before computing this one's
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        # backprop: fills .grad on every parameter with d(loss)/d(parameter)
        loss.backward()
        # the gradient-descent update; adamw scales each step per weight and applies weight decay directly to the weights
        optimizer.step()
        # .item() syncs with the gpu once per batch; cheap at ~53 batches per epoch
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(y)
    # running averages over an epoch where the weights kept changing and dropout was on, so they read a bit worse than an eval-mode pass would
    return total_loss / n, correct / n


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 200,
    patience: int = 15,
) -> tuple[list[dict], int]:
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val, _ = evaluate(model, val_loader, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"epoch {epoch:3d}  train {train_loss:.4f} / {train_acc:.3f}  val {val['loss']:.4f} / {val['accuracy']:.3f}")
        # select on val loss: it moves smoothly, while accuracy and f1 jump in steps of one object
        if val["loss"] < best_loss:
            best_loss, best_epoch = val["loss"], epoch
            # state_dict() returns references to the live tensors, so without clone() this "snapshot" would keep training
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    # hand back the selected model, not whatever the last epoch happened to leave
    model.load_state_dict(best_state)
    return history, best_epoch