"""Train VoxelMLP or PointNet; select on val, evaluate test once, save weights and metrics."""

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shapes.model import MODELS
from shapes.pointcloud import GRID, N_POINTS, load_features, load_points


def set_seed(seed: int) -> None:
    # seed every generator a run touches, so the same seed reproduces the same init, dropout masks, and batch order
    random.seed(seed)
    np.random.seed(seed)
    # also seeds cuda on every device
    torch.manual_seed(seed)


def to_tensors(path: Path, device: torch.device, load: Callable[[Path], tuple[np.ndarray, np.ndarray]]) -> TensorDataset:
    X, y = load(path)
    # convert and move once, up front: train is ~56 MB as voxels or ~42 MB as points, so it fits on the gpu and nothing gets cast inside the loop
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
        # mean per-class recall: the "avg. class" accuracy the pointnet and voxnet papers report
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
    }
    # predictions come back separately, for the confusion matrix and for finding a misclassified object for the video
    return metrics, preds


def train_one_epoch(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[float, float]:
    # train mode turns dropout on (and, for pointnet, makes batch norm use batch statistics)
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
        # .item() syncs with the gpu once per batch; cheap at this size
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


# level 1: conventional, cited defaults (reasons in the decisions doc); any later change is judged on val and logged
LR = 1e-3
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 200
PATIENCE = 15
EVAL_BATCH_SIZE = 256
CONFIGS = {
    "mlp": {"load": load_features, "batch_size": 64, "model_kwargs": {"in_features": GRID**3, "hidden": 256, "dropout": 0.5}},
    # batch 32 and dropout 0.3 are the pointnet paper's values (appendix c)
    "pointnet": {"load": load_points, "batch_size": 32, "model_kwargs": {"dropout": 0.3}},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--points-dir", type=Path, default=Path("data/points"))
    parser.add_argument("--meta", type=Path, default=Path("data/meta.json"))
    parser.add_argument("--weights-dir", type=Path, default=Path("backend/model"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    cfg = CONFIGS[args.model]

    # the stored class list fixes what each output index means; never re-derived, so labels can't shift under a trained model
    classes = json.loads(args.meta.read_text())["classes"]
    model_kwargs = {**cfg["model_kwargs"], "n_classes": len(classes)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    # all three splits load up front, so a missing or malformed file fails now rather than after training
    train_ds, val_ds, test_ds = (to_tensors(args.points_dir / f"{s}.npz", device, cfg["load"]) for s in ("train", "val", "test"))
    # a seeded generator makes the shuffle order part of what the seed reproduces
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    # in eval mode batch norm uses running averages, so eval batch size changes speed, not results
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, num_workers=0)

    # fresh model every run, so nothing carries over from a previous one
    model = MODELS[args.model](**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params:,} parameters on {device}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history, best_epoch = fit(model, train_loader, val_loader, loss_fn, optimizer, MAX_EPOCHS, PATIENCE)

    # selection is finished; this is the only place test is ever evaluated
    # train is re-evaluated in eval mode for a clean number (the per-epoch train figures had dropout on)
    results = {}
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        results[name], preds = evaluate(model, DataLoader(ds, batch_size=EVAL_BATCH_SIZE, num_workers=0), loss_fn)
    # preds is test's, from the last pass; rows are true classes, columns predicted
    cm = confusion_matrix(test_ds.tensors[1].cpu().numpy(), preds, labels=list(range(len(classes))))

    config = {"seed": args.seed, "batch_size": cfg["batch_size"], "lr": LR, "weight_decay": WEIGHT_DECAY,
              "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "best_epoch": best_epoch}
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.weights_dir / f"{args.model}.pt"
    # everything the backend needs to rebuild the model and to refuse a checkpoint that doesn't match its preprocessing
    torch.save({
        "arch": args.model,
        "model_kwargs": model_kwargs,
        # cpu tensors, so the file loads the same way on a machine without a gpu
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "classes": classes,
        "n_points": N_POINTS,
        # only the voxel model depends on the grid
        "grid": GRID if args.model == "mlp" else None,
        "config": config,
    }, weights_path)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_dir / f"{args.model}.json"
    results_path.write_text(json.dumps({
        "model": args.model, "n_parameters": n_params, "model_kwargs": model_kwargs, "config": config,
        **results, "classes": classes, "confusion_matrix": cm.tolist(), "history": history,
    }, indent=2))

    print(f"best epoch {best_epoch}; test accuracy {results['test']['accuracy']:.3f}, macro-F1 {results['test']['macro_f1']:.3f}"
          f" (logistic regression: 0.870, 0.866)")
    print(f"saved {weights_path} and {results_path}")


if __name__ == "__main__":
    main()