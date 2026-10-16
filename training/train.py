"""Train VoxelMLP or PointNet; select on val, evaluate test once, save weights and metrics."""

import argparse
import json
import math
import random
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shapes.defaults import MODEL_DEFAULTS, TRAINING
from shapes.model import ACTIVATIONS, MODELS
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
    on_epoch: Callable[[dict], None] | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[list[dict], int]:
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    for epoch in range(1, max_epochs + 1):
        start = time.perf_counter()
        # read before training, so the record holds the rate this epoch's updates actually used
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val, _ = evaluate(model, val_loader, loss_fn)
        # once per epoch, after that epoch's optimizer steps (pytorch warns if the scheduler steps before the optimizer)
        if scheduler is not None:
            scheduler.step()
        record = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_accuracy": train_acc,
                  **{f"val_{k}": v for k, v in val.items()}, "seconds": time.perf_counter() - start}
        history.append(record)
        print(f"epoch {epoch:3d}  lr {lr:.2e}  train {train_loss:.4f} / {train_acc:.3f}  val {val['loss']:.4f} / {val['accuracy']:.3f}")
        # a callback, not a file path: fit stays unaware of where progress goes (the cli prints, a run also writes a file)
        if on_epoch is not None:
            on_epoch(record)
        # select on val loss: it moves smoothly, while accuracy and f1 jump in steps of one object
        # a nan loss compares false with everything, so a diverged epoch can never become "best"
        if val["loss"] < best_loss:
            best_loss, best_epoch = val["loss"], epoch
            # state_dict() returns references to the live tensors, so without clone() this "snapshot" would keep training
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    # with a user-chosen learning rate the loss can diverge from epoch 1, leaving nothing to restore; fail with the reason
    if best_state is None:
        raise RuntimeError("validation loss was never finite (the learning rate may be too high); no weights to restore")
    # hand back the selected model, not whatever the last epoch happened to leave
    model.load_state_dict(best_state)
    return history, best_epoch


# how each model reads a split; the numbers live in shapes.defaults, shared with the backend
LOADERS = {"mlp": load_features, "pointnet": load_points}
# in eval mode batch norm uses running averages, so eval batch size changes speed, not results
EVAL_BATCH_SIZE = 256
# the pointnet paper's factor (appendix c: "divided by 2 every 20 epochs"); only the step is a flag
LR_FACTOR = 0.5


# argparse type checks: a bad value fails at the command line, before any data loads
def positive_int(s: str) -> int:
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {v}")
    return v


def positive_float(s: str) -> float:
    v = float(s)
    # isfinite rejects nan and inf, which "v > 0" alone would let through (inf) or reject only by accident (nan)
    if not (math.isfinite(v) and v > 0):
        raise argparse.ArgumentTypeError(f"must be a finite number above 0, got {s}")
    return v


def nonneg_float(s: str) -> float:
    v = float(s)
    if not (math.isfinite(v) and v >= 0):
        raise argparse.ArgumentTypeError(f"must be a finite number at least 0, got {s}")
    return v


def probability(s: str) -> float:
    v = float(s)
    # dropout 1 would zero every activation
    if not 0 <= v < 1:
        raise argparse.ArgumentTypeError(f"must be in [0, 1), got {s}")
    return v


def seed_value(s: str) -> int:
    v = int(s)
    # np.random.seed accepts only 0 ..< 2**32, so a seed outside that fails here instead of inside set_seed
    if not 0 <= v < 2**32:
        raise argparse.ArgumentTypeError(f"must be in [0, 2**32), got {v}")
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--seed", type=seed_value, default=TRAINING["seed"])
    parser.add_argument("--lr", type=positive_float, default=TRAINING["lr"])
    parser.add_argument("--weight-decay", type=nonneg_float, default=TRAINING["weight_decay"])
    parser.add_argument("--max-epochs", type=positive_int, default=TRAINING["max_epochs"])
    parser.add_argument("--patience", type=positive_int, default=TRAINING["patience"])
    # absent means a constant rate (level 1); 0 is an error rather than a hidden "off"
    parser.add_argument("--lr-step", type=positive_int, default=TRAINING["lr_step"],
                        help=f"multiply the learning rate by {LR_FACTOR} every N epochs")
    # model-specific: left as None, they become that model's level-1 value
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--dropout", type=probability)
    parser.add_argument("--hidden", type=positive_int, nargs="+", help="mlp only: hidden layer sizes, e.g. --hidden 512 128")
    parser.add_argument("--activation", choices=sorted(ACTIVATIONS), help="mlp only")
    parser.add_argument("--points-dir", type=Path, default=Path("data/points"))
    parser.add_argument("--meta", type=Path, default=Path("data/meta.json"))
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    # a run directory: weights, results, and per-epoch progress all go here, so a run can't overwrite the level-1 files
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    # loud, not ignored: a --hidden passed to pointnet would otherwise look like it had been applied
    if args.model != "mlp" and (args.hidden or args.activation):
        parser.error("--hidden and --activation apply to the mlp only")
    if args.out_dir and (args.weights_dir or args.results_dir):
        parser.error("--out-dir replaces --weights-dir and --results-dir; give one or the other")
    weights_dir = args.out_dir or args.weights_dir or Path("backend/model")
    results_dir = args.out_dir or args.results_dir or Path("results")
    progress_path = args.out_dir / "progress.jsonl" if args.out_dir else None
    if progress_path and progress_path.exists():
        parser.error(f"{progress_path} already exists; each run gets a fresh directory")

    # created before training, so an unwritable folder fails in a second instead of after the run
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    defaults = MODEL_DEFAULTS[args.model]
    # "is None", not "or": dropout 0.0 is a real choice, and "or" would replace it with the default
    pick = lambda value, key: defaults[key] if value is None else value
    batch_size = pick(args.batch_size, "batch_size")

    # the stored class list fixes what each output index means; never re-derived, so labels can't shift under a trained model
    classes = json.loads(args.meta.read_text())["classes"]
    model_kwargs = {"dropout": pick(args.dropout, "dropout"), "n_classes": len(classes)}
    if args.model == "mlp":
        model_kwargs |= {"in_features": GRID**3, "hidden": pick(args.hidden, "hidden"),
                         "activation": pick(args.activation, "activation")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    # all three splits load up front, so a missing or malformed file fails now rather than after training
    load = LOADERS[args.model]
    train_ds, val_ds, test_ds = (to_tensors(args.points_dir / f"{s}.npz", device, load) for s in ("train", "val", "test"))
    # a seeded generator makes the shuffle order part of what the seed reproduces
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, num_workers=0)

    # fresh model every run, so nothing carries over from a previous one
    model = MODELS[args.model](**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params:,} parameters on {device}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # no scheduler at all when off, so a level-1 run takes exactly the code path it took before
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=LR_FACTOR)
                 if args.lr_step is not None else None)

    log_epoch = None
    if progress_path:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("")

        def log_epoch(record: dict) -> None:
            # one whole line per write, closed (so flushed) each time: the backend reads this file while
            # training runs, and treats a last line without its newline as not yet written
            with progress_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    history, best_epoch = fit(model, train_loader, val_loader, loss_fn, optimizer,
                              args.max_epochs, args.patience, on_epoch=log_epoch, scheduler=scheduler)

    # selection is finished; this is the only place test is ever evaluated
    # train is re-evaluated in eval mode for a clean number (the per-epoch train figures had dropout on)
    results = {}
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        results[name], preds = evaluate(model, DataLoader(ds, batch_size=EVAL_BATCH_SIZE, num_workers=0), loss_fn)
    # preds is test's, from the last pass; rows are true classes, columns predicted
    cm = confusion_matrix(test_ds.tensors[1].cpu().numpy(), preds, labels=list(range(len(classes))))

    config = {"seed": args.seed, "batch_size": batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
                "max_epochs": args.max_epochs, "patience": args.patience, "lr_step": args.lr_step,
                "best_epoch": best_epoch}
    weights_path = weights_dir / f"{args.model}.pt"
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

    results_path = results_dir / f"{args.model}.json"
    results_path.write_text(json.dumps({
        "model": args.model, "n_parameters": n_params, "model_kwargs": model_kwargs, "config": config,
        **results, "classes": classes, "confusion_matrix": cm.tolist(), "history": history,
    }, indent=2))

    print(f"best epoch {best_epoch}; test accuracy {results['test']['accuracy']:.3f}, macro-F1 {results['test']['macro_f1']:.3f}"
          f" (logistic regression: 0.870, 0.866)")
    print(f"saved {weights_path} and {results_path}")


if __name__ == "__main__":
    main()
    