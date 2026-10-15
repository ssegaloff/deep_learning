"""Plot training curves and test confusion matrices from results/<model>.json; no retraining needed."""

import argparse
import json
from pathlib import Path

import matplotlib

# legion is reached over ssh with no display, so render straight to files
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# two series colors checked for colorblind separation and contrast on a light background; text stays in ink, not series colors
TRAIN, VAL = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
NAMES = {"mlp": "VoxelMLP", "pointnet": "PointNet"}


def plot_curves(runs: dict, out: Path) -> None:
    # rows share a y axis, so the two models' losses and accuracies are read on the same scale
    fig, axes = plt.subplots(2, len(runs), figsize=(5 * len(runs), 7), sharey="row", squeeze=False)
    for col, (model, r) in enumerate(runs.items()):
        epochs = [h["epoch"] for h in r["history"]]
        best = r["config"]["best_epoch"]
        for row, (key, ylabel) in enumerate((("loss", "Cross-entropy loss"), ("accuracy", "Accuracy"))):
            ax = axes[row][col]
            ax.plot(epochs, [h[f"train_{key}"] for h in r["history"]], color=TRAIN, lw=2, label="train")
            ax.plot(epochs, [h[f"val_{key}"] for h in r["history"]], color=VAL, lw=2, label="val")
            # the restored weights come from this epoch; everything after it is the patience window
            ax.axvline(best, color=MUTED, ls="--", lw=1)
            ax.annotate(f"best epoch {best}", xy=(best, 1), xycoords=("data", "axes fraction"),
                        xytext=(4, -12), textcoords="offset points", color=MUTED, fontsize=9)
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(ylabel, color=INK)
            if row == 1:
                ax.set_xlabel("Epoch", color=INK)
        test = r["test"]
        axes[0][col].set_title(f"{NAMES.get(model, model)}: test accuracy {test['accuracy']:.3f}", color=INK)
    axes[0][0].legend(frameon=False)
    # the per-epoch train numbers are running averages with dropout on, so they read slightly worse than an eval-mode pass
    fig.text(0.5, 0.005, "Train curves are per-epoch running averages (dropout on); val is evaluated after each epoch.",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_confusion(model: str, r: dict, out: Path) -> None:
    cm = np.array(r["confusion_matrix"])
    classes = [c.replace("_", " ") for c in r["classes"]]
    # row-normalize: each row shows where one true class's objects went, so bathtub (50 test objects) reads on the same scale as the 100s
    rates = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    # one hue, light to dark: color encodes a magnitude (the fraction), not an identity
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    # raw counts in the nonzero cells only, so the off-diagonal mistakes stand out against empty cells
    for i, j in zip(*np.nonzero(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                color="white" if rates[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted class", color=INK)
    ax.set_ylabel("True class", color=INK)
    test = r["test"]
    ax.set_title(f"{NAMES.get(model, model)} on test: accuracy {test['accuracy']:.3f}, "
                 f"balanced {test['balanced_accuracy']:.3f}", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of the true class (cells show counts)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["mlp", "pointnet"])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    runs = {m: json.loads((args.results_dir / f"{m}.json").read_text()) for m in args.models}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(runs, args.out_dir / "curves.png")
    for model, r in runs.items():
        plot_confusion(model, r, args.out_dir / f"confusion_{model}.png")
    print(f"wrote {len(runs) + 1} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
    