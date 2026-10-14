"""Majority-class and logistic-regression baselines on voxel occupancy features."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from shapeprep.pointcloud import voxelize

N_POINTS = 1024
GRID = 16


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # NpzFile holds the file open; the with block closes it as soon as the arrays are read
    with np.load(path) as d:
        # the first N_POINTS form a fixed sample; stored points are independent draws, so any fixed prefix is a valid uniform sample
        X = np.stack([voxelize(p[:N_POINTS], GRID) for p in d["points"]])
        y = d["labels"]
    return X, y


def scores(model, X: np.ndarray, y: np.ndarray) -> dict:
    pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        # macro averages per-class F1 equally, so ignoring the small classes is punished; that matters with chair 8x bathtub
        # zero_division=0: a class that's never predicted has undefined precision, and counting it as 0 is the usual convention
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-dir", type=Path, default=Path("data/points"))
    parser.add_argument("--out", type=Path, default=Path("results/baselines.json"))
    args = parser.parse_args()

    X_tr, y_tr = load_features(args.points_dir / "train.npz")
    X_va, y_va = load_features(args.points_dir / "val.npz")
    X_te, y_te = load_features(args.points_dir / "test.npz")
    print(f"features: train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

    majority = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)

    # choose the regularization strength on val only; test is used once, at the end
    # smaller C = stronger L2 penalty, needed because there are more features (4,096) than training objects (3,392)
    best = None
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        # max_iter raised because the default 100 often stops before converging on this many features
        lr = LogisticRegression(C=C, max_iter=2000).fit(X_tr, y_tr)
        val = scores(lr, X_va, y_va)
        print(f"C={C}: val accuracy {val['accuracy']:.3f}, macro-F1 {val['macro_f1']:.3f}")
        # select on macro-F1, not accuracy, for the same imbalance reason
        if best is None or val["macro_f1"] > best[1]["macro_f1"]:
            best = (C, val, lr)
    best_C, best_val, lr = best

    results = {
        # recorded so the numbers can be traced to the exact features they came from
        "features": {"type": "binary voxel occupancy", "grid": GRID, "n_points": N_POINTS},
        "majority": {"val": scores(majority, X_va, y_va), "test": scores(majority, X_te, y_te)},
        "logistic_regression": {"C": best_C, "val": best_val, "test": scores(lr, X_te, y_te)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()