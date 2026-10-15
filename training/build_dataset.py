"""Sample every manifest object once and write data/points/{train,val,test}.npz."""

import argparse
import os
from pathlib import Path

import numpy as np

from shapes.manifest import read_manifest
from shapes.pointcloud import load_mesh, normalize, sample_points

SPLITS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("MODELNET_ROOT"),
        help="ModelNet10 folder containing <class>/{train,test}/ (default: $MODELNET_ROOT)",
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/points"))
    # store more than the model uses, so an optional resampling experiment doesn't need to reload meshes
    parser.add_argument("--n-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.data_root is None:
        parser.error("set --data-root or the MODELNET_ROOT environment variable")
    root = args.data_root.expanduser()

    rows, _ = read_manifest(args.manifest_dir)
    # one generator consumed in manifest order, so a single seed reproduces every file
    rng = np.random.default_rng(args.seed)

    # preallocate instead of appending: avoids building a list of 4,899 arrays and then copying it
    points = np.empty((len(rows), args.n_points, 3), dtype=np.float32)
    for i, row in enumerate(rows):
        points[i] = sample_points(normalize(load_mesh(root / row.relpath)), args.n_points, rng)
        # the run takes about a minute; this shows it's alive without flooding the terminal
        if (i + 1) % 500 == 0:
            print(f"sampled {i + 1}/{len(rows)}")

    # int64 because CrossEntropyLoss expects long targets; converting here keeps casts out of the training loop
    labels = np.array([r.label for r in rows], dtype=np.int64)
    # fixed-width unicode, not Python objects, so np.load works without allow_pickle (loading pickles can run code)
    ids = np.array([r.id for r in rows])
    splits = np.array([r.split for r in rows])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        mask = splits == split
        # ids travel with the points so any prediction can be traced back to its manifest row
        # n_points and seed are stored so the file records how it was made
        # plain savez: float coordinates compress poorly, so compression would mostly cost time
        np.savez(
            args.out_dir / f"{split}.npz",
            points=points[mask],
            labels=labels[mask],
            ids=ids[mask],
            n_points=args.n_points,
            seed=args.seed,
        )
        print(f"{split}: {mask.sum()} objects -> {args.out_dir / f'{split}.npz'}")


if __name__ == "__main__":
    main()