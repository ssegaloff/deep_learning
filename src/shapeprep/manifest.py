"""Build and read the dataset manifest: which objects exist, their class, and their split.

The writer and reader live together so the file format is defined in one place.
"""

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

MANIFEST_NAME = "manifest.csv"
META_NAME = "meta.json"


@dataclass(frozen=True)
class ManifestRow:
    id: str
    class_name: str
    label: int
    split: str          # "train" | "val" | "test"  (our split)
    source_split: str   # "train" | "test"         (ModelNet's folder)
    relpath: str        # relative to the data root, e.g. "chair/train/chair_0890.off"


def discover_files(data_root: Path) -> list[tuple[str, str, str]]:
    """Return (class_name, source_split, relpath) for every .off file, SORTED.

    TODO:
    - walk data_root/<class>/{train,test}/*.off
    - relpath relative to data_root, as a string with forward slashes
    - sort the result so order never depends on the filesystem
    """
    ...


def build_manifest(
    data_root: Path, val_fraction: float = 0.15, seed: int = 0
) -> tuple[list[ManifestRow], dict]:
    """Pure: no file writes. Returns (rows, meta).

    TODO:
    - classes = sorted class names; label = index into that list
    - for each class: take its source_split == "train" files,
      shuffle with a LOCAL random.Random(seed), move val_fraction of them to split "val"
    - source_split == "test" files get split "test", untouched
    - check: ids are unique (raise ValueError listing duplicates)
    - meta = {"classes": [...], "val_fraction": ..., "seed": ...}
    """
    ...


def write_manifest(rows: list[ManifestRow], meta: dict, out_dir: Path) -> None:
    """Write out_dir/manifest.csv and out_dir/meta.json.

    TODO:
    - create out_dir if needed
    - CSV header = the dataclass field names, one row per ManifestRow
    """
    ...


def read_manifest(out_dir: Path) -> tuple[list[ManifestRow], list[str]]:
    """Return (rows, classes). The backend and training call this; they never parse the CSV themselves.

    TODO:
    - csv gives back strings: convert label back to int
    - classes come from meta.json, not re-derived
    """
    ...


def main() -> None:
    """CLI: python -m shapeprep.manifest --out-dir data"""
    parser = argparse.ArgumentParser(description=__doc__)
    # TODO: --data-root (default: MODELNET_ROOT env var; error clearly if neither is given)
    # TODO: --out-dir, --val-fraction, --seed
    # TODO: build, write, then print per-class counts for each split
    ...


if __name__ == "__main__":
    main()