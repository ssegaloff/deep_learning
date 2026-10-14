"""Build and read the dataset manifest: which objects exist, their class, and their split.

The writer and reader live together so the file format is defined in one place.
"""

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

from sklearn.model_selection import train_test_split

MANIFEST_NAME = "manifest.csv"
META_NAME = "meta.json"
SOURCE_SPLITS = ("train", "test")  # ModelNet's folder names; anything else is an error


@dataclass(frozen=True)
class ManifestRow:
    id: str
    class_name: str
    label: int
    split: Literal["train", "val", "test"]  # ours
    source_split: Literal["train", "test"]  # ModelNet's folder
    relpath: str  # relative to the data root, forward slashes, e.g. "chair/train/chair_0890.off"


def discover_files(data_root: Path) -> list[tuple[str, str, str]]:
    """Return sorted (class_name, source_split, relpath) for every .off file under data_root.

    TODO:
    - data_root.glob("*/*/*.off"); class and split from p.relative_to(data_root).parts
    - raise ValueError if a split folder isn't in SOURCE_SPLITS (don't silently include it)
    - relpath = p.relative_to(data_root).as_posix()
    - return sorted(...)  (tuple order = class, split, relpath)
    """
    ...


def build_manifest(
    data_root: Path, val_fraction: float = 0.15, seed: int = 0
) -> tuple[list[ManifestRow], dict]:
    """Pure: no file writes. Returns (rows, meta).

    TODO:
    - files = discover_files(data_root); raise ValueError naming data_root if empty
    - classes = sorted class names; label_of = {name: i for i, name in enumerate(classes)}
    - pool = files with source_split == "train"   (ModelNet test never enters the split)
    - train_part, val_part = train_test_split(
          pool, test_size=val_fraction, stratify=<labels aligned with pool>, random_state=seed)
    - rows: train_part -> "train", val_part -> "val", source test files -> "test";
      id = Path(relpath).stem
    - ids unique: Counter over ids; raise ValueError listing the duplicates
    - sort rows by (class_name, id)
    - meta = {"classes": classes, "val_fraction": val_fraction, "seed": seed}
      (no absolute paths: meta.json is committed)
    """
    ...


def write_manifest(rows: list[ManifestRow], meta: dict, out_dir: Path) -> None:
    """Write out_dir/manifest.csv and out_dir/meta.json.

    TODO:
    - out_dir.mkdir(parents=True, exist_ok=True)
    - CSV: open(..., "w", newline="", encoding="utf-8");
      DictWriter(fieldnames=[f.name for f in fields(ManifestRow)]); writeheader; writerows(asdict(r) ...)
    - JSON: open(..., "w", encoding="utf-8"); json.dump(meta, f, indent=2); then write "\\n"
    """
    ...


def read_manifest(out_dir: Path) -> tuple[list[ManifestRow], list[str]]:
    """Return (rows, classes). Training and the backend call this; they never parse the CSV themselves.

    TODO:
    - DictReader with newline="", encoding="utf-8"
    - label = int(d["label"])  (csv returns strings; the dataclass won't check types for you)
    - classes come from meta.json, never re-derived
    """
    ...


def main() -> None:
    """CLI: python -m shapeprep.manifest --out-dir data"""
    parser = argparse.ArgumentParser(description=__doc__)
    # TODO: --data-root  type=Path, default=os.environ.get("MODELNET_ROOT")
    #       if still None: parser.error("set --data-root or MODELNET_ROOT")
    #       then .expanduser()
    # TODO: --out-dir (type=Path, required), --val-fraction (float, 0.15), --seed (int, 0)
    # TODO: build_manifest -> write_manifest
    # TODO: Counter((r.class_name, r.split) for r in rows); print one line per class: train / val / test
    ...


if __name__ == "__main__":
    main()