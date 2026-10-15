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
    """Return sorted (class_name, source_split, relpath) for every .off file under data_root."""
    results = []
    for path in data_root.glob("*/*/*.off"): # each * matches exactly one path component, so this only finds class/split/file and can't wander into nested folders
        rel = path.relative_to(data_root) # strip the machine-specific root so relpath is the same on Legion and the Mac
        class_name, source_split, _ = rel.parts # unpacking also acts as an assertion that there are exactly 3 parts
        if source_split not in SOURCE_SPLITS: # split names are structure later code depends on (pool = train); class names are just data, so only splits get checked
            raise ValueError(f"Unexpected split folder {source_split!r}: {rel.as_posix()}")
        results.append((class_name, source_split, rel.as_posix())) # rel.as_posix gives forward slashes on every OS (so a manifest written on windows and one on linux would be byte identical)
    return sorted(results) # since tuples compare element by element, this sorts by class, then split, then relpath


def build_manifest(
    data_root: Path, val_fraction: float = 0.15, seed: int = 0
) -> tuple[list[ManifestRow], dict]:
    """Pure: no file writes. Returns (rows, meta)."""
    files = discover_files(data_root)
    if not files: # a wrong data root fails here with a clear message instead of deep inside sklearn
        raise ValueError(f"No .off files found under {data_root}")

    classes = sorted({class_name for class_name, _, _ in files})  # set removes repeats; sorted gives a stable order. stored in meta.json and never re-derived, so labels can't shift under a trained model
    label_of = {name: i for i, name in enumerate(classes)} # dict = constant-time name -> label lookup

    # split into pool and test so test files don't enter train test split and can't leak into train or val
    pool = [f for f in files if f[1] == "train"]
    test = [f for f in files if f[1] == "test"]

    train_part, val_part = train_test_split(
        pool,
        test_size=val_fraction,
        stratify=[label_of[class_name] for class_name, _, _ in pool], # built from pool itself so labels stay aligned with pool's order
        random_state=seed, # reproducible only because pool's order is fixed (discover_files sorts)
    )

    def make_row(file: tuple[str, str, str], split: str) -> ManifestRow:  # nested so it can use label_of without passing it in; tradeoff: can't be imported or tested on its own
        class_name, source_split, relpath = file
        return ManifestRow(
            id=Path(relpath).stem,
            class_name=class_name,
            label=label_of[class_name],
            split=split,
            source_split=source_split,
            relpath=relpath,
        )

    rows = (
        [make_row(f, "train") for f in train_part]
        + [make_row(f, "val") for f in val_part]
        + [make_row(f, "test") for f in test]
    )

    counts = Counter(r.id for r in rows) # ids are how the backend looks objects up, so they must be unique
    duplicates = sorted(i for i, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(f"Duplicate ids ({len(duplicates)}), e.g. {duplicates[:5]}")

    rows.sort(key=lambda r: (r.class_name, r.id)) # final order doesn't depend on sklearn's shuffle, so the CSV is identical run to run
    meta = {"classes": classes, "val_fraction": val_fraction, "seed": seed} # no absolute paths: meta.json is committed and the data root differs per machine
    return rows, meta


def write_manifest(rows: list[ManifestRow], meta: dict, out_dir: Path) -> None:
    """Write out_dir/manifest.csv and out_dir/meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True) # creates missing parents, no error if it exists, so re-running is safe

    with open(out_dir / MANIFEST_NAME, "w", newline="", encoding="utf-8") as f: # csv module writes its own line endings (newline="" stops them being doubled); default encoding varies by platform
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(ManifestRow)], lineterminator="\n") # header comes from the dataclass, so adding a field updates the CSV automatically. \n instead of csv's default \r\n so unix tools like grep behave
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)

    with open(out_dir / META_NAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n") # json.dump doesn't end with a newline; git flags files without one


def read_manifest(out_dir: Path) -> tuple[list[ManifestRow], list[str]]:
    """Return (rows, classes). Training and the backend call this; they never parse the CSV themselves."""
    with open(out_dir / META_NAME, encoding="utf-8") as f:
        classes = json.load(f)["classes"] # read the stored list, never re-derive it from the rows

    with open(out_dir / MANIFEST_NAME, newline="", encoding="utf-8") as f:
        rows = [ManifestRow(**{**d, "label": int(d["label"])}) for d in csv.DictReader(f)] # csv gives strings and dataclasses don't check types, so convert label explicitly. ManifestRow(**d) raises TypeError on extra/missing columns = free schema check

    return rows, classes


def main() -> None:
    """CLI: python -m shapes.manifest --out-dir data"""
    parser = argparse.ArgumentParser(description=__doc__) # inside main, __doc__ is the module's docstring, not the functions, which is what --help should show
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("MODELNET_ROOT"),
        help="ModelNet10 folder containing <class>/{train,test}/ (default: $MODELNET_ROOT)",
    )
    parser.add_argument("--out-dir", type=Path, required=True) # required=True: no default, so can't accidentally write manifest into random current directory
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.data_root is None:
        parser.error("set --data-root or the MODELNET_ROOT environment variable") # prints usage and exits with status 2 (bad arguments) instead of a traceback

    rows, meta = build_manifest(args.data_root.expanduser(), args.val_fraction, args.seed)
    write_manifest(rows, meta, args.out_dir)

    counts = Counter((r.class_name, r.split) for r in rows) # Counter returns 0 for missing keys instead of raising KeyError
    print(f"Wrote {len(rows)} rows to {args.out_dir / MANIFEST_NAME}")
    print(f"{'class':<12} {'train':>6} {'val':>6} {'test':>6}") # left align in 12 characters, right align numbers
    for name in meta["classes"]: # label order, same order the confusion matrix will use
        print(f"{name:<12} {counts[name, 'train']:>6} {counts[name, 'val']:>6} {counts[name, 'test']:>6}")


if __name__ == "__main__":
    main()