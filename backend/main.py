"""hw2 model server: loads both trained classifiers once at startup and serves objects, meshes, and predictions."""

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI

from shapes.manifest import ManifestRow, read_manifest
from shapes.model import MODELS
from shapes.pointcloud import GRID, N_POINTS, load_points

# resolved from this file, not the working directory, so the server finds its files wherever uvicorn is started
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "backend" / "model"
SPLITS = ("train", "val", "test")
# inference is one object per click, so cpu is fast enough, and the gpu stays free for training runs
DEVICE = torch.device("cpu")


def load_model(path: Path, classes: list[str]) -> torch.nn.Module:
    # weights_only defaults to True: torch.load refuses arbitrary pickled objects, so loading a file can't run code
    ckpt = torch.load(path, map_location=DEVICE)
    # load_state_dict catches wrong shapes; these catch mismatches with the right shapes and the wrong meaning
    if ckpt["arch"] != path.stem:
        raise ValueError(f"{path.name} holds a {ckpt['arch']!r} model")
    if ckpt["classes"] != classes:
        raise ValueError(f"{path.name} was trained with classes {ckpt['classes']}; meta.json has {classes}")
    if ckpt["n_points"] != N_POINTS:
        raise ValueError(f"{path.name} was trained on {ckpt['n_points']} points; N_POINTS is {N_POINTS}")
    if ckpt["arch"] == "mlp" and ckpt["grid"] != GRID:
        raise ValueError(f"{path.name} was trained on a {ckpt['grid']}^3 grid; GRID is {GRID}")
    model = MODELS[ckpt["arch"]](**ckpt["model_kwargs"])
    # strict by default: a missing or unexpected key is an error, not a silently random layer
    model.load_state_dict(ckpt["state_dict"])
    # dropout off, and batch norm uses its running averages instead of the current batch's statistics
    return model.to(DEVICE).eval()


def load_all_points(rows: list[ManifestRow]) -> dict[str, np.ndarray]:
    label_of = {r.id: r.label for r in rows}
    points = {}
    for split in SPLITS:
        path = DATA_DIR / "points" / f"{split}.npz"
        # load_points decides which stored points a model sees, so the backend gets the same 1,024 that training used
        pts, labels = load_points(path)
        # ids sit in the same file in the same order; load_points doesn't return them, so read them alongside
        with np.load(path) as d:
            ids = d["ids"]
        for obj_id, p, label in zip(ids, pts, labels, strict=True):
            obj_id = str(obj_id)
            # a mismatch means the .npz and the manifest drifted apart, and every prediction would be shown against the wrong label
            if label_of[obj_id] != label:
                raise ValueError(f"{obj_id}: label {label} in {path.name}, {label_of[obj_id]} in the manifest")
            points[obj_id] = p
    if points.keys() != label_of.keys():
        raise ValueError("the .npz files and the manifest list different objects")
    return points


@asynccontextmanager
async def lifespan(app: FastAPI):
    rows, classes = read_manifest(DATA_DIR)
    app.state.classes = classes
    app.state.rows = {r.id: r for r in rows}
    app.state.points = load_all_points(rows)
    # every registered architecture, so the app can show them side by side; a missing .pt fails here, not on the first click
    app.state.models = {arch: load_model(MODEL_DIR / f"{arch}.pt", classes) for arch in MODELS}
    for arch, model in app.state.models.items():
        print(f"loaded {arch}: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"loaded {len(app.state.points):,} objects")
    yield


app = FastAPI(lifespan=lifespan)
