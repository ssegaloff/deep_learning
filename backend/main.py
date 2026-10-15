"""hw2 model server: loads both trained classifiers once at startup and serves objects, meshes, and predictions."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from shapes.manifest import ManifestRow, read_manifest
from shapes.model import MODELS
from shapes.pointcloud import GRID, N_POINTS, load_mesh, load_points, normalize, voxelize

# resolved from this file, not the working directory, so the server finds its files wherever uvicorn is started
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "backend" / "model"
SPLITS = ("train", "val", "test")
# inference is one object per click, so cpu is fast enough, and the gpu stays free for training runs
DEVICE = torch.device("cpu")
# how each architecture turns stored points into its input, mirroring load_features and load_points in training
PREPARE = {
    "mlp": lambda points: voxelize(points, GRID),
    "pointnet": lambda points: points,
}


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
    # a model registered in model.py without an input recipe here would otherwise fail only on the first click
    if PREPARE.keys() != MODELS.keys():
        raise RuntimeError(f"PREPARE covers {sorted(PREPARE)}; MODELS has {sorted(MODELS)}")
    root = os.environ.get("MODELNET_ROOT")
    # meshes are only for the viewer, but a missing root should fail at startup, not on the first click
    if root is None or not Path(root).expanduser().is_dir():
        raise RuntimeError("set MODELNET_ROOT to the ModelNet10 folder containing <class>/{train,test}/")
    app.state.mesh_root = Path(root).expanduser()
    rows, classes = read_manifest(DATA_DIR)
    app.state.classes = classes
    app.state.rows = {r.id: r for r in rows}
    app.state.points = load_all_points(rows)
    # every registered architecture, so the app can show them side by side; a missing .pt fails here
    app.state.models = {arch: load_model(MODEL_DIR / f"{arch}.pt", classes) for arch in MODELS}
    for arch, model in app.state.models.items():
        print(f"loaded {arch}: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"loaded {len(app.state.points):,} objects")
    yield


app = FastAPI(lifespan=lifespan)


def get_row(request: Request, obj_id: str) -> ManifestRow:
    row = request.app.state.rows.get(obj_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no object {obj_id!r}")
    return row


# plain def, not async def: fastapi runs these in a thread pool, so slow file reads and torch calls don't block other requests
@app.get("/objects")
def list_objects(request: Request) -> list[dict]:
    # manifest order (class, then id), so the frontend can group by class without sorting
    return [{"id": r.id, "class_name": r.class_name, "split": r.split} for r in request.app.state.rows.values()]


@app.get("/objects/{obj_id}/mesh")
def get_mesh(obj_id: str, request: Request) -> Response:
    row = get_row(request, obj_id)
    # same normalization as training, so the mesh on screen and the points the models saw share one coordinate frame
    mesh = normalize(load_mesh(request.app.state.mesh_root / row.relpath))
    # binary ply: smaller than ascii, and three.js's PLYLoader reads both
    return Response(mesh.export(file_type="ply"), media_type="application/octet-stream")


class PredictRequest(BaseModel):
    id: str


@app.post("/predict")
def predict(body: PredictRequest, request: Request) -> dict:
    state = request.app.state
    row = get_row(request, body.id)
    points = state.points[row.id]
    predictions = {}
    for arch, model in state.models.items():
        # batch of one: the models expect a leading batch dimension
        x = torch.from_numpy(PREPARE[arch](points)).unsqueeze(0).to(DEVICE)
        # inference_mode skips all autograd bookkeeping: a stricter, slightly faster no_grad
        with torch.inference_mode():
            # the models return logits; softmax turns them into probabilities only for display
            probs = torch.softmax(model(x), dim=1)[0]
        predictions[arch] = {
            "predicted": state.classes[int(probs.argmax())],
            "probabilities": dict(zip(state.classes, probs.tolist(), strict=True)),
        }
    return {"id": row.id, "true_class": row.class_name, "split": row.split, "predictions": predictions}