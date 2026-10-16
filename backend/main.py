"""hw2 model server: loads both trained classifiers once at startup and serves objects, meshes, and predictions."""

import os
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from shapes.manifest import ManifestRow, read_manifest
from shapes.model import MODELS
from shapes.pointcloud import GRID, N_POINTS, load_mesh, load_points, normalize, voxelize
from runs import REQUEST_MODELS, RunManager, RunRequest


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
# objects per forward pass; pointnet's widest activation is batch × 1,024 points × 1,024 channels of float32,
# about 268 MB at 64 vs. about 3 GB for the largest class-and-split (755 train chairs) in one pass
BATCH = 64


# label reviews are user-generated state: the file sits beside the code but is gitignored
DB_PATH = ROOT / "backend" / "reviews.db"
# bump when the table changes; startup refuses a file written for a different layout
SCHEMA_VERSION = 1
Verdict = Literal["ok", "wrong", "ambiguous"]


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


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        # the connection's own context manager commits on success and rolls back on an exception, but doesn't close
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            return
        # create table if not exists would skip an old table silently, so a file from another layout fails here instead
        if version != 0:
            raise RuntimeError(f"{DB_PATH.name} has schema version {version}; this server expects {SCHEMA_VERSION}")
        # built from the same Literal the api validates with, so the two lists of verdicts can't drift apart
        verdicts = ", ".join(f"'{v}'" for v in get_args(Verdict))
        conn.execute(f"""
            CREATE TABLE reviews (
                review_id      INTEGER PRIMARY KEY,
                object_id      TEXT    NOT NULL,
                verdict        TEXT    NOT NULL CHECK (verdict IN ({verdicts})),
                should_be      TEXT,
                mesh_problem   INTEGER NOT NULL CHECK (mesh_problem IN (0, 1)),
                saw_prediction INTEGER NOT NULL CHECK (saw_prediction IN (0, 1)),
                note           TEXT,
                created_at     TEXT    NOT NULL,
                CHECK ((verdict = 'wrong') = (should_be IS NOT NULL))
            )
        """)
        # history per object, newest first, without scanning the table
        conn.execute("CREATE INDEX reviews_by_object ON reviews (object_id, review_id)")
        # pragmas don't take ? placeholders; the value is our own integer constant, never input
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def review_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    # sqlite has no boolean type: the columns hold 0/1, and the api speaks true/false
    d["mesh_problem"] = bool(d["mesh_problem"])
    d["saw_prediction"] = bool(d["saw_prediction"])
    return d


@asynccontextmanager
async def lifespan(app: FastAPI):
    # a model registered in model.py without an input recipe here would otherwise fail only on the first click
    if PREPARE.keys() != MODELS.keys():
        raise RuntimeError(f"PREPARE covers {sorted(PREPARE)}; MODELS has {sorted(MODELS)}")
    # a model in model.py without a run request class would be listed but untrainable; fail at startup instead
    if REQUEST_MODELS.keys() != MODELS.keys():
        raise RuntimeError(f"REQUEST_MODELS covers {sorted(REQUEST_MODELS)}; MODELS has {sorted(MODELS)}")
    init_db()
    print(f"reviews database: {DB_PATH}")
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
    app.state.runs = RunManager(ROOT / "runs")
    app.state.runs.mark_interrupted()
    yield
    # lifespan code after yield runs when the server shuts down, including on ctrl-c
    app.state.runs.shutdown()


app = FastAPI(lifespan=lifespan)


def get_row(request: Request, obj_id: str) -> ManifestRow:
    row = request.app.state.rows.get(obj_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no object {obj_id!r}")
    return row


def run_models(state, ids: list[str]) -> dict[str, np.ndarray]:
    """class probabilities per architecture, shape (len(ids), n_classes): the one forward path behind /predict and /predictions."""
    out = {}
    for arch, model in state.models.items():
        chunks = []
        for start in range(0, len(ids), BATCH):
            batch = np.stack([PREPARE[arch](state.points[i]) for i in ids[start:start + BATCH]])
            x = torch.from_numpy(batch).to(DEVICE)
            # eval mode: batch norm uses its running averages, so an object's output doesn't depend on which others share its batch
            with torch.inference_mode():
                # the models return logits; softmax turns them into probabilities only for display
                chunks.append(torch.softmax(model(x), dim=1).numpy())
        out[arch] = np.concatenate(chunks)
    return out


# plain def, not async def: fastapi runs these in a thread pool, so slow file reads and torch calls don't block other requests
@app.get("/objects")
def list_objects(request: Request) -> list[dict]:
    # manifest order (class, then id), so the frontend can group by class without sorting
    return [{"id": r.id, "class_name": r.class_name, "split": r.split} for r in request.app.state.rows.values()]


@app.get("/objects/{obj_id}")
def get_object(obj_id: str, request: Request) -> dict:
    row = get_row(request, obj_id)
    # the same load as /mesh (unused vertices dropped), so the counts describe the mesh on screen
    mesh = load_mesh(request.app.state.mesh_root / row.relpath)
    return {
        "id": row.id,
        "class_name": row.class_name,
        "split": row.split,
        # val is carved out of modelnet's train folder, so split alone doesn't say where the file came from
        "source_split": row.source_split,
        "relpath": row.relpath,
        "n_vertices": len(mesh.vertices),
        "n_faces": len(mesh.faces),
    }


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
    # a batch of one through the same path as /predictions, so the list's label and this one can't disagree
    probs = run_models(state, [row.id])
    predictions = {
        arch: {
            "predicted": state.classes[int(p[0].argmax())],
            "probabilities": dict(zip(state.classes, p[0].tolist(), strict=True)),
        }
        for arch, p in probs.items()
    }
    return {"id": row.id, "true_class": row.class_name, "split": row.split, "predictions": predictions}


@app.get("/predictions")
def list_predictions(request: Request, split: Literal["train", "val", "test"], class_name: str | None = None) -> list[dict]:
    state = request.app.state
    # an unknown class would match nothing and return an empty list, which looks exactly like "no errors"
    if class_name is not None and class_name not in state.classes:
        raise HTTPException(status_code=422, detail=f"unknown class {class_name!r}")
    rows = [r for r in state.rows.values() if r.split == split and class_name in (None, r.class_name)]
    if not rows:
        return []
    probs = run_models(state, [r.id for r in rows])
    # labels only: the list needs right or wrong, and the full breakdown comes from /predict on click
    return [
        {
            "id": r.id,
            "true_class": r.class_name,
            "predicted": {arch: state.classes[int(p[i].argmax())] for arch, p in probs.items()},
        }
        for i, r in enumerate(rows)
    ]


class ReviewIn(BaseModel):
    verdict: Verdict
    # the class the object should have: required for "wrong", not allowed otherwise
    should_be: str | None = None
    mesh_problem: bool = False
    # no default: the frontend must say whether the predictions were on screen, so a forgotten field can't be recorded as "no"
    saw_prediction: bool
    note: str | None = None


@app.post("/objects/{obj_id}/reviews", status_code=201)
def add_review(obj_id: str, body: ReviewIn, request: Request) -> dict:
    row = get_row(request, obj_id)
    # the table enforces the same rule; checking here returns a readable 422 instead of a 500 from sqlite
    if (body.verdict == "wrong") != (body.should_be is not None):
        raise HTTPException(status_code=422, detail="should_be is required for verdict 'wrong' and not allowed otherwise")
    if body.should_be is not None:
        if body.should_be not in request.app.state.classes:
            raise HTTPException(status_code=422, detail=f"unknown class {body.should_be!r}")
        if body.should_be == row.class_name:
            raise HTTPException(status_code=422, detail=f"{row.class_name!r} is the current label; use verdict 'ok'")
    # a blank note is stored as null, so "no note" has one representation
    note = (body.note or "").strip() or None
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO reviews (object_id, verdict, should_be, mesh_problem, saw_prediction, note, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            # placeholders: values travel separately from the sql, so no note can change the statement
            (row.id, body.verdict, body.should_be, body.mesh_problem, body.saw_prediction, note,
             datetime.now(UTC).isoformat(timespec="seconds")),
        )
        new = conn.execute("SELECT * FROM reviews WHERE review_id = ?", (cur.lastrowid,)).fetchone()
    return review_dict(new)


@app.get("/objects/{obj_id}/reviews")
def object_reviews(obj_id: str, request: Request) -> list[dict]:
    row = get_row(request, obj_id)
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE object_id = ? ORDER BY review_id DESC", (row.id,)
        ).fetchall()
    # newest first: the first entry is the current verdict, the rest are how it got there
    return [review_dict(r) for r in rows]


@app.get("/reviews")
def latest_reviews() -> list[dict]:
    # one row per reviewed object, its latest; review_id, not created_at, because timestamps can tie within a second
    with db() as conn:
        rows = conn.execute("""
            SELECT * FROM reviews AS r
            WHERE review_id = (SELECT MAX(review_id) FROM reviews WHERE object_id = r.object_id)
            ORDER BY object_id
        """).fetchall()
    return [review_dict(r) for r in rows]


# declared before /runs/{run_id}: routes match in order, so "defaults" would otherwise be read as a run id
@app.get("/runs/defaults")
def run_defaults() -> dict:
    # the request classes hold the defaults, so the form's prefill is exactly what an empty request would run
    return {arch: cls(model=arch).model_dump() for arch, cls in REQUEST_MODELS.items()}


@app.post("/runs", status_code=201)
def start_run(body: RunRequest, request: Request) -> dict:
    return request.app.state.runs.start(body)


@app.get("/runs")
def list_runs(request: Request) -> list[dict]:
    return request.app.state.runs.list()


@app.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    return request.app.state.runs.get(run_id)


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, request: Request) -> dict:
    return request.app.state.runs.cancel(run_id)

