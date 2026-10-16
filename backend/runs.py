"""Training runs started from the app: what a run request may contain, the train.py command it becomes,
and the manager that starts, polls, and cancels the process."""

import json
import os
import re
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shapes.defaults import MODEL_DEFAULTS, TRAINING
from shapes.model import ACTIVATIONS

# resolved from this file, like main.py's ROOT, so the command doesn't depend on where uvicorn was started
TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "training" / "train.py"

# upper limits (sabine, oct 16): they turn typos into readable 422s; they are not claims about good values
MAX_EPOCHS = 500
MAX_HIDDEN_LAYERS = 4
MAX_HIDDEN_UNITS = 5000
# any batch at or above the train split's 3,392 objects is full-batch gradient descent, so this only stops absurd values
MAX_MLP_BATCH = 4096
# batch norm in train mode fails on a batch of one object; the last batch holds 3,392 mod b objects, and the only
# b that leave 1 are 1 and 3,391 (3,391 is prime), so a 2..512 range rules both out (and bounds activation memory)
MIN_POINTNET_BATCH, MAX_POINTNET_BATCH = 2, 512
# np.random.seed accepts only 0 ..< 2**32
MAX_SEED = 2**32 - 1


class _RunBase(BaseModel):
    # strict: 256.0 or "256" is a 422, not quietly turned into an int
    # extra="forbid": a field this model doesn't use (hidden for pointnet) is an error, not silently ignored
    # allow_inf_nan=False: inf and nan are refused here rather than by argparse after the process starts
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    lr: float = Field(TRAINING["lr"], gt=0)
    weight_decay: float = Field(TRAINING["weight_decay"], ge=0)
    max_epochs: int = Field(TRAINING["max_epochs"], ge=1, le=MAX_EPOCHS)
    # no "patience <= max_epochs" rule: a larger patience just means early stopping never triggers, a valid choice
    patience: int = Field(TRAINING["patience"], ge=1, le=MAX_EPOCHS)
    seed: int = Field(TRAINING["seed"], ge=0, le=MAX_SEED)
    # null means a constant learning rate (level 1); otherwise halve it every lr_step epochs
    lr_step: int | None = Field(TRAINING["lr_step"], ge=1, le=MAX_EPOCHS)


class MlpRun(_RunBase):
    model: Literal["mlp"]
    batch_size: int = Field(MODEL_DEFAULTS["mlp"]["batch_size"], ge=1, le=MAX_MLP_BATCH)
    dropout: float = Field(MODEL_DEFAULTS["mlp"]["dropout"], ge=0, lt=1)
    # default_factory hands each request its own list, so no two requests share one mutable default
    hidden: list[Annotated[int, Field(ge=1, le=MAX_HIDDEN_UNITS)]] = Field(
        default_factory=lambda: list(MODEL_DEFAULTS["mlp"]["hidden"]), min_length=1, max_length=MAX_HIDDEN_LAYERS)
    activation: str = MODEL_DEFAULTS["mlp"]["activation"]

    # checked against model.py's dict, so a new activation there is accepted here without a second list to update
    @field_validator("activation")
    @classmethod
    def known_activation(cls, v: str) -> str:
        if v not in ACTIVATIONS:
            raise ValueError(f"must be one of {sorted(ACTIVATIONS)}")
        return v


class PointNetRun(_RunBase):
    model: Literal["pointnet"]
    batch_size: int = Field(MODEL_DEFAULTS["pointnet"]["batch_size"], ge=MIN_POINTNET_BATCH, le=MAX_POINTNET_BATCH)
    dropout: float = Field(MODEL_DEFAULTS["pointnet"]["dropout"], ge=0, lt=1)


# the "model" field picks which class validates the rest, so errors name the right fields for that model
RunRequest = Annotated[MlpRun | PointNetRun, Field(discriminator="model")]
# one request class per trainable architecture; main.py checks this against MODELS at startup
REQUEST_MODELS = {"mlp": MlpRun, "pointnet": PointNetRun}


def build_command(req: MlpRun | PointNetRun, out_dir: Path) -> list[str]:
    # a list, never a shell string: each value is one argv entry, so nothing in it can be read as shell syntax
    # every flag spelled out, so the saved command reproduces the run even if shapes.defaults changes later
    # sys.executable is the python running the server, so the child uses the same environment
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--model", req.model,
        "--out-dir", str(out_dir),
        "--lr", str(req.lr),
        "--weight-decay", str(req.weight_decay),
        "--max-epochs", str(req.max_epochs),
        "--patience", str(req.patience),
        "--seed", str(req.seed),
        "--batch-size", str(req.batch_size),
        "--dropout", str(req.dropout),
    ]
    # the one flag that can't always be spelled out: "off" is the flag's absence, so the saved command relies on
    # train.py's default for it; the record's "request" field still says lr_step: null
    if req.lr_step is not None:
        cmd += ["--lr-step", str(req.lr_step)]
    if isinstance(req, MlpRun):
        cmd += ["--hidden", *(str(h) for h in req.hidden), "--activation", req.activation]
    return cmd


# the server makes every id, and every id is checked against this before it becomes a path, so "../x" can't escape runs/
RUN_ID = re.compile(r"\d{8}T\d{6}Z-(" + "|".join(REQUEST_MODELS) + ")")
LOG_TAIL_LINES = 20


def write_json(path: Path, data: dict) -> None:
    # write a temporary file, then rename it over the old one: os.replace is atomic, so a reader (or a crash)
    # sees either the old file or the new one, never half of each
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def read_progress(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text().split("\n")
    # the last piece is "" after a complete line, or a line train.py is still writing; either way, not a record yet
    return [json.loads(line) for line in lines[:-1] if line]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunManager:
    """At most one training subprocess at a time; each run's record lives in runs/<id>/run.json."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        # routes run in a thread pool, so two requests can arrive together; the lock makes check-then-start one step
        self.lock = threading.Lock()
        self.active: tuple[str, subprocess.Popen] | None = None

    def mark_interrupted(self) -> None:
        # at startup no run can be ours, so a record still saying "running" lost its server mid-run
        # (any surviving process is not tracked: a stored pid may have been reused by an unrelated process)
        self.runs_dir.mkdir(exist_ok=True)
        for record_path in self.runs_dir.glob("*/run.json"):
            record = json.loads(record_path.read_text())
            if record["status"] == "running":
                # train.py writes its results file last, so if it exists the run finished; its exit code is lost
                finished = (record_path.parent / f"{record['model']}.json").exists()
                write_json(record_path, record | {"status": "done" if finished else "interrupted", "ended_at": now()})

    def start(self, req: MlpRun | PointNetRun) -> dict:
        with self.lock:
            self._refresh()
            if self.active is not None:
                raise HTTPException(409, f"run {self.active[0]} is still running; cancel it or wait")
            run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{req.model}"
            run_dir = self.runs_dir / run_id
            try:
                # exist_ok=False: a cancel and a restart within one second would otherwise share a folder
                run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                raise HTTPException(409, "a run started this second; try again") from None
            command = build_command(req, run_dir)
            record = {"id": run_id, "model": req.model, "request": req.model_dump(), "command": command,
                      "status": "running", "created_at": now(), "ended_at": None, "returncode": None}
            # written before the process starts, so even a run that fails to launch has a record
            write_json(run_dir / "run.json", record)
            try:
                # a file, not stdout=PIPE: an unread pipe fills (about 64 KiB on linux) and the child blocks on print
                with open(run_dir / "log.txt", "w") as log:
                    # cwd=ROOT because train.py's --points-dir and --meta defaults are relative paths
                    # the child keeps its own copy of the log's file descriptor, so closing ours here is fine
                    proc = subprocess.Popen(command, cwd=TRAIN_SCRIPT.parent.parent,
                                            stdout=log, stderr=subprocess.STDOUT)
            except OSError as e:
                write_json(run_dir / "run.json", record | {"status": "failed", "ended_at": now(), "error": str(e)})
                raise HTTPException(500, f"could not start training: {e}") from None
            self.active = (run_id, proc)
            return record

    def _refresh(self) -> None:
        # caller holds the lock; poll() returns None while running, and reaps the child once it has exited
        if self.active is None:
            return
        run_id, proc = self.active
        code = proc.poll()
        if code is None:
            return
        record_path = self.runs_dir / run_id / "run.json"
        record = json.loads(record_path.read_text())
        # exit 0 without a results file would mean train.py stopped early without raising; call that a failure
        results_exist = (self.runs_dir / run_id / f"{record['model']}.json").exists()
        status = "done" if code == 0 and results_exist else "failed"
        write_json(record_path, record | {"status": status, "ended_at": now(), "returncode": code})
        self.active = None

    def _record(self, run_id: str) -> dict:
        record_path = self.runs_dir / run_id / "run.json"
        if not RUN_ID.fullmatch(run_id) or not record_path.exists():
            raise HTTPException(404, f"no run {run_id!r}")
        return json.loads(record_path.read_text())

    def get(self, run_id: str) -> dict:
        with self.lock:
            self._refresh()
            record = self._record(run_id)
        run_dir = self.runs_dir / run_id
        record["progress"] = read_progress(run_dir / "progress.jsonl")
        if record["status"] == "failed":
            log = run_dir / "log.txt"
            record["log_tail"] = log.read_text().splitlines()[-LOG_TAIL_LINES:] if log.exists() else []
        if record["status"] == "done":
            record["results"] = json.loads((run_dir / f"{record['model']}.json").read_text())
        return record

    def list(self) -> list[dict]:
        with self.lock:
            self._refresh()
            records = [json.loads(p.read_text()) for p in self.runs_dir.glob("*/run.json") if RUN_ID.fullmatch(p.parent.name)]
        # ids start with a utc timestamp, so sorting them is sorting by start time
        return sorted(records, key=lambda r: r["id"], reverse=True)

    def cancel(self, run_id: str) -> dict:
        with self.lock:
            self._refresh()
            record = self._record(run_id)
            if self.active is None or self.active[0] != run_id:
                raise HTTPException(409, f"run {run_id} is not running (status: {record['status']})")
            proc = self.active[1]
            self._stop(proc)
            record |= {"status": "cancelled", "ended_at": now(), "returncode": proc.returncode}
            write_json(self.runs_dir / run_id / "run.json", record)
            self.active = None
            return record

    def shutdown(self) -> None:
        # called when the server stops, so a run never outlives the server that could report on it
        with self.lock:
            # a run that finished since the last request is recorded as done, not interrupted
            self._refresh()
            if self.active is not None:
                run_id, proc = self.active
                self._stop(proc)
                record_path = self.runs_dir / run_id / "run.json"
                record = json.loads(record_path.read_text())
                write_json(record_path, record | {"status": "interrupted", "ended_at": now(), "returncode": proc.returncode})
                self.active = None

    @staticmethod
    def _stop(proc: subprocess.Popen) -> None:
        # sigterm first (python's default handling exits at once); sigkill only if it hasn't exited in 10 s
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            