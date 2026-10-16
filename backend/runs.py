"""Training runs started from the app: what a run request may contain, and the train.py command it becomes."""

import sys
from pathlib import Path
from typing import Annotated, Literal

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
    if isinstance(req, MlpRun):
        cmd += ["--hidden", *(str(h) for h in req.hidden), "--activation", req.activation]
    return cmd
