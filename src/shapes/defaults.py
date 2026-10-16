"""Level-1 training defaults, shared by train.py (its CLI defaults) and the backend (the Train form's prefill and labels)."""

# level 1: conventional, cited defaults (reasons in the decisions doc); any later change is judged on val and logged
# lr_step None means a constant learning rate, as at level 1; a number halves the rate every that many epochs
TRAINING = {"lr": 1e-3, "weight_decay": 1e-2, "max_epochs": 200, "patience": 15, "seed": 0, "lr_step": None}

MODEL_DEFAULTS = {
    # conventional values, not from a paper
    "mlp": {"batch_size": 64, "hidden": [256], "dropout": 0.5, "activation": "relu"},
    # batch 32 and dropout 0.3 are the pointnet paper's values (appendix c)
    "pointnet": {"batch_size": 32, "dropout": 0.3},
}

# where each default comes from, shown beside it in the train form; kept next to the numbers so one isn't changed
# without the other in view (main.py checks at startup that every form field has exactly one entry; it can't check the wording)
_SHARED_SOURCES = {
    "weight_decay": "PyTorch AdamW default",
    "max_epochs": "our level-1 choice",
    "patience": "our level-1 choice",
    "seed": "arbitrary",
}
SOURCES = {
    "mlp": _SHARED_SOURCES | {
        "lr": "Adam paper's default",
        "lr_step": "off at level 1",
        "batch_size": "conventional, not cited",
        "hidden": "conventional, not cited",
        "dropout": "conventional, not cited",
        "activation": "conventional, not cited",
    },
    "pointnet": _SHARED_SOURCES | {
        "lr": "PointNet paper (App. C); also Adam's default",
        "lr_step": "off at level 1; the PointNet paper (App. C) halves it every 20 epochs",
        "batch_size": "PointNet paper (App. C)",
        "dropout": "PointNet paper (App. C): keep probability 0.7",
    },
}
