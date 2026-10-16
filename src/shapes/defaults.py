"""Level-1 training defaults, shared by train.py (its CLI defaults) and the backend (the Train form's prefill)."""

# level 1: conventional, cited defaults (reasons in the decisions doc); any later change is judged on val and logged
TRAINING = {"lr": 1e-3, "weight_decay": 1e-2, "max_epochs": 200, "patience": 15, "seed": 0}

MODEL_DEFAULTS = {
    # conventional values, not from a paper
    "mlp": {"batch_size": 64, "hidden": [256], "dropout": 0.5, "activation": "relu"},
    # batch 32 and dropout 0.3 are the pointnet paper's values (appendix c)
    "pointnet": {"batch_size": 32, "dropout": 0.3},
}
