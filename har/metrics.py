"""Metrics + result files. OWNER: Person 1 (written at the Day-1 kickoff, then frozen).

Every experiment writes ONE JSON file per (model, feature_set) to results/metrics/
and one confusion-matrix CSV to results/confusion/. Both people use save_result().
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from har.config import LABEL_ORDER, RESULTS_DIR

REQUIRED_KEYS = {"model", "owner", "feature_set", "protocol", "seed", "n_train", "n_test",
                 "train_accuracy", "test_accuracy", "test_macro_f1", "params",
                 "fit_seconds", "predict_seconds", "paper_test_accuracy", "notes"}


def evaluate(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER,
                                       zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                          zero_division=0))}


def confusion_df(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    return pd.DataFrame(cm, index=[f"true_{c}" for c in LABEL_ORDER],
                        columns=[f"pred_{c}" for c in LABEL_ORDER])


def versions() -> dict:
    v = {"python": platform.python_version(), "numpy": np.__version__,
         "pandas": pd.__version__, "sklearn": sklearn.__version__}
    try:
        import tensorflow as tf
        v["tensorflow"] = tf.__version__
    except ImportError:
        pass
    return v


def save_result(record: dict, y_true=None, y_pred=None, results_dir: Path = RESULTS_DIR,
                tag: str = "") -> Path:
    missing = REQUIRED_KEYS - record.keys()
    if missing:
        raise KeyError(f"result record missing keys: {sorted(missing)}")
    record = {**record, "versions": versions(),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    stem = f"{record['model']}_{record['feature_set']}{tag}"
    out = Path(results_dir) / "metrics" / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=float))
    if y_true is not None and y_pred is not None:
        cdir = Path(results_dir) / "confusion"
        cdir.mkdir(parents=True, exist_ok=True)
        confusion_df(y_true, y_pred).to_csv(cdir / f"{stem}.csv")
    return out
