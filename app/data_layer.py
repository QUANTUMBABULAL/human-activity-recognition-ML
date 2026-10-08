"""Read-only data access for the Streamlit dashboard.

Everything here READS existing project files; nothing trains a model or writes to results/.
Kept free of Streamlit so it can be unit-tested; app/streamlit_app.py wraps these in caches.

Sources:
  results comparison  -> experiments.compare_results.discover/build_table (same rules as the CLI)
  result details      -> results/metrics/<stem>.json
  confusion matrices  -> results/confusion/<stem>.csv (12 x 12, LABEL_ORDER)
  dataset statistics  -> results/split_meta.json, results/cleaning_log.json, har.config
  test samples        -> data/processed/*.parquet + the frozen split (test rows only)
  demo model          -> artifacts/<stem>.joblib + <stem>_metadata.json (written only by
                         app/export_demo_model.py, which the dashboard never runs)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from har.config import (ACTIVITY_NAMES, FEATURE_SETS, LABEL_ORDER, PROCESSED_DIR, REPO_ROOT,
                        RESULTS_DIR, SEED, TEST_FRACTION)

MODELS_DIR = REPO_ROOT / "artifacts"            # deployment artifacts for the prediction demo
METADATA_SUFFIX = "_metadata.json"              # <stem>_metadata.json next to <stem>.joblib
STATUS_ORDER = ["FULL", "SUBSAMPLE", "UNVERIFIED", "QUICK", "MALFORMED", "NOT RUN"]
RESULT_STATUSES = ["FULL", "SUBSAMPLE", "UNVERIFIED", "QUICK"]   # statuses that carry real metrics
# Models trained on standardised features: an artifact is only usable if it carries its own scaler.
NEEDS_SCALER = {"logreg", "svm_rbf", "mlp"}


# ----------------------------------------------------------------------------- results

def load_comparison(results_dir: Path = RESULTS_DIR):
    """(table, problems) using the project's own comparison logic. Never writes a file."""
    from experiments import compare_results as cr
    rows, problems = cr.discover(results_dir)
    return cr.build_table(rows, problems), problems


def completed_results(table: pd.DataFrame) -> pd.DataFrame:
    """Only configurations that actually produced metrics (NOT RUN / MALFORMED rows dropped)."""
    keep = table["status"].isin(RESULT_STATUSES) & table["test_accuracy"].notna()
    return table[keep].reset_index(drop=True)


def feature_set_comparison(table: pd.DataFrame) -> pd.DataFrame:
    from experiments import compare_results as cr
    return cr.feature_set_comparison(table)


def paper_tables(table: pd.DataFrame):
    from experiments import compare_results as cr
    return cr.paper_table(table)


def load_result_json(stem: str, results_dir: Path = RESULTS_DIR) -> dict | None:
    """The raw result record, or None if missing / unreadable."""
    try:
        rec = json.loads((Path(results_dir) / "metrics" / f"{stem}.json").read_text())
        return rec if isinstance(rec, dict) else None
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------- confusion

def list_confusion_stems(results_dir: Path = RESULTS_DIR) -> list[str]:
    d = Path(results_dir) / "confusion"
    return sorted(p.stem for p in d.glob("*.csv")) if d.is_dir() else []


def load_confusion(stem: str, results_dir: Path = RESULTS_DIR) -> pd.DataFrame | None:
    """12 x 12 raw counts indexed by activity name (true rows, predicted columns), or None
    if the file is missing or does not match the project's LABEL_ORDER layout."""
    try:
        cm = pd.read_csv(Path(results_dir) / "confusion" / f"{stem}.csv", index_col=0)
    except (OSError, ValueError):
        return None
    want_rows = [f"true_{c}" for c in LABEL_ORDER]
    want_cols = [f"pred_{c}" for c in LABEL_ORDER]
    if list(cm.index) != want_rows or list(cm.columns) != want_cols:
        return None
    names = [ACTIVITY_NAMES[c] for c in LABEL_ORDER]
    return pd.DataFrame(cm.to_numpy(dtype=np.int64), index=names, columns=names)


def row_normalise(cm: pd.DataFrame) -> pd.DataFrame:
    """Each row divided by its total (= recall per true class). Empty rows stay 0."""
    a = cm.to_numpy(dtype=float)
    s = a.sum(axis=1, keepdims=True)
    return pd.DataFrame(np.divide(a, s, out=np.zeros_like(a), where=s > 0),
                        index=cm.index, columns=cm.columns)


def per_class_recall(cm: pd.DataFrame) -> pd.Series:
    a = cm.to_numpy(dtype=float)
    s = a.sum(axis=1)
    return pd.Series(np.divide(np.diag(a), s, out=np.full_like(s, np.nan), where=s > 0),
                     index=cm.index)


def top_confusions(cm: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Largest off-diagonal cells, computed from the matrix itself."""
    a = cm.to_numpy(dtype=np.int64).copy()
    np.fill_diagonal(a, 0)
    totals = cm.to_numpy().sum(axis=1)
    out = []
    for flat in np.argsort(a, axis=None)[::-1][:k]:
        i, j = np.unravel_index(flat, a.shape)
        if a[i, j] == 0:
            break
        out.append({"True activity": cm.index[i], "Predicted as": cm.columns[j],
                    "Count": int(a[i, j]), "Share of true class": a[i, j] / totals[i]})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- dataset

def _read_json(path: Path) -> dict | None:
    try:
        d = json.loads(Path(path).read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def dataset_stats(results_dir: Path = RESULTS_DIR) -> dict:
    """Statistics read from files written by scripts.build_dataset (values None if missing)."""
    meta = _read_json(Path(results_dir) / "split_meta.json") or {}
    log = _read_json(Path(results_dir) / "cleaning_log.json") or {}
    subjects = log.get("subjects") or []
    counts = {int(k): int(v) for k, v in (log.get("class_counts") or {}).items()}
    return {
        "n_subjects": len(subjects) or None,
        "n_activities": len(ACTIVITY_NAMES),
        "n_rows": meta.get("n_rows"), "n_train": meta.get("n_train"), "n_test": meta.get("n_test"),
        "n_reduced": len(FEATURE_SETS["reduced"]), "n_full": len(FEATURE_SETS["full"]),
        "seed": meta.get("seed", SEED), "test_fraction": meta.get("test_fraction", TEST_FRACTION),
        "split_fingerprint": meta.get("fingerprint"), "rows_fingerprint": meta.get("rows_fingerprint"),
        "hr_method": meta.get("hr_method") or log.get("hr_method"),
        "rows_raw": sum(s.get("rows_raw", 0) for s in subjects) or None,
        "rows_label0": sum(s.get("rows_label0", 0) for s in subjects) or None,
        "rows_dropped_nan": sum(s.get("rows_dropped_nan", 0) for s in subjects) or None,
        "subjects": subjects,
        "class_counts": counts,
    }


def load_test_samples(processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, dict]:
    """Held-out TEST rows only (31 full-set columns + ids + label) and a verification dict.

    Raises FileNotFoundError if the processed parquet or split file is missing.
    """
    import pyarrow.parquet as pq
    from har.data import PROCESSED_FILE
    from har.splits import SPLIT_FILE, load_split, split_fingerprint

    split = load_split(Path(processed_dir) / SPLIT_FILE)
    cols = ["row_id", "subject_id", "timestamp", "activity_id"] + FEATURE_SETS["full"]
    table = pq.read_table(Path(processed_dir) / PROCESSED_FILE, columns=cols)
    test = table.take(split["test"]).to_pandas()
    meta = _read_json(RESULTS_DIR / "split_meta.json") or {}
    check = {"fingerprint": split_fingerprint(split), "expected": meta.get("fingerprint"),
             "n_test": int(len(test)), "row_id_matches_position":
                 bool(np.array_equal(test["row_id"].to_numpy(), split["test"]))}
    check["ok"] = check["fingerprint"] == check["expected"] and check["row_id_matches_position"]
    return test.reset_index(drop=True), check


# ----------------------------------------------------------------------------- demo model

def find_demo_models(models_dir: Path | None = None) -> list[dict]:
    """Metadata of every exported demo model (<stem>_metadata.json next to <stem>.joblib)."""
    out = []
    d = Path(models_dir or MODELS_DIR)
    if not d.is_dir():
        return out
    for meta_path in sorted(d.glob(f"*{METADATA_SUFFIX}")):
        meta = _read_json(meta_path)
        art = meta_path.with_name(meta_path.name[:-len(METADATA_SUFFIX)] + ".joblib")
        if not meta or not art.is_file():
            continue
        if meta.get("feature_set") not in FEATURE_SETS:
            continue
        out.append({**meta, "artifact_path": str(art), "artifact_mb": art.stat().st_size / 1e6})
    return out


def load_demo_model(artifact_path: str, feature_set: str, model_key: str | None = None):
    """Load an exported model and check it fits the feature set. Raises ValueError otherwise.

    For models trained on standardised inputs (NEEDS_SCALER) the artifact must be a Pipeline whose
    first step is a scaler, otherwise raw sensor values would be fed to it unscaled."""
    import joblib
    model = joblib.load(artifact_path)
    if not hasattr(model, "predict"):
        raise ValueError("artifact has no predict()")
    if model_key in NEEDS_SCALER:
        steps = getattr(model, "steps", None)
        if not steps or "scal" not in type(steps[0][1]).__name__.lower():
            raise ValueError(f"'{model_key}' needs its scaler, but the artifact has no scaling step")
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and n_in != len(FEATURE_SETS[feature_set]):
        raise ValueError(f"model expects {n_in} features, feature set '{feature_set}' has "
                         f"{len(FEATURE_SETS[feature_set])}")
    return model


def predict_one(model, features: np.ndarray) -> tuple[int, pd.Series | None]:
    """(predicted activity id, probabilities indexed by activity name or None).

    Probabilities are returned only if the model really implements predict_proba."""
    X = np.asarray(features, dtype=np.float32).reshape(1, -1)
    pred = int(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            p = np.asarray(model.predict_proba(X))[0]
            classes = [int(c) for c in getattr(model, "classes_", LABEL_ORDER)]
            proba = pd.Series(p, index=[ACTIVITY_NAMES.get(c, str(c)) for c in classes])
        except (AttributeError, NotImplementedError, ValueError):
            proba = None
    return pred, proba
