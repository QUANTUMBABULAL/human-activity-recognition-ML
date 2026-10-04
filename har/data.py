"""Loading and cleaning PAMAP2. OWNER: Person 1.

Public interface (frozen at kickoff):
    build_dataset(raw_dir, hr_method="linear") -> (DataFrame, cleaning_log dict)
    load_processed(path=None)                  -> DataFrame
    get_xy(df, feature_set, idx=None)          -> (X float32 [n, d], y int64 [n])
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from har.config import (FEATURE_SETS, FORBIDDEN_FEATURES, LABEL_ORDER,
                        PROCESSED_DIR, RAW_COLUMNS, TRANSIENT_LABEL)

KEEP_RAW = ["timestamp", "activity_id"] + FEATURE_SETS["full"]  # read only what we use
PROCESSED_FILE = "pamap2_protocol_clean.parquet"


def load_subject(path: str | Path) -> pd.DataFrame:
    """Read one subjectXXX.dat file; keep only the columns we need (memory-aware)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS,
                     usecols=KEEP_RAW, dtype="float64")  # header=None: line 1 is DATA
    df["activity_id"] = df["activity_id"].astype("int16")
    float_cols = [c for c in KEEP_RAW if c not in ("timestamp", "activity_id")]
    df[float_cols] = df[float_cols].astype("float32")
    return df


def fill_heart_rate(df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
    """Fill the ~9 Hz heart-rate channel up to the 100 Hz IMU rate, within ONE subject.

    method="linear": report text - linear interpolation between nearest valid readings
        (in time). Rows before the first / after the last reading stay NaN
        (limit_area="inside") and are dropped later - no extrapolation.
    method="ffill":  authors' public code - carry the previous reading forward.
    """
    df = df.sort_values("timestamp").copy()
    hr = df.set_index("timestamp")["heart_rate"].astype("float64")
    if method == "linear":
        hr = hr.interpolate(method="index", limit_area="inside")
    elif method == "ffill":
        hr = hr.ffill()
    else:
        raise ValueError(f"unknown hr_method {method!r}")
    df["heart_rate"] = hr.to_numpy(dtype="float32")
    return df


def clean_subject(df: pd.DataFrame, subject_id: int, hr_method: str = "linear"):
    """Apply the cleaning policy to one subject. Returns (clean_df, log_dict)."""
    log = {"subject_id": subject_id, "rows_raw": int(len(df)),
           "hr_nan_frac_raw": float(df["heart_rate"].isna().mean()),
           "rows_label0": int((df["activity_id"] == TRANSIENT_LABEL).sum())}
    if hr_method == "ffill":   # authors' code order: drop label 0 first
        df = df[df["activity_id"] != TRANSIENT_LABEL]
        df = fill_heart_rate(df, "ffill")
    else:                      # interpolate on the full time series first
        df = fill_heart_rate(df, hr_method)
    df = df[df["activity_id"].isin(LABEL_ORDER)]
    log["rows_after_label_filter"] = int(len(df))
    feats = FEATURE_SETS["full"]                 # same rows for BOTH feature sets
    nan_rows = df[feats].isna().any(axis=1)
    log["rows_dropped_nan"] = int(nan_rows.sum())
    df = df[~nan_rows].copy()
    df.insert(0, "subject_id", np.int16(subject_id))
    log["rows_clean"] = int(len(df))
    return df, log


def build_dataset(raw_dir: str | Path, hr_method: str = "linear"):
    """Read every subject10X.dat in raw_dir, clean, concatenate in a fixed order."""
    files = sorted(Path(raw_dir).glob("subject1*.dat"))
    if not files:
        raise FileNotFoundError(f"no subject1*.dat files in {raw_dir}")
    parts, logs = [], []
    for f in files:
        sid = int(f.stem.replace("subject", ""))
        clean, log = clean_subject(load_subject(f), sid, hr_method)
        parts.append(clean)
        logs.append(log)
    df = pd.concat(parts, ignore_index=True)    # order: subject, then timestamp
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    return df, {"hr_method": hr_method, "files": [f.name for f in files], "subjects": logs,
                "rows_total": int(len(df)),
                "class_counts": {int(k): int(v) for k, v in
                                 df["activity_id"].value_counts().sort_index().items()}}


def load_processed(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or PROCESSED_DIR / PROCESSED_FILE)


def get_xy(df: pd.DataFrame, feature_set: str, idx=None):
    """Return (X, y) for a named feature set; optional row positions idx."""
    cols = FEATURE_SETS[feature_set]
    assert not FORBIDDEN_FEATURES & set(cols), "leakage: forbidden column in features"
    sub = df if idx is None else df.iloc[idx]
    X = sub[cols].to_numpy(dtype=np.float32)
    y = sub["activity_id"].to_numpy(dtype=np.int64)
    return X, y


def rows_fingerprint(df: pd.DataFrame) -> str:
    """Short hash of WHICH rows survived cleaning (subject, timestamp, label, in order).

    Stored next to the split fingerprint: the split fingerprint alone only depends on the
    number of rows, so two laptops with different cleaned data but the same row count
    would otherwise look identical.
    """
    h = hashlib.sha256()
    h.update(df["subject_id"].to_numpy(dtype=np.int16).tobytes())
    h.update(df["timestamp"].to_numpy(dtype=np.float64).tobytes())
    h.update(df["activity_id"].to_numpy(dtype=np.int16).tobytes())
    return h.hexdigest()[:16]
