"""Fake PAMAP2-shaped data for tests and for P2 on Day 1. OWNER: Person 1.

NEVER report numbers produced from this data. It exists only so code can be tested
before (or without) the real files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from har.config import (FEATURE_SETS, IMU_FIELDS, IMU_LOCATIONS, LABEL_ORDER, RAW_COLUMNS,
                        SEED)


def _class_labels(n_rows: int, rng, segment: int | None = None) -> np.ndarray:
    """Activities come in contiguous segments (like real recordings); all 12 appear."""
    seg = segment or max(1, min(50, n_rows // (12 * 3)))
    n_seg = int(np.ceil(n_rows / seg))
    order = np.concatenate([rng.permutation(LABEL_ORDER) for _ in range(int(np.ceil(n_seg / 12)))])
    return np.repeat(order[:n_seg], seg)[:n_rows]


def _class_means(rng, n_cols: int) -> np.ndarray:
    # +1 row so index 0 (the transient label) also has a mean
    return rng.normal(0, 3.0, size=(max(LABEL_ORDER) + 1, n_cols))


def fake_processed_df(n_rows: int = 5000, seed: int = SEED, n_subjects: int = 3) -> pd.DataFrame:
    """DataFrame in the *processed* format: row_id, subject_id, timestamp, activity_id + 31 features."""
    rng = np.random.default_rng(seed)
    feats = FEATURE_SETS["full"]
    y = _class_labels(n_rows, rng)
    means = _class_means(rng, len(feats))
    X = means[y] + rng.normal(0, 1.0, size=(n_rows, len(feats)))
    df = pd.DataFrame(X.astype("float32"), columns=feats)
    df.insert(0, "activity_id", y.astype("int16"))
    df.insert(0, "timestamp", (np.arange(n_rows) % max(1, n_rows // n_subjects)) * 0.01)
    df.insert(0, "subject_id", (np.arange(n_rows) * n_subjects // n_rows + 101).astype("int16"))
    df.insert(0, "row_id", np.arange(n_rows, dtype=np.int64))
    return df


def fake_raw_subject(n_rows: int = 3000, seed: int = SEED, with_nans: bool = True) -> pd.DataFrame:
    """One fake subject in the RAW 54-column layout (label 0 stretches, ~9 Hz heart rate, dropouts)."""
    rng = np.random.default_rng(seed)
    y = _class_labels(n_rows, rng)
    # insert transient (label 0) stretches
    y = y.copy()
    for start in range(0, n_rows, 400):
        y[start:start + 40] = 0
    means = _class_means(rng, len(RAW_COLUMNS))
    X = means[y] + rng.normal(0, 1.0, size=(n_rows, len(RAW_COLUMNS)))
    df = pd.DataFrame(X, columns=RAW_COLUMNS)
    df["timestamp"] = 8.38 + 0.01 * np.arange(n_rows)
    df["activity_id"] = y.astype("int64")
    hr = np.full(n_rows, np.nan)
    hr[5::11] = 70 + 3 * y[5::11] % 60 + rng.normal(0, 1, size=len(hr[5::11]))  # ~9 Hz
    df["heart_rate"] = hr
    if with_nans:   # wireless dropouts in a few IMU cells
        for col in ("hand_acc16_x", "chest_mag_y", "ankle_gyro_z"):
            bad = rng.choice(n_rows, size=max(1, n_rows // 500), replace=False)
            df.loc[bad, col] = np.nan
    return df


def write_dat(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a raw-format .dat file: whitespace separated, no header, 'NaN' for missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[RAW_COLUMNS].copy()
    out.to_csv(path, sep=" ", header=False, index=False, na_rep="NaN", float_format="%.5f")
    return path


def write_fake_dat_files(directory: str | Path, n_subjects: int = 3, rows: int = 3000) -> list[Path]:
    """Write subject101.dat ... into `directory` (raw format) and return the paths."""
    paths = []
    for i in range(n_subjects):
        sid = 101 + i
        paths.append(write_dat(fake_raw_subject(rows, seed=SEED + i), Path(directory) / f"subject{sid}.dat"))
    return paths
