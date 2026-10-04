"""Train/test split. OWNER: Person 1.

Interface:
    make_random_split(n_rows, seed=SEED, test_fraction=TEST_FRACTION) -> dict(train=, test=)
    save_split(split, path) / load_split(path=None) -> dict(train=np.ndarray, test=np.ndarray)
    split_fingerprint(split) -> short sha256 string (both people must get the SAME value)
    make_subject_split(df, test_subjects) -> dict   [optional extension only]
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from har.config import PROCESSED_DIR, SEED, TEST_FRACTION

SPLIT_FILE = f"split_random_seed{SEED}.npz"


def make_random_split(n_rows: int, seed: int = SEED, test_fraction: float = TEST_FRACTION):
    """Report protocol: pool all subjects, random (not stratified) row-level 85/15 split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_test = int(round(test_fraction * n_rows))
    return {"train": np.sort(perm[n_test:]), "test": np.sort(perm[:n_test])}


def make_subject_split(df, test_subjects):
    """OPTIONAL extension: whole subjects held out. Never replaces the main protocol."""
    is_test = df["subject_id"].isin(test_subjects).to_numpy()
    return {"train": np.flatnonzero(~is_test), "test": np.flatnonzero(is_test)}


def split_fingerprint(split) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(split["train"], dtype=np.int64).tobytes())
    h.update(np.asarray(split["test"], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def save_split(split, path: str | Path | None = None):
    path = Path(path or PROCESSED_DIR / SPLIT_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train=split["train"], test=split["test"])
    return path


def load_split(path: str | Path | None = None):
    with np.load(path or PROCESSED_DIR / SPLIT_FILE) as z:
        return {"train": z["train"], "test": z["test"]}


def write_split_meta(split, n_rows: int, path: str | Path, extra: dict | None = None):
    meta = {"seed": SEED, "test_fraction": TEST_FRACTION, "n_rows": n_rows,
            "n_train": int(len(split["train"])), "n_test": int(len(split["test"])),
            "fingerprint": split_fingerprint(split)}
    if extra:
        meta.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(meta, indent=2))
    return meta
