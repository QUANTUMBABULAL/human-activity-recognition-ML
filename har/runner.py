"""Shared fit -> predict -> time -> save runner. OWNER: Person 1 (kickoff, then frozen).

Any model object with .fit(X, y) and .predict(X) works (sklearn estimators/Pipelines,
or Person 2's KerasMLP wrapper).
"""
from __future__ import annotations

import time

import numpy as np

from har.config import SEED
from har.data import get_xy
from har.metrics import evaluate, save_result


def subsample(idx: np.ndarray, n: int | None, seed: int = SEED) -> np.ndarray:
    """Deterministic random subset of row positions (returns idx unchanged if n is None)."""
    idx = np.asarray(idx)
    if n is None or n >= len(idx):
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=n, replace=False))


def run_experiment(*, model_name, owner, model, df, split, feature_set, params,
                   paper_test_accuracy=None, notes="", fit_n=None, test_n=None,
                   train_eval_n=100_000, cv_info=None, protocol="random_85_15", tag="",
                   results_dir=None, post_fit=None):
    """Fit on the training split (optionally a subsample), evaluate on the test split.

    post_fit: optional callable(model) -> dict, evaluated right after fit; its items are
    merged into the JSON record (e.g. the number of SVM support vectors).
    """
    tr = subsample(split["train"], fit_n)
    te = subsample(split["test"], test_n, seed=SEED + 1)
    X_tr, y_tr = get_xy(df, feature_set, tr)
    X_te, y_te = get_xy(df, feature_set, te)

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    extra = post_fit(model) if post_fit else {}

    t0 = time.perf_counter()
    y_pred = model.predict(X_te)
    pred_s = time.perf_counter() - t0

    tr_eval = subsample(np.arange(len(y_tr)), train_eval_n, seed=SEED + 2)
    train_acc = evaluate(y_tr[tr_eval], model.predict(X_tr[tr_eval]))["accuracy"]
    test_m = evaluate(y_te, y_pred)

    record = {"model": model_name, "owner": owner, "feature_set": feature_set,
              "protocol": protocol, "seed": SEED,
              "n_train": int(len(tr)), "n_test": int(len(te)),
              "train_accuracy": train_acc, "train_accuracy_rows": int(len(tr_eval)),
              "test_accuracy": test_m["accuracy"], "test_macro_f1": test_m["macro_f1"],
              "test_weighted_f1": test_m["weighted_f1"], "params": params,
              "cv": cv_info or {}, "fit_seconds": round(fit_s, 2),
              "predict_seconds": round(pred_s, 2),
              "paper_test_accuracy": paper_test_accuracy, "notes": notes, **extra}
    path = save_result(record, y_te, y_pred, tag=tag,
                       **({"results_dir": results_dir} if results_dir else {}))
    print(f"[{model_name}/{feature_set}] test acc={test_m['accuracy']:.4f} "
          f"macroF1={test_m['macro_f1']:.4f} fit={fit_s:.1f}s -> {path}")
    return record
