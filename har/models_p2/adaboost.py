"""AdaBoost with decision-tree base learners. OWNER: Person 2.

Paper: default learning rate; chosen 500 trees / depth 10 (reduced), 250 trees / depth 9 (full).
We tune (max_depth, n_estimators) by 5-fold CV on training rows. Trees are scale-invariant, so
no scaler (D-09).

sklearn API (checked on 1.9.1): the base learner goes in `estimator=` (`base_estimator` was
removed in 1.4) and only discrete SAMME exists (`algorithm=` / SAMME.R were removed in 1.6; the
2018 report's sklearn defaulted to SAMME.R). `random_state` seeds every base tree in turn.
AdaBoost has no `n_jobs`: the trees are fitted one after another.

Compute: the first k trees of an n-tree fit are exactly the trees of a k-tree fit, so CV fits the
largest tree count once per (depth, fold) and scores the smaller counts with `staged_predict`.
Folds run in parallel (n_jobs). Probe on real training rows (one weighted tree, single thread):
reduced/depth 10 ~2.6 s at 200k rows, ~22.5 s at 1.63M; full/depth 9 ~6.8 s / ~59 s.
"""
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from har.config import N_FOLDS, SEED

LEARNING_RATE = 1.0   # sklearn default ("default learning rate" in the report)
PAPER = {"reduced": {"n_estimators": 500, "max_depth": 10, "learning_rate": LEARNING_RATE},
         "full": {"n_estimators": 250, "max_depth": 9, "learning_rate": LEARNING_RATE}}
PAPER_TEST_ACC = {"full": 0.985, "reduced": 0.940}
DEPTH_GRID = [6, 8, 9, 10, 12]
N_ESTIMATORS_GRID = [50, 100, 250, 500]
RULES = ("max", "one_sd")


def build(n_estimators: int, max_depth: int | None, learning_rate: float = LEARNING_RATE,
          criterion: str = "gini") -> AdaBoostClassifier:
    """AdaBoost-SAMME over explicit Gini decision trees of depth `max_depth`."""
    tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=SEED)
    return AdaBoostClassifier(estimator=tree, n_estimators=n_estimators,
                              learning_rate=learning_rate, random_state=SEED)


def build_paper(feature_set: str) -> AdaBoostClassifier:
    return build(**PAPER[feature_set])


def boost_info(model: AdaBoostClassifier) -> dict:
    """Size of the fitted ensemble (merged into the result JSON via run_experiment(post_fit=...)).

    sklearn stops early if a tree reaches zero weighted error, so n_trees_fitted can be below
    n_estimators; that is recorded, not hidden.
    """
    depths = [e.get_depth() for e in model.estimators_]
    return {"n_trees_fitted": len(model.estimators_),
            "stopped_early": len(model.estimators_) < model.n_estimators,
            "mean_tree_depth": float(np.mean(depths)), "max_tree_depth": int(max(depths)),
            "total_nodes": int(sum(e.tree_.node_count for e in model.estimators_)),
            "last_estimator_error": float(model.estimator_errors_[len(model.estimators_) - 1])}


def staged_accuracy(model: AdaBoostClassifier, X, y, checkpoints) -> dict:
    """Accuracy after each tree count in `checkpoints`. If boosting stopped early, larger counts
    get the final ensemble's accuracy (what a fresh fit with that count would also produce)."""
    want, out, pred = sorted(set(checkpoints)), {}, None
    for k, pred in enumerate(model.staged_predict(X), start=1):
        if k in want:
            out[k] = float((pred == y).mean())
    last = float((pred == y).mean())
    return {k: out.get(k, last) for k in want}


def _fold(X, y, tr, va, max_depth, checkpoints, learning_rate):
    m = build(max(checkpoints), max_depth, learning_rate)
    t0 = time.perf_counter()
    m.fit(X[tr], y[tr])
    fit_s = time.perf_counter() - t0
    return (staged_accuracy(m, X[tr], y[tr], checkpoints),
            staged_accuracy(m, X[va], y[va], checkpoints), fit_s, len(m.estimators_))


def cv_grid(X, y, depth_grid=DEPTH_GRID, n_estimators_grid=N_ESTIMATORS_GRID,
            learning_rate: float = LEARNING_RATE, n_jobs: int = -1):
    """Mean/std 5-fold CV accuracy (train + validation) per (n_estimators, max_depth). Training rows only.

    One max(n_estimators_grid)-tree fit per (depth, fold); smaller counts are read off with
    staged_predict. Folds run in parallel (n_jobs); fit_seconds_mean is for the largest count.
    """
    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X))
    grid = sorted(set(n_estimators_grid))
    rows = []
    for d in depth_grid:
        res = Parallel(n_jobs=n_jobs)(delayed(_fold)(X, y, tr, va, d, grid, learning_rate)
                                      for tr, va in folds)
        for n in grid:
            val = np.array([r[1][n] for r in res])
            rows.append({"n_estimators": n, "max_depth": d,
                         "train_mean": float(np.mean([r[0][n] for r in res])),
                         "val_mean": float(val.mean()), "val_std": float(val.std()),
                         "fit_seconds_mean": float(np.mean([r[2] for r in res])),
                         "min_trees_fitted": int(min(r[3] for r in res))})
            print(rows[-1], flush=True)
    return rows


def select_config(rows, rule: str = "max") -> dict:
    """Pick a row of the CV table.

    "max":    highest mean validation accuracy.
    "one_sd": the cheapest config (fewest trees, then shallowest) within one standard
              deviation of the best mean validation accuracy.
    """
    best = max(rows, key=lambda r: r["val_mean"])
    if rule == "max":
        return best
    if rule != "one_sd":
        raise ValueError(f"unknown rule {rule!r}")
    ok = [r for r in rows if r["val_mean"] >= best["val_mean"] - best["val_std"]]
    return min(ok, key=lambda r: (r["n_estimators"], r["max_depth"]))
