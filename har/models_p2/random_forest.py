"""Random forest. OWNER: Person 2.

Paper: 100 trees, sqrt(features) considered per split, max depth 20, both feature sets.
We tune max_depth (and optionally the number of trees) by 5-fold CV on training rows.
Trees are scale-invariant, so no scaler (D-09).

Memory: a fitted tree on ~1.6M rows is several MB (probe: ~2-6 MB per tree at 400k rows),
so CV folds run one after another and only the trees inside a forest run in parallel.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_validate

from har.config import N_FOLDS, SEED

PAPER = {"n_estimators": 100, "max_features": "sqrt", "max_depth": 20, "criterion": "gini"}
PAPER_TEST_ACC = {"full": 0.980, "reduced": 0.937}
DEPTH_GRID = [10, 15, 20, 25, 30]
N_ESTIMATORS_GRID = [100]
RULES = ("max", "one_sd")


def build(n_estimators: int = PAPER["n_estimators"], max_depth: int | None = PAPER["max_depth"],
          max_features=PAPER["max_features"], criterion: str = PAPER["criterion"],
          n_jobs: int = -1) -> RandomForestClassifier:
    # random_state fixes bootstrap samples and feature draws: same trees and predictions for any
    # n_jobs (predict_proba can differ by ~1e-16 from the threaded summation order).
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  max_features=max_features, criterion=criterion,
                                  bootstrap=True, n_jobs=n_jobs, random_state=SEED)


def forest_info(model: RandomForestClassifier) -> dict:
    """Size of the fitted forest (merged into the result JSON via run_experiment(post_fit=...))."""
    depths = [e.get_depth() for e in model.estimators_]
    nodes = [e.tree_.node_count for e in model.estimators_]
    return {"n_trees": len(model.estimators_), "mean_tree_depth": float(np.mean(depths)),
            "max_tree_depth": int(max(depths)), "mean_tree_nodes": float(np.mean(nodes)),
            "total_nodes": int(sum(nodes))}


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


def cv_grid(X, y, depth_grid=DEPTH_GRID, n_estimators_grid=N_ESTIMATORS_GRID,
            max_features=PAPER["max_features"], n_jobs: int = -1):
    """Mean/std 5-fold CV accuracy (train + validation) per (n_estimators, max_depth). Training rows only.

    Folds run sequentially; n_jobs parallelises the trees inside each forest.
    """
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for n_est in n_estimators_grid:
        for d in depth_grid:
            s = cross_validate(build(n_est, d, max_features, n_jobs=n_jobs), X, y, cv=cv,
                               scoring="accuracy", n_jobs=1, return_train_score=True)
            rows.append({"n_estimators": n_est, "max_depth": d,
                         "train_mean": float(s["train_score"].mean()),
                         "val_mean": float(s["test_score"].mean()),
                         "val_std": float(s["test_score"].std()),
                         "fit_seconds_mean": float(s["fit_time"].mean())})
            print(rows[-1], flush=True)
    return rows
