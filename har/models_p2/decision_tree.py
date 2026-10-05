"""Decision tree. OWNER: Person 2.

Paper: Gini impurity, max depth chosen by 5-fold CV with the one-standard-deviation rule ->
max depth 15 for both feature sets. Trees are scale-invariant, so no scaler (D-09).
"""
from sklearn.model_selection import KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from har.config import N_FOLDS, SEED

PAPER = {"criterion": "gini", "max_depth": 15}
PAPER_TEST_ACC = {"full": 0.927, "reduced": 0.873}
DEPTH_GRID = [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30]
RULES = ("one_sd", "max")


def build(max_depth: int | None = PAPER["max_depth"], criterion: str = PAPER["criterion"],
          min_samples_leaf: int = 1) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf, random_state=SEED)


def tree_info(model: DecisionTreeClassifier) -> dict:
    """Size of the fitted tree (merged into the result JSON via run_experiment(post_fit=...))."""
    return {"tree_depth": int(model.get_depth()), "n_leaves": int(model.get_n_leaves())}


def select_depth(rows, rule: str = "one_sd") -> dict:
    """Pick a row of the CV table.

    "max":    highest mean validation accuracy.
    "one_sd": the shallowest depth whose mean validation accuracy is within one standard
              deviation of the best (the paper's rule; favours the simpler tree).
    """
    best = max(rows, key=lambda r: r["val_mean"])
    if rule == "max":
        return best
    if rule != "one_sd":
        raise ValueError(f"unknown rule {rule!r}")
    ok = [r for r in rows if r["val_mean"] >= best["val_mean"] - best["val_std"]]
    return min(ok, key=lambda r: r["max_depth"])


def cv_grid(X, y, depth_grid=DEPTH_GRID, criterion: str = PAPER["criterion"], n_jobs: int = -1):
    """Mean/std 5-fold CV accuracy (train + validation) for every max_depth. Training rows only."""
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for d in depth_grid:
        s = cross_validate(build(d, criterion), X, y, cv=cv, scoring="accuracy", n_jobs=n_jobs,
                           return_train_score=True)
        rows.append({"max_depth": d,
                     "train_mean": float(s["train_score"].mean()),
                     "val_mean": float(s["test_score"].mean()),
                     "val_std": float(s["test_score"].std())})
        print(rows[-1], flush=True)
    return rows
