"""Logistic regression. OWNER: Person 1.

Paper: L2-regularised logistic regression, SAG solver, C tuned by 5-fold CV, chosen C ~ 0.01
for both feature sets; features standardised (SAG needs similar feature scales).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from har.config import N_FOLDS, SEED

PAPER = {"C": 0.01, "solver": "sag", "penalty": "l2"}
C_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]   # range used in the authors' code
PAPER_TEST_ACC = {"full": 0.815, "reduced": 0.639}


def build(C: float = PAPER["C"], max_iter: int = 1000) -> Pipeline:
    # L2 is LogisticRegression's default penalty; recent scikit-learn deprecates the
    # `penalty=` argument, so we rely on the default instead of passing it.
    return Pipeline([("scale", StandardScaler()),   # fit on training rows only
                     ("clf", LogisticRegression(C=C, solver="sag", max_iter=max_iter,
                                                random_state=SEED))])


def tune_C(X, y, grid=C_GRID, max_iter: int = 300, n_jobs: int = -1):
    """5-fold CV on TRAINING data only. Report rule: pick C with highest mean val. accuracy."""
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr, va = validation_curve(build(max_iter=max_iter), X, y, param_name="clf__C",
                              param_range=grid, cv=cv, scoring="accuracy", n_jobs=n_jobs)
    table = [{"C": c, "train_mean": float(t.mean()), "val_mean": float(v.mean()),
              "val_std": float(v.std())} for c, t, v in zip(grid, tr, va)]
    best = grid[int(np.argmax(va.mean(axis=1)))]
    return best, table
