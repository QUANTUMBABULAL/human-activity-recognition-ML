"""Support vector machine. OWNER: Person 1.

Paper: grid search over kernels {linear, poly, rbf} x C with 5-fold CV; rbf best;
chosen C = 100 (full) and 1000 (reduced); standardised inputs.
Authors' code: SVC(kernel="rbf", gamma="auto"), training capped at 175,000 rows.
Kernel SVMs scale roughly quadratically with rows -> we tune and fit on subsamples.
"""
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from har.config import N_FOLDS, SEED

PAPER = {"full": {"kernel": "rbf", "C": 100.0}, "reduced": {"kernel": "rbf", "C": 1000.0}}
PAPER_TEST_ACC = {"full": 0.989, "reduced": 0.950}
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]   # range used in the authors' code
KERNEL_C_GRID = [0.1, 1.0, 10.0, 100.0]          # Stage B (kernel comparison)


def build(kernel: str = "rbf", C: float = 1.0, degree: int = 3,
          cache_size_mb: int = 2000, max_iter: int = -1) -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("clf", SVC(kernel=kernel, C=C, gamma="auto", degree=degree,
                                 cache_size=cache_size_mb, max_iter=max_iter,
                                 random_state=SEED))])


def n_support_vectors(model: Pipeline) -> int:
    return int(model.named_steps["clf"].n_support_.sum())


def cv_grid(X, y, kernels=("rbf",), C_grid=C_GRID, n_jobs: int = -1, max_iter: int = -1,
            cache_mb: int = 500):
    """Mean/std 5-fold CV accuracy for every (kernel, C). Training rows only.

    cache_mb is per worker process: with n_jobs=5 and 2000 MB each you could need 10 GB.
    """
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for k in kernels:
        for C in C_grid:
            s = cross_validate(build(k, C, max_iter=max_iter, cache_size_mb=cache_mb), X, y,
                               cv=cv, scoring="accuracy", n_jobs=n_jobs,
                               return_train_score=True)
            rows.append({"kernel": k, "C": C,
                         "train_mean": float(s["train_score"].mean()),
                         "val_mean": float(s["test_score"].mean()),
                         "val_std": float(s["test_score"].std())})
            print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r["val_mean"])
    return best, rows
