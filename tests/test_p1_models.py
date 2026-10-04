"""LR + SVM model tests. OWNER: Person 1."""
import numpy as np
import pytest

from har.config import LABEL_ORDER
from har.data import get_xy
from har.models_p1 import logreg, svm
from har.splits import make_random_split
from har.synthetic import fake_processed_df


@pytest.fixture(scope="module")
def data():
    df = fake_processed_df(2400)
    s = make_random_split(len(df))
    return get_xy(df, "reduced", s["train"]), get_xy(df, "reduced", s["test"])


@pytest.mark.parametrize("name", ["logreg", "svm"])
def test_p1_model_output(name, data):
    (X_tr, y_tr), (X_te, y_te) = data
    model = logreg.build(C=1.0, max_iter=300) if name == "logreg" else svm.build("rbf", C=10.0)
    pred = model.fit(X_tr, y_tr).predict(X_te)
    assert pred.shape == y_te.shape
    assert set(np.unique(pred)) <= set(LABEL_ORDER)          # real activity IDs, not 0..11
    assert (pred == y_te).mean() > 3 / len(LABEL_ORDER)      # far better than chance (1/12)


def test_tune_C_returns_grid_value_and_table(data):
    (X_tr, y_tr), _ = data
    grid = [1e-3, 1e-1, 10.0]
    best, table = logreg.tune_C(X_tr, y_tr, grid=grid, max_iter=100, n_jobs=1)
    assert best in grid and [r["C"] for r in table] == grid
    assert all(0 <= r["val_mean"] <= 1 and r["val_std"] >= 0 for r in table)


def test_svm_cv_grid_and_support_vectors(data):
    (X_tr, y_tr), _ = data
    best, rows = svm.cv_grid(X_tr[:600], y_tr[:600], kernels=("linear", "rbf"), C_grid=[1.0, 10.0], n_jobs=1)
    assert len(rows) == 4 and best in rows
    assert {"kernel", "C", "train_mean", "val_mean", "val_std"} <= rows[0].keys()
    m = svm.build("rbf", 10.0).fit(X_tr[:600], y_tr[:600])
    assert 0 < svm.n_support_vectors(m) <= 600
