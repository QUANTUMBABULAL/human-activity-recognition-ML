"""P2 model + interface tests. OWNER: Person 2.

All data here is synthetic (har/synthetic.py): accuracy checks are smoke checks only and say
nothing about PAMAP2 results.
"""
import importlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from har.config import FEATURE_SETS, LABEL_ORDER
from har.data import get_xy
from har.metrics import REQUIRED_KEYS
from har.models_p2 import decision_tree as dt
from har.models_p2 import random_forest as rf
from har.runner import run_experiment
from har.splits import make_random_split
from har.synthetic import fake_processed_df

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fake():
    df = fake_processed_df(2400)
    return df, make_random_split(len(df))


class _MajorityStub:
    """Minimal .fit/.predict object: the interface run_experiment needs, nothing more."""

    def fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        self.label_ = values[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(len(X), self.label_, dtype=np.int64)


def test_p2_packages_import():
    importlib.import_module("har.models_p2")
    importlib.import_module("experiments.p2")


def test_runner_contract_for_p2_owner(tmp_path):
    df = fake_processed_df(1200)
    split = make_random_split(len(df))
    rec = run_experiment(model_name="p2_stub", owner="P2", model=_MajorityStub(), df=df, split=split,
                         feature_set="full", params={}, results_dir=tmp_path,
                         post_fit=lambda m: {"stub_label": int(m.label_)})
    data = json.loads((tmp_path / "metrics" / "p2_stub_full.json").read_text())
    assert REQUIRED_KEYS <= data.keys() and data["owner"] == "P2"
    assert data["n_train"] == len(split["train"]) and data["n_test"] == len(split["test"])
    assert rec["stub_label"] in LABEL_ORDER                     # original activity IDs, not 0..11
    cm = pd.read_csv(tmp_path / "confusion" / "p2_stub_full.csv", index_col=0)
    assert cm.shape == (12, 12) and cm.to_numpy().sum() == data["n_test"]


def test_p2_experiment_scripts_never_touch_test_rows():
    """Same rule as P1: split['test'] may only be used inside run_experiment."""
    for f in (ROOT / "experiments" / "p2").glob("run_*.py"):
        assert not re.search(r"""split\[['"]test['"]\]""", f.read_text()), f"{f.name} reads test rows"


# ---------------------------------------------------------------- decision tree (E4)

def test_decision_tree_build_matches_paper_defaults():
    m = dt.build()
    assert m.criterion == "gini" and m.max_depth == dt.PAPER["max_depth"] == 15
    assert m.random_state is not None
    assert set(dt.PAPER_TEST_ACC) == {"full", "reduced"}


@pytest.mark.parametrize("fs", ["reduced", "full"])
def test_decision_tree_fit_predict(fs, fake):
    df, s = fake
    X_tr, y_tr = get_xy(df, fs, s["train"])
    X_te, y_te = get_xy(df, fs, s["test"])
    m = dt.build(max_depth=10).fit(X_tr, y_tr)
    assert m.n_features_in_ == len(FEATURE_SETS[fs]) == {"reduced": 11, "full": 31}[fs]
    pred = m.predict(X_te)
    assert pred.shape == y_te.shape
    assert set(np.unique(pred)) <= set(LABEL_ORDER)          # real activity IDs, not 0..11
    assert (pred == y_te).mean() > 3 / len(LABEL_ORDER)      # smoke check on separable fake data
    info = dt.tree_info(m)
    assert 1 <= info["tree_depth"] <= 10 and info["n_leaves"] >= len(LABEL_ORDER)


def test_decision_tree_is_deterministic(fake):
    df, s = fake
    X_tr, y_tr = get_xy(df, "full", s["train"])
    X_te, _ = get_xy(df, "full", s["test"])
    p1 = dt.build(8).fit(X_tr, y_tr).predict(X_te)
    p2 = dt.build(8).fit(X_tr, y_tr).predict(X_te)
    assert (p1 == p2).all()


def test_decision_tree_cv_grid_table(fake):
    df, s = fake
    X, y = get_xy(df, "reduced", s["train"])
    grid = [2, 6, 12]
    rows = dt.cv_grid(X, y, depth_grid=grid, n_jobs=1)
    assert [r["max_depth"] for r in rows] == grid
    assert all({"max_depth", "train_mean", "val_mean", "val_std"} <= r.keys() for r in rows)
    assert all(0 <= r["val_mean"] <= 1 and r["val_std"] >= 0 for r in rows)
    assert dt.select_depth(rows, "one_sd")["max_depth"] in grid


def test_decision_tree_select_depth_rules():
    rows = [{"max_depth": 5, "val_mean": 0.80, "val_std": 0.01},
            {"max_depth": 10, "val_mean": 0.895, "val_std": 0.01},
            {"max_depth": 15, "val_mean": 0.90, "val_std": 0.01},
            {"max_depth": 20, "val_mean": 0.899, "val_std": 0.01}]
    assert dt.select_depth(rows, "max")["max_depth"] == 15
    assert dt.select_depth(rows, "one_sd")["max_depth"] == 10   # shallowest within 1 SD of best
    with pytest.raises(ValueError):
        dt.select_depth(rows, "median")


@pytest.mark.parametrize("fs", ["reduced", "full"])
def test_decision_tree_experiment_script_end_to_end(fs, fake, tmp_path):
    """Script logic on fake data: CV on training rows -> run_experiment -> JSON + confusion CSV."""
    from experiments.p2 import run_decision_tree as script
    df, s = fake
    a = script.parse_args(["--feature-set", fs, "--tune-n", "800", "--n-jobs", "1",
                           "--results-dir", str(tmp_path)])
    a.depth_grid = [3, 8]
    rec = script.run_feature_set(a, df, s, fs)
    data = json.loads((tmp_path / "metrics" / f"decision_tree_{fs}.json").read_text())
    assert REQUIRED_KEYS <= data.keys() and data["owner"] == "P2" and data["feature_set"] == fs
    assert {"test_weighted_f1", "tree_depth", "n_leaves"} <= data.keys()
    assert data["params"]["criterion"] == "gini" and data["params"]["max_depth"] in (3, 8)
    assert data["cv"]["tune_rows"] == 800 and data["cv"]["chosen_max_depth"] in (3, 8)
    assert data["n_train"] == len(s["train"]) and data["n_test"] == len(s["test"])
    assert (tmp_path / "metrics" / f"decision_tree_{fs}_cv.csv").exists()
    cm = pd.read_csv(tmp_path / "confusion" / f"decision_tree_{fs}.csv", index_col=0)
    assert cm.shape == (12, 12) and cm.to_numpy().sum() == data["n_test"]
    assert rec["test_accuracy"] == data["test_accuracy"]


def test_decision_tree_smoke_flag_never_writes_to_results():
    from experiments.p2 import run_decision_tree as script
    a = script.parse_args(["--smoke"])
    assert Path(a.results_dir).resolve() != (ROOT / "results").resolve()
    assert a.tag == "_smoke" and a.fit_n and a.test_n and a.tune_n


# ---------------------------------------------------------------- random forest (E5)

def test_random_forest_build_matches_paper_defaults():
    m = rf.build()
    assert (m.n_estimators, m.max_depth, m.max_features, m.criterion) == (100, 20, "sqrt", "gini")
    assert m.bootstrap and m.random_state is not None
    assert set(rf.PAPER_TEST_ACC) == {"full", "reduced"}


@pytest.mark.parametrize("fs", ["reduced", "full"])
def test_random_forest_fit_predict(fs, fake):
    df, s = fake
    X_tr, y_tr = get_xy(df, fs, s["train"])
    X_te, y_te = get_xy(df, fs, s["test"])
    m = rf.build(n_estimators=15, max_depth=10, n_jobs=1).fit(X_tr, y_tr)
    assert m.n_features_in_ == len(FEATURE_SETS[fs]) == {"reduced": 11, "full": 31}[fs]
    pred = m.predict(X_te)
    assert pred.shape == y_te.shape
    assert set(np.unique(pred)) <= set(LABEL_ORDER)          # real activity IDs, not 0..11
    assert (pred == y_te).mean() > 3 / len(LABEL_ORDER)      # smoke check on separable fake data
    info = rf.forest_info(m)
    assert info["n_trees"] == 15 and 1 <= info["max_tree_depth"] <= 10
    assert info["total_nodes"] >= 15 * len(LABEL_ORDER)


def test_random_forest_is_deterministic_across_n_jobs(fake):
    """Same seed -> identical trees and predictions for any n_jobs. Probabilities may differ by
    float rounding (~1e-16) because threads sum the per-tree averages in a different order."""
    df, s = fake
    X_tr, y_tr = get_xy(df, "full", s["train"])
    X_te, _ = get_xy(df, "full", s["test"])
    m1 = rf.build(10, 8, n_jobs=1).fit(X_tr, y_tr)
    m2 = rf.build(10, 8, n_jobs=2).fit(X_tr, y_tr)
    for t1, t2 in zip(m1.estimators_, m2.estimators_):
        np.testing.assert_array_equal(t1.tree_.feature, t2.tree_.feature)
        np.testing.assert_array_equal(t1.tree_.threshold, t2.tree_.threshold)
    np.testing.assert_array_equal(m1.predict(X_te), m2.predict(X_te))
    np.testing.assert_allclose(m1.predict_proba(X_te), m2.predict_proba(X_te), rtol=0, atol=1e-12)


def test_random_forest_cv_grid_table(fake):
    df, s = fake
    X, y = get_xy(df, "reduced", s["train"])
    rows = rf.cv_grid(X[:800], y[:800], depth_grid=[3, 8], n_estimators_grid=[5, 10], n_jobs=1)
    assert [(r["n_estimators"], r["max_depth"]) for r in rows] == [(5, 3), (5, 8), (10, 3), (10, 8)]
    assert all({"max_depth", "n_estimators", "train_mean", "val_mean", "val_std"} <= r.keys() for r in rows)
    assert all(0 <= r["val_mean"] <= 1 and r["val_std"] >= 0 for r in rows)
    assert rf.select_config(rows) in rows


def test_random_forest_select_config_rules():
    rows = [{"n_estimators": 100, "max_depth": 10, "val_mean": 0.80, "val_std": 0.01},
            {"n_estimators": 100, "max_depth": 20, "val_mean": 0.895, "val_std": 0.01},
            {"n_estimators": 200, "max_depth": 20, "val_mean": 0.90, "val_std": 0.01},
            {"n_estimators": 200, "max_depth": 30, "val_mean": 0.899, "val_std": 0.01}]
    assert rf.select_config(rows, "max") == rows[2]
    assert rf.select_config(rows, "one_sd") == rows[1]          # fewest trees, then shallowest
    with pytest.raises(ValueError):
        rf.select_config(rows, "median")


@pytest.mark.parametrize("fs", ["reduced", "full"])
def test_random_forest_experiment_script_end_to_end(fs, fake, tmp_path):
    """Script logic on fake data: CV on training rows -> run_experiment -> JSON + confusion CSV."""
    from experiments.p2 import run_random_forest as script
    df, s = fake
    a = script.parse_args(["--feature-set", fs, "--tune-n", "800", "--depth-grid", "3", "8",
                           "--trees-grid", "10", "--n-jobs", "1", "--results-dir", str(tmp_path)])
    rec = script.run_feature_set(a, df, s, fs)
    data = json.loads((tmp_path / "metrics" / f"random_forest_{fs}.json").read_text())
    assert REQUIRED_KEYS <= data.keys() and data["owner"] == "P2" and data["feature_set"] == fs
    assert {"test_weighted_f1", "n_trees", "mean_tree_depth", "total_nodes"} <= data.keys()
    assert data["params"]["max_features"] == "sqrt" and data["params"]["n_estimators"] == 10
    assert data["params"]["max_depth"] in (3, 8) and data["cv"]["tune_rows"] == 800
    assert data["n_train"] == len(s["train"]) and data["n_test"] == len(s["test"])
    assert (tmp_path / "metrics" / f"random_forest_{fs}_cv.csv").exists()
    cm = pd.read_csv(tmp_path / "confusion" / f"random_forest_{fs}.csv", index_col=0)
    assert cm.shape == (12, 12) and cm.to_numpy().sum() == data["n_test"]
    assert rec["test_accuracy"] == data["test_accuracy"]


def test_random_forest_smoke_flag_never_writes_to_results():
    from experiments.p2 import run_random_forest as script
    a = script.parse_args(["--smoke"])
    assert Path(a.results_dir).resolve() != (ROOT / "results").resolve()
    assert a.tag == "_smoke" and a.fit_n and a.test_n and a.tune_n
