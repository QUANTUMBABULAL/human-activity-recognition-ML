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
