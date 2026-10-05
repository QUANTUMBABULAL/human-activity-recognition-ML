"""P2 interface tests. OWNER: Person 2.

Structure/contract checks only: no P2 model is implemented yet, and nothing here measures
model quality. Model tests (decision tree, random forest, AdaBoost, MLP) are added with the models.
"""
import importlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from har.config import LABEL_ORDER
from har.metrics import REQUIRED_KEYS
from har.runner import run_experiment
from har.splits import make_random_split
from har.synthetic import fake_processed_df

ROOT = Path(__file__).resolve().parents[1]


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
