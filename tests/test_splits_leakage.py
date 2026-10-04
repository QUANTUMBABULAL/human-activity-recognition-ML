"""Split + leakage tests. OWNER: Person 1."""
import re
from pathlib import Path

import numpy as np

from har.data import get_xy
from har.models_p1 import logreg
from har.splits import (load_split, make_random_split, make_subject_split, save_split,
                        split_fingerprint)
from har.synthetic import fake_processed_df


def test_split_disjoint_complete_deterministic():
    s1, s2 = make_random_split(10_000), make_random_split(10_000)
    assert split_fingerprint(s1) == split_fingerprint(s2)
    assert len(np.intersect1d(s1["train"], s1["test"])) == 0
    assert len(s1["train"]) + len(s1["test"]) == 10_000
    assert len(s1["test"]) == 1500
    assert split_fingerprint(make_random_split(10_000, seed=1)) != split_fingerprint(s1)


def test_split_save_load_roundtrip(tmp_path):
    s = make_random_split(1000)
    p = save_split(s, tmp_path / "s.npz")
    s2 = load_split(p)
    assert split_fingerprint(s) == split_fingerprint(s2)


def test_subject_split_holds_out_whole_subjects():
    df = fake_processed_df(900, n_subjects=3)
    s = make_subject_split(df, [102])
    assert set(df.iloc[s["test"]]["subject_id"]) == {102}
    assert 102 not in set(df.iloc[s["train"]]["subject_id"])


def test_scaler_is_fit_on_training_rows_only():
    df = fake_processed_df(2000)
    s = make_random_split(len(df))
    X_tr, y_tr = get_xy(df, "full", s["train"])
    X_te, _ = get_xy(df, "full", s["test"])
    X_te = X_te + 1000.0                                       # poison test rows
    model = logreg.build(C=1.0, max_iter=200).fit(X_tr, y_tr)
    np.testing.assert_allclose(model.named_steps["scale"].mean_, X_tr.mean(axis=0), rtol=1e-4)
    model.predict(X_te)                                        # predicting must not refit
    np.testing.assert_allclose(model.named_steps["scale"].mean_, X_tr.mean(axis=0), rtol=1e-4)


def test_p1_experiment_scripts_never_touch_test_rows():
    """Review check automated: split['test'] may only be used inside run_experiment."""
    root = Path(__file__).resolve().parents[1] / "experiments" / "p1"
    for f in root.glob("run_*.py"):
        assert not re.search(r"""split\[['"]test['"]\]""", f.read_text()), f"{f.name} reads test rows"
