"""Tests for the --quick experiment mode (P2-21) and the QUICK status in compare_results.

Synthetic data only (har.synthetic), tiny fits, results in temp directories. The CV helpers are
patched to raise, so a quick run that tried to tune would fail. No accuracy claims about PAMAP2.
"""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments import compare_results as cr
from experiments import quick_mode
from experiments.p1 import run_svm
from experiments.p2 import run_adaboost, run_mlp, run_random_forest
from har.config import RESULTS_DIR
from har.metrics import REQUIRED_KEYS, save_result
from har.splits import make_random_split
from har.synthetic import fake_processed_df

needs_tf = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None,
                              reason="TensorFlow not installed (MLP is P2-only)")


@pytest.fixture(scope="module")
def fake():
    df = fake_processed_df(2400)
    return df, make_random_split(len(df))


@pytest.fixture
def no_cv(monkeypatch):
    """Any CV / tuning call during a quick run is a failure."""
    def forbidden(*_a, **_k):
        raise AssertionError("quick mode must not run cross-validation")
    for mod in (run_random_forest.rf, run_adaboost.ab, run_svm.svm):
        monkeypatch.setattr(mod, "cv_grid", forbidden)
    monkeypatch.setattr(run_svm, "get_xy", forbidden)   # run_svm only calls get_xy for CV


def _check_quick_json(path, fs="reduced"):
    data = json.loads(Path(path).read_text())
    assert REQUIRED_KEYS <= data.keys()
    assert data["status"] == "QUICK" and data["experiment_mode"] == "quick"
    assert data["notes"].startswith("QUICK") and "NOT a paper reproduction" in data["notes"]
    assert data["hyperparameter_tuning"].startswith("skipped")
    assert data["cv"]["performed"] is False
    assert data["feature_set"] == data["quick_config"]["feature_set"] == fs
    assert data["fit_seconds"] >= 0 and 0 <= data["test_accuracy"] <= 1
    assert {"test_macro_f1", "test_weighted_f1"} <= data.keys()
    return data


# ---------------------------------------------------------------- CLI presets

def test_normal_mode_defaults_unchanged():
    rf = run_random_forest.parse_args([])
    assert (rf.quick, rf.feature_set, rf.tag, rf.n_jobs, rf.tune_n) == (False, "both", "", -1, 200_000)
    assert rf.depth_grid == [10, 15, 20, 25, 30] and rf.trees_grid == [100]
    ab = run_adaboost.parse_args([])
    assert (ab.quick, ab.feature_set, ab.tag, ab.tune_n, ab.learning_rate) == (False, "both", "", 100_000, 1.0)
    assert ab.trees_grid == [50, 100, 250, 500]
    mlp = run_mlp.parse_args([])
    assert (mlp.quick, mlp.feature_set, mlp.tag) == (False, "both", "")
    assert (mlp.epochs, mlp.batch_size, mlp.learning_rate, mlp.val_frac) == (100, 32, 0.01, 0.1)
    svm = run_svm.parse_args([])
    assert (svm.quick, svm.feature_set, svm.stage, svm.fit_n, svm.tune_n) == (False, "both", "A", 50_000, 20_000)
    for a in (rf, ab, mlp):
        assert Path(a.results_dir).resolve() == RESULTS_DIR.resolve()


def test_quick_presets_select_cheap_hyperparameters():
    rf = run_random_forest.parse_args(["--quick"])
    assert (rf.feature_set, rf.tag, rf.n_jobs) == ("reduced", "_quick", 2)
    assert run_random_forest.QUICK == {"feature_set": "reduced", "n_estimators": 20, "max_depth": 10, "n_jobs": 2}
    assert run_random_forest.parse_args(["--quick", "--n-jobs", "1"]).n_jobs == 1   # n_jobs never changes trees
    ab = run_adaboost.parse_args(["--quick"])
    assert (ab.feature_set, ab.tag) == ("reduced", "_quick")
    assert run_adaboost.QUICK == {"feature_set": "reduced", "n_estimators": 20, "max_depth": 6, "learning_rate": 1.0}
    mlp = run_mlp.parse_args(["--quick"])
    assert (mlp.feature_set, mlp.tag, mlp.epochs, mlp.batch_size) == ("reduced", "_quick", 5, 256)
    assert (mlp.learning_rate, mlp.val_frac, mlp.patience, mlp.fit_n) == (0.08, 0.1, None, None)
    svm = run_svm.parse_args(["--quick"])
    assert (svm.feature_set, svm.stage) == ("reduced", "A")
    assert 20_000 <= run_svm.QUICK["fit_n"] <= 50_000
    for a in (rf, ab, mlp, svm):          # quick results are real (educational) results, not temp files
        assert a.quick and Path(a.results_dir).resolve() == RESULTS_DIR.resolve()


def test_quick_feature_set_can_be_chosen_explicitly():
    assert run_random_forest.parse_args(["--quick", "--feature-set", "full"]).feature_set == "full"
    assert run_mlp.parse_args(["--quick", "--feature-set", "both"]).feature_set == "both"


@pytest.mark.parametrize("script, argv", [
    (run_random_forest, ["--quick", "--max-depth", "20"]),
    (run_random_forest, ["--quick", "--trees-grid", "50", "100"]),
    (run_random_forest, ["--quick", "--smoke"]),
    (run_adaboost, ["--quick", "--n-estimators", "500"]),
    (run_adaboost, ["--quick", "--skip-tune"]),
    (run_mlp, ["--quick", "--epochs", "100"]),
    (run_mlp, ["--quick", "--batch-size", "32"]),
    (run_svm, ["--quick", "--stage", "B"]),
    (run_svm, ["--quick", "--C", "10"]),
    (run_svm, ["--quick", "--fit-n", "100000"]),
])
def test_quick_refuses_flags_that_would_change_the_preset(script, argv):
    with pytest.raises(SystemExit):
        script.parse_args(argv)


# ---------------------------------------------------------------- quick runs on synthetic data

def test_random_forest_quick_run(fake, tmp_path, no_cv):
    df, s = fake
    a = run_random_forest.parse_args(["--quick", "--results-dir", str(tmp_path)])
    rec = run_random_forest.run_feature_set(a, df, s, "reduced")
    data = _check_quick_json(tmp_path / "metrics" / "random_forest_reduced_quick.json")
    assert data["params"]["n_estimators"] == data["n_trees"] == 20
    assert data["params"]["max_depth"] == 10 and data["max_tree_depth"] <= 10
    q = data["quick_config"]
    assert (q["n_estimators"], q["max_depth"], q["n_jobs"]) == (20, 10, 2)
    assert data["n_train"] == len(s["train"]) and data["n_test"] == len(s["test"])
    assert not (tmp_path / "metrics" / "random_forest_reduced.json").exists()   # never the FULL file
    assert not list((tmp_path / "metrics").glob("*_cv.csv"))
    assert (tmp_path / "confusion" / "random_forest_reduced_quick.csv").exists()
    assert rec["test_accuracy"] == data["test_accuracy"]


def test_adaboost_quick_run(fake, tmp_path, no_cv):
    df, s = fake
    a = run_adaboost.parse_args(["--quick", "--results-dir", str(tmp_path)])
    run_adaboost.run_feature_set(a, df, s, "reduced")
    data = _check_quick_json(tmp_path / "metrics" / "adaboost_reduced_quick.json")
    p, q = data["params"], data["quick_config"]
    assert (p["n_estimators"], p["max_depth"], p["learning_rate"], p["algorithm"]) == (20, 6, 1.0, "SAMME")
    assert (q["n_estimators"], q["max_depth"], q["learning_rate"]) == (20, 6, 1.0)
    assert "resource-constrained educational" in q["resource_note"]
    assert 1 <= data["n_trees_fitted"] <= 20
    assert not list((tmp_path / "metrics").glob("*_cv.csv"))


def test_svm_quick_run(fake, tmp_path, no_cv, monkeypatch):
    df, s = fake
    monkeypatch.setitem(run_svm.QUICK, "fit_n", 1000)        # keep the synthetic fit tiny
    a = run_svm.parse_args(["--quick", "--results-dir", str(tmp_path)])
    run_svm.stage_a(a, df, s, "reduced")
    data = _check_quick_json(tmp_path / "metrics" / "svm_rbf_reduced_quick.json")
    assert data["params"] == {"kernel": "rbf", "C": 1000.0, "gamma": "auto"}       # paper C, not tuned
    assert data["n_train"] == data["quick_config"]["fit_rows"] == 1000
    assert data["n_test"] == len(s["test"]) and data["n_support_vectors"] > 0


@needs_tf
def test_mlp_quick_run(fake, tmp_path):
    df, s = fake
    a = run_mlp.parse_args(["--quick", "--verbose", "0", "--results-dir", str(tmp_path)])
    run_mlp.run_feature_set(a, df, s, "reduced")
    data = _check_quick_json(tmp_path / "metrics" / "mlp_reduced_quick.json")
    p = data["params"]
    assert (p["epochs"], p["batch_size"], p["learning_rate"], p["val_frac"], p["patience"]) == (5, 256, 0.08, 0.1, None)
    assert p["hidden_units"] == [512, 512] and data["epochs_run"] == 5
    assert data["quick_config"]["epochs"] == 5 and data["device"] in ("CPU", "GPU")
    assert len(data["history"]["val_loss"]) == len(data["history"]["val_accuracy"]) == 5
    assert data["n_val_rows"] == round(0.1 * len(s["train"]))
    hist = pd.read_csv(tmp_path / "metrics" / "mlp_reduced_quick_history.csv")
    assert list(hist["epoch"]) == [1, 2, 3, 4, 5]
    assert not (tmp_path / "metrics" / "mlp_reduced_history.csv").exists()     # FULL history untouched


def test_quick_output_is_classified_quick_by_compare_results(fake, tmp_path, no_cv):
    """A quick RF run uses ALL training/test rows, yet must never be counted as FULL."""
    df, s = fake
    (tmp_path / "split_meta.json").write_text(json.dumps({"n_train": len(s["train"]), "n_test": len(s["test"])}))
    a = run_random_forest.parse_args(["--quick", "--results-dir", str(tmp_path)])
    run_random_forest.run_feature_set(a, df, s, "reduced")
    rows, problems = cr.discover(tmp_path)
    assert not problems and [(r["status"], r["primary"]) for r in rows] == [("QUICK", True)]


# ---------------------------------------------------------------- compare_results statuses

N_TRAIN, N_TEST = 1000, 200


def _write(rdir, model, fs, acc, n_train=N_TRAIN, n_test=N_TEST, tag="", notes="", **extra):
    rec = {"model": model, "owner": "P2", "feature_set": fs, "protocol": "random_85_15", "seed": 229,
           "n_train": n_train, "n_test": n_test, "train_accuracy": 0.99, "test_accuracy": acc,
           "test_macro_f1": acc - 0.01, "test_weighted_f1": acc, "params": {}, "fit_seconds": 1.0,
           "predict_seconds": 0.1, "paper_test_accuracy": None, "notes": notes, **extra}
    return save_result(rec, results_dir=rdir, tag=tag)


def _quick(rdir, model, fs, acc, **kw):
    return _write(rdir, model, fs, acc, tag=quick_mode.TAG, notes=quick_mode.NOTE,
                  **quick_mode.record({"feature_set": fs}), **kw)


@pytest.fixture
def rdir(tmp_path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "split_meta.json").write_text(json.dumps({"n_train": N_TRAIN, "n_test": N_TEST}))
    return tmp_path


def _status(df, model, fs):
    r = df[(df["model"] == model) & (df["feature_set"] == fs) & df["primary"].astype(bool)]
    return sorted(r["status"])


def test_compare_distinguishes_full_quick_subsample_smoke_not_run(rdir, capsys):
    _write(rdir, "decision_tree", "reduced", 0.90)                                     # FULL
    _quick(rdir, "random_forest", "reduced", 0.99)                                     # QUICK, all rows
    _write(rdir, "svm_rbf", "reduced", 0.95, n_train=50)                               # SUBSAMPLE
    _write(rdir, "adaboost", "reduced", 0.97, tag="_smoke", notes="SMOKE TEST - not a result")
    df = cr.main(["--results-dir", str(rdir)])
    assert _status(df, "decision_tree", "reduced") == ["FULL"]
    assert _status(df, "random_forest", "reduced") == ["QUICK"]
    assert _status(df, "svm_rbf", "reduced") == ["SUBSAMPLE"]
    assert _status(df, "adaboost", "reduced") == ["NOT RUN"]                           # smoke never counts
    assert _status(df, "mlp", "full") == ["NOT RUN"]
    assert (df["status"] == "NOT RUN").sum() == 9
    q = df[df["status"] == "QUICK"].iloc[0]
    assert q["test_accuracy"] == 0.99 and q["stem"] == "random_forest_reduced_quick"
    nr = df[df["status"] == "NOT RUN"]
    assert nr["test_accuracy"].isna().all()                                            # never 0
    out = capsys.readouterr().out
    assert "QUICK runs only" in out and "Only a QUICK result (full-scale experiment not run): 1" in out
    csv = pd.read_csv(rdir / "comparison.csv")
    assert set(csv["status"]) == {"FULL", "QUICK", "SUBSAMPLE", "NOT RUN"}


def test_quick_is_never_ranked_or_compared_as_full(rdir):
    _write(rdir, "decision_tree", "reduced", 0.90)
    _write(rdir, "decision_tree", "full", 0.93)
    _quick(rdir, "random_forest", "reduced", 0.99)
    _quick(rdir, "random_forest", "full", 0.995)
    _quick(rdir, "mlp", "reduced", 0.80, n_train=1, n_test=1)                          # QUICK beats SUBSAMPLE too
    df, _ = cr.build_table(*cr.discover(rdir)), None
    for pool in (["FULL"], cr.NON_QUICK):
        assert "random_forest" not in set(cr.rank(cr.rankable(df, pool), "test_accuracy")["model"])
    quick_rank = cr.rank(cr.rankable(df, ["QUICK"]), "test_accuracy")
    assert list(quick_rank["stem"]) == ["random_forest_full_quick", "random_forest_reduced_quick",
                                        "mlp_reduced_quick"]
    _, gap = cr.paper_table(df)
    assert set(gap["model_name"]) == {"Decision Tree"}                                 # no QUICK vs paper
    fsc = cr.feature_set_comparison(df).set_index("model_name")
    assert fsc.loc["Random Forest", "basis"].startswith("cannot compare")
    assert fsc.loc["Decision Tree", "basis"] == "both FULL"


def test_quick_and_full_results_for_the_same_pair_stay_separate(rdir, capsys):
    _write(rdir, "random_forest", "reduced", 0.93)
    _quick(rdir, "random_forest", "reduced", 0.90)
    df = cr.main(["--results-dir", str(rdir), "--no-write"])
    assert _status(df, "random_forest", "reduced") == ["FULL", "QUICK"]
    full = cr.rank(cr.rankable(df, ["FULL"]), "test_accuracy")
    assert list(full["stem"]) == ["random_forest_reduced"]
    assert "Only a QUICK result" not in capsys.readouterr().out


def test_quick_detected_from_status_field_or_notes_alone(rdir):
    _write(rdir, "adaboost", "full", 0.9, status="QUICK")                              # untagged file
    _write(rdir, "mlp", "full", 0.9, notes=quick_mode.NOTE)
    df = cr.build_table(*cr.discover(rdir))
    assert _status(df, "adaboost", "full") == _status(df, "mlp", "full") == ["QUICK"]
    assert cr.rank(cr.rankable(df, ["FULL"]), "test_accuracy").empty


def test_only_quick_results_still_write_csv_and_say_so(rdir, capsys):
    _quick(rdir, "adaboost", "reduced", 0.8)
    cr.main(["--results-dir", str(rdir)])
    out = capsys.readouterr().out
    assert "Only QUICK (reduced-compute educational) results so far" in out
    assert (rdir / "comparison.csv").exists()
