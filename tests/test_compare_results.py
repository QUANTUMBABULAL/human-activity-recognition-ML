"""Tests for experiments/compare_results.py. OWNER: Person 2.

Every result JSON here is a hand-made fixture in a temp directory (written with the real
har.metrics.save_result so the format matches). The numbers are arbitrary test inputs, not results.
No dataset is loaded and no model is trained.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from experiments import compare_results as cr
from har.metrics import save_result

N_TRAIN, N_TEST = 1000, 200        # the fake "frozen split" counts for these tests


def _write(results_dir, model, fs, acc, macro, weighted, fit_s=1.5, pred_s=0.25, n_train=N_TRAIN,
           n_test=N_TEST, tag="", notes="", **extra):
    rec = {"model": model, "owner": "P2", "feature_set": fs, "protocol": "random_85_15", "seed": 229,
           "n_train": n_train, "n_test": n_test, "train_accuracy": 0.99, "test_accuracy": acc,
           "test_macro_f1": macro, "test_weighted_f1": weighted, "params": {"depth": 3},
           "fit_seconds": fit_s, "predict_seconds": pred_s, "paper_test_accuracy": None,
           "notes": notes, **extra}
    return save_result(rec, results_dir=results_dir, tag=tag)


@pytest.fixture
def rdir(tmp_path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "split_meta.json").write_text(json.dumps({"n_train": N_TRAIN, "n_test": N_TEST}))
    return tmp_path


def _table(rdir):
    rows, problems = cr.discover(rdir)
    return cr.build_table(rows, problems), rows, problems


def _row(df, model, fs):
    r = df[(df["model"] == model) & (df["feature_set"] == fs) & df["primary"].astype(bool)]
    assert len(r) == 1
    return r.iloc[0]


def test_empty_results_dir_gives_all_not_run_and_writes_nothing(rdir, capsys):
    df = cr.main(["--results-dir", str(rdir)])
    assert len(df) == 12 and (df["status"] == "NOT RUN").all()
    assert df[["test_accuracy", "test_macro_f1", "test_weighted_f1", "fit_seconds"]].isna().all().all()
    out = capsys.readouterr().out
    assert "no final experiments have been run yet" in out
    assert not (rdir / "comparison.csv").exists()
    assert [p.name for p in (rdir / "metrics").iterdir()] == []   # no fake result files created


def test_missing_metrics_dir_does_not_crash(tmp_path):
    df = cr.main(["--results-dir", str(tmp_path), "--no-write"])
    assert (df["status"] == "NOT RUN").all()


def test_discovery_and_exact_metric_extraction(rdir):
    _write(rdir, "decision_tree", "reduced", 0.8731, 0.8012, 0.8705, fit_s=12.34, pred_s=0.56)
    _write(rdir, "decision_tree", "full", 0.9271, 0.9100, 0.9265, fit_s=20.0, pred_s=0.75)
    df, rows, problems = _table(rdir)
    assert len(rows) == 2 and not problems
    r = _row(df, "decision_tree", "reduced")
    assert (r["test_accuracy"], r["test_macro_f1"], r["test_weighted_f1"]) == (0.8731, 0.8012, 0.8705)
    assert (r["fit_seconds"], r["predict_seconds"]) == (12.34, 0.56)
    assert (r["n_train"], r["n_test"], r["status"]) == (N_TRAIN, N_TEST, "FULL")
    assert json.loads(r["params"]) == {"depth": 3}
    f = _row(df, "decision_tree", "full")                         # feature sets kept apart
    assert f["test_accuracy"] == 0.9271 and f["fit_seconds"] == 20.0
    assert _row(df, "mlp", "full")["status"] == "NOT RUN"
    assert (df["status"] == "NOT RUN").sum() == 10


def test_smoke_results_never_count_as_final(rdir, capsys):
    _write(rdir, "adaboost", "full", 0.96, 0.95, 0.96, tag="_smoke", notes="SMOKE TEST - not a result")
    _write(rdir, "mlp", "full", 0.70, 0.60, 0.70, notes="SMOKE TEST - not a result; untagged")
    df = cr.main(["--results-dir", str(rdir)])
    assert _row(df, "adaboost", "full")["status"] == "NOT RUN"
    assert _row(df, "mlp", "full")["status"] == "NOT RUN"
    assert not df["status"].eq("SMOKE").any() and df["test_accuracy"].isna().all()
    assert not (rdir / "comparison.csv").exists()                # smoke alone is not a result
    out = capsys.readouterr().out
    assert "Excluded SMOKE result files (2" in out and "no final experiments have been run yet" in out


def test_subsample_and_unverified_status(rdir, tmp_path_factory):
    _write(rdir, "svm_rbf", "full", 0.98, 0.97, 0.98, n_train=50)           # fewer training rows
    _write(rdir, "logreg", "full", 0.81, 0.75, 0.80, n_test=100)            # test subset
    df, _, _ = _table(rdir)
    assert _row(df, "svm_rbf", "full")["status"] == "SUBSAMPLE"
    assert _row(df, "logreg", "full")["status"] == "SUBSAMPLE"
    other = tmp_path_factory.mktemp("no_meta")                               # no split_meta.json
    _write(other, "logreg", "full", 0.81, 0.75, 0.80)
    df2, _, _ = _table(other)
    assert _row(df2, "logreg", "full")["status"] == "UNVERIFIED"


def test_missing_optional_metric_stays_missing_not_zero(rdir):
    path = _write(rdir, "random_forest", "reduced", 0.93, 0.90, 0.93)
    rec = json.loads(path.read_text())
    del rec["test_weighted_f1"]
    rec["predict_seconds"] = None
    path.write_text(json.dumps(rec))
    df, _, _ = _table(rdir)
    r = _row(df, "random_forest", "reduced")
    assert r["status"] == "FULL" and r["test_accuracy"] == 0.93
    assert math.isnan(r["test_weighted_f1"]) and math.isnan(r["predict_seconds"])
    ranked = cr.rank(cr.rankable(df, ["FULL"]), "test_weighted_f1")
    assert ranked.empty                                                       # missing is not ranked


def test_malformed_files_are_reported_not_fatal(rdir, capsys):
    _write(rdir, "adaboost", "reduced", 0.94, 0.93, 0.94)
    (rdir / "metrics" / "mlp_full.json").write_text("{not json")
    (rdir / "metrics" / "random_forest_full.json").write_text(json.dumps({"model": "random_forest"}))
    (rdir / "metrics" / "weird.json").write_text("[1, 2]")
    df = cr.main(["--results-dir", str(rdir), "--no-write"])
    assert _row(df, "adaboost", "reduced")["status"] == "FULL"
    assert _row(df, "mlp", "full")["status"] == "MALFORMED"
    assert _row(df, "random_forest", "full")["status"] == "MALFORMED"
    assert df.loc[df["status"] == "MALFORMED", "test_accuracy"].isna().all()
    out = capsys.readouterr().out
    assert "Problems (3 unreadable" in out and "missing required keys" in out


def test_rankings_with_ties_and_full_only_pool(rdir):
    _write(rdir, "logreg", "full", 0.90, 0.80, 0.85)
    _write(rdir, "decision_tree", "full", 0.95, 0.80, 0.94)
    _write(rdir, "random_forest", "full", 0.95, 0.85, 0.95)
    _write(rdir, "svm_rbf", "full", 0.99, 0.99, 0.99, n_train=50)            # SUBSAMPLE
    df, _, _ = _table(rdir)
    full = cr.rank(cr.rankable(df, ["FULL"]), "test_accuracy")
    assert "svm_rbf" not in set(full["model"])                                # subsample not in FULL ranking
    assert list(zip(full["model"], full["rank_label"])) == [
        ("decision_tree", "1="), ("random_forest", "1="), ("logreg", "3")]   # tie -> 1, 1, 3
    macro = cr.rank(cr.rankable(df, ["FULL"]), "test_macro_f1")
    assert list(macro["rank_label"]) == ["1", "2=", "2="]
    allr = cr.rank(cr.rankable(df, ["FULL", "SUBSAMPLE", "UNVERIFIED"]), "test_accuracy")
    assert allr.iloc[0]["model"] == "svm_rbf" and allr.iloc[0]["status"] == "SUBSAMPLE"


def test_extra_tagged_run_is_listed_but_not_primary(rdir):
    _write(rdir, "svm_rbf", "full", 0.97, 0.96, 0.97)
    _write(rdir, "svm_rbf", "full", 0.98, 0.97, 0.98, tag="_stageC")
    df, _, _ = _table(rdir)
    rows = df[(df["model"] == "svm_rbf") & (df["feature_set"] == "full")]
    assert len(rows) == 2 and rows["primary"].tolist() == [True, False]
    ranked = cr.rank(cr.rankable(df, ["FULL"]), "test_accuracy")
    assert ranked["stem"].tolist() == ["svm_rbf_full"]                        # one entry per model/set


def test_feature_set_comparison_only_when_both_exist(rdir):
    _write(rdir, "adaboost", "reduced", 0.90, 0.85, 0.89)
    _write(rdir, "adaboost", "full", 0.97, 0.96, 0.97)
    _write(rdir, "mlp", "full", 0.95, 0.94, 0.95)                              # reduced missing
    _write(rdir, "logreg", "reduced", 0.64, 0.50, 0.60, n_train=10)          # SUBSAMPLE
    _write(rdir, "logreg", "full", 0.81, 0.70, 0.80)
    fsc = cr.feature_set_comparison(_table(rdir)[0]).set_index("model_name")
    assert fsc.loc["AdaBoost", "basis"] == "both FULL"
    assert fsc.loc["AdaBoost", "Accuracy (full - reduced)"] == pytest.approx(0.07)
    assert fsc.loc["MLP", "basis"] == "cannot compare: reduced missing"
    assert pd.isna(fsc.loc["MLP", "Accuracy (full - reduced)"])
    assert "not a full-data comparison" in fsc.loc["Logistic Regression", "basis"]


def test_paper_values_kept_separate_from_ours(rdir):
    _write(rdir, "random_forest", "full", 0.97, 0.96, 0.97)
    df = cr.main(["--results-dir", str(rdir)])
    ref, gap = cr.paper_table(df)
    assert len(ref) == 12 and ref["PAPER test accuracy"].notna().all()
    assert ref.set_index(["model_name", "feature_set"]).loc[("MLP", "reduced"), "PAPER test accuracy"] == 0.814
    assert len(gap) == 1 and gap.iloc[0]["PAPER test accuracy"] == 0.980
    assert gap.iloc[0]["OURS test accuracy"] == 0.97
    csv = pd.read_csv(rdir / "comparison.csv")                               # our numbers only
    assert not any("paper" in c.lower() for c in csv.columns)


def test_csv_has_real_rows_and_empty_cells_for_missing(rdir):
    _write(rdir, "decision_tree", "full", 0.9271, 0.91, 0.9265, fit_s=20.0)
    cr.main(["--results-dir", str(rdir)])
    csv = pd.read_csv(rdir / "comparison.csv")
    assert len(csv) == 12 and (csv["status"] == "NOT RUN").sum() == 11
    real = csv[csv["status"] == "FULL"].iloc[0]
    assert real["test_accuracy"] == 0.9271 and real["fit_seconds"] == 20.0
    nr = csv[csv["status"] == "NOT RUN"]
    assert nr[["test_accuracy", "test_macro_f1", "test_weighted_f1", "fit_seconds"]].isna().all().all()
    assert not (nr[["test_accuracy", "fit_seconds"]] == 0).any().any()
    assert sorted(p.name for p in (rdir / "metrics").iterdir()) == ["decision_tree_full.json"]


def test_comparison_needs_no_dataset_and_trains_nothing(rdir, monkeypatch):
    """Generating the comparison must not load PAMAP2 or call the experiment runner."""
    import har.data
    import har.runner

    def forbidden(*_a, **_k):
        raise AssertionError("compare_results touched the dataset / training code")
    monkeypatch.setattr(har.data, "load_processed", forbidden)
    monkeypatch.setattr(har.data, "get_xy", forbidden)
    monkeypatch.setattr(har.runner, "run_experiment", forbidden)
    _write(rdir, "logreg", "reduced", 0.64, 0.5, 0.6)
    df = cr.main(["--results-dir", str(rdir), "--no-write"])
    assert _row(df, "logreg", "reduced")["status"] == "FULL"
    src = open(cr.__file__, encoding="utf-8").read()
    assert "run_experiment" not in src and "load_processed" not in src and ".fit(" not in src


def test_expected_models_and_paper_numbers_cover_all_twelve():
    assert list(cr.MODELS) == ["logreg", "svm_rbf", "decision_tree", "random_forest", "adaboost", "mlp"]
    paper = {(m, fs): cr.PAPER_TEST_ACC[m][fs] for m in cr.MODELS for fs in cr.FEATURE_SETS}
    assert len(paper) == 12 and all(0 < v < 1 for v in paper.values())
    np.testing.assert_allclose([paper[("adaboost", "full")], paper[("logreg", "reduced")]], [0.985, 0.639])
