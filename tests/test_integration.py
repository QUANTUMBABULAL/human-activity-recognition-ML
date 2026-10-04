"""End-to-end: fake raw files -> clean -> split -> LR -> JSON + confusion CSV. OWNER: Person 1."""
import json

import pandas as pd

from har.config import LABEL_ORDER
from har.data import build_dataset
from har.metrics import REQUIRED_KEYS
from har.models_p1 import logreg, svm
from har.plots import plot_confusion, plot_curve
from har.runner import run_experiment, subsample
from har.splits import make_random_split
from har.synthetic import write_fake_dat_files


def test_end_to_end(tmp_path):
    raw = tmp_path / "raw"; write_fake_dat_files(raw, n_subjects=3, rows=2500)
    df, _ = build_dataset(raw)
    split = make_random_split(len(df))
    res = tmp_path / "res"
    rec = run_experiment(model_name="logreg", owner="P1", model=logreg.build(1.0, 300), df=df, split=split,
                         feature_set="reduced", params={"C": 1.0}, paper_test_accuracy=0.639,
                         cv_info={"chosen_C": 1.0}, results_dir=res)
    data = json.loads((res / "metrics" / "logreg_reduced.json").read_text())
    assert REQUIRED_KEYS <= data.keys() and "versions" in data and "timestamp_utc" in data
    assert data["n_train"] == len(split["train"]) and data["n_test"] == len(split["test"])
    cm = pd.read_csv(res / "confusion" / "logreg_reduced.csv", index_col=0)
    assert cm.shape == (12, 12) and cm.to_numpy().sum() == data["n_test"]
    assert plot_confusion("logreg_reduced", res).exists()
    assert rec["test_accuracy"] > 0.3


def test_svm_run_with_post_fit_and_curve_plot(tmp_path):
    raw = tmp_path / "raw"; write_fake_dat_files(raw, n_subjects=2, rows=2000)
    df, _ = build_dataset(raw)
    split = make_random_split(len(df))
    res = tmp_path / "res"
    rec = run_experiment(model_name="svm_rbf", owner="P1", model=svm.build("rbf", 10.0), df=df, split=split,
                         feature_set="full", params={"C": 10.0}, fit_n=500, test_n=200, train_eval_n=300,
                         post_fit=lambda m: {"n_support_vectors": svm.n_support_vectors(m)}, results_dir=res)
    assert rec["n_train"] == 500 and rec["n_test"] == 200 and rec["n_support_vectors"] > 0
    (res / "metrics").mkdir(exist_ok=True)
    pd.DataFrame({"C": [0.1, 1, 10], "train_mean": [.5, .7, .9], "val_mean": [.5, .65, .8],
                  "val_std": [.01] * 3}).to_csv(res / "metrics" / "svm_rbf_full_cv.csv", index=False)
    assert plot_curve("svm_rbf_full", "C", log=True, results_dir=res).exists()


def test_subsample_is_deterministic_and_subset():
    idx = make_random_split(5000)["train"]
    a, b = subsample(idx, 100), subsample(idx, 100)
    assert (a == b).all() and set(a) <= set(idx) and len(a) == 100
    assert subsample(idx, None) is idx or (subsample(idx, None) == idx).all()
