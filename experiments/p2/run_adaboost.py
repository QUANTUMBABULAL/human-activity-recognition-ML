"""E6: AdaBoost (SAMME, Gini decision-tree base learners), (max_depth x n_estimators) by 5-fold CV
on training rows. OWNER: Person 2.

    python -m experiments.p2.run_adaboost --feature-set both
    python -m experiments.p2.run_adaboost --smoke          # quick pipeline check, NOT a result

Options: --tune-n 100000 (CV rows), --depth-grid 6 8 9 10 12, --trees-grid 50 100 250 500,
         --rule max|one_sd, --learning-rate 1.0, --fit-n (default: all training rows),
         --skip-tune (paper: 500 trees / depth 10 reduced, 250 / depth 9 full),
         --max-depth / --n-estimators (forced, no CV; the missing one comes from the paper)
--smoke uses tiny row counts and writes to a temp directory (never results/), so its numbers
cannot be mistaken for real results.

Compute: AdaBoost fits trees one after another (no n_jobs). Extrapolated from a one-tree probe,
a final fit on all 1,633,207 training rows takes roughly 3 h (reduced, 500 trees) and 4 h (full,
250 trees); --n-jobs only parallelises CV folds.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from har.config import N_FOLDS, RESULTS_DIR
from har.data import get_xy, load_processed
from har.models_p2 import adaboost as ab
from har.runner import run_experiment, subsample
from har.splits import load_split, split_fingerprint

SMOKE = {"tune_n": 3_000, "fit_n": 10_000, "test_n": 5_000, "depth_grid": [3, 6], "trees_grid": [10, 25]}


def run_feature_set(a, df, split, fs):
    paper = ab.PAPER[fs]
    tune_idx = subsample(split["train"], a.tune_n)          # TRAINING rows only
    cv_info = {"n_folds": N_FOLDS, "tune_rows": int(len(tune_idx)), "depth_grid": a.depth_grid,
               "trees_grid": a.trees_grid, "learning_rate": a.learning_rate, "rule": a.rule}
    if a.max_depth is not None or a.n_estimators is not None:
        n_est = a.n_estimators or paper["n_estimators"]
        depth = a.max_depth or paper["max_depth"]
        note = f"forced n_estimators={n_est}, max_depth={depth} (no CV)"
        cv_info["rule"] = "forced"
    elif a.skip_tune:
        n_est, depth = paper["n_estimators"], paper["max_depth"]
        note = "n_estimators and max_depth taken from paper (no CV)"
        cv_info["rule"] = "paper value"
    else:
        X, y = get_xy(df, fs, tune_idx)
        rows = ab.cv_grid(X, y, depth_grid=a.depth_grid, n_estimators_grid=a.trees_grid,
                          learning_rate=a.learning_rate, n_jobs=a.n_jobs)
        pd.DataFrame(rows).to_csv(Path(a.results_dir) / "metrics" / f"adaboost_{fs}_cv.csv",
                                  index=False)
        best = ab.select_config(rows, a.rule)
        n_est, depth = best["n_estimators"], best["max_depth"]
        top = ab.select_config(rows, "max")
        cv_info["best_by_max_val"] = {"n_estimators": top["n_estimators"], "max_depth": top["max_depth"]}
        note = f"config chosen by {N_FOLDS}-fold CV ({a.rule} rule) on {len(tune_idx)} training rows"
    cv_info["chosen"] = {"n_estimators": n_est, "max_depth": depth}
    print(f"[{fs}] n_estimators = {n_est}, max_depth = {depth}, learning_rate = {a.learning_rate} "
          f"(paper: {paper['n_estimators']}, {paper['max_depth']}, {paper['learning_rate']})", flush=True)

    fit_note = (f"fit on {min(a.fit_n, len(split['train']))} sampled training rows" if a.fit_n
                else "final fit on all training rows")
    test_note = f"; test on a seeded {a.test_n}-row subset" if a.test_n else "; test on the full test split"
    return run_experiment(model_name="adaboost", owner="P2",
                          model=ab.build(n_est, depth, a.learning_rate), df=df, split=split,
                          feature_set=fs,
                          params={"n_estimators": n_est, "max_depth": depth,
                                  "learning_rate": a.learning_rate, "algorithm": "SAMME",
                                  "estimator": "DecisionTreeClassifier", "criterion": "gini"},
                          paper_test_accuracy=ab.PAPER_TEST_ACC[fs], cv_info=cv_info,
                          fit_n=a.fit_n, test_n=a.test_n,
                          notes=f"{'SMOKE TEST - not a result; ' if a.smoke else ''}{note}; {fit_note}{test_note}",
                          post_fit=ab.boost_info, tag=a.tag, results_dir=a.results_dir)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["full", "reduced", "both"], default="both")
    ap.add_argument("--tune-n", type=int, default=100_000, help="training rows used for CV")
    ap.add_argument("--depth-grid", type=int, nargs="+", default=ab.DEPTH_GRID)
    ap.add_argument("--trees-grid", type=int, nargs="+", default=ab.N_ESTIMATORS_GRID)
    ap.add_argument("--learning-rate", type=float, default=ab.LEARNING_RATE)
    ap.add_argument("--fit-n", type=int, default=None, help="final-fit rows (default: all training rows)")
    ap.add_argument("--test-n", type=int, default=None, help="evaluate on a seeded test subset (fallback)")
    ap.add_argument("--rule", choices=ab.RULES, default="max", help="config selection rule")
    ap.add_argument("--skip-tune", action="store_true", help="use the paper's per-feature-set config")
    ap.add_argument("--max-depth", type=int, default=None, help="force a max_depth (logged as such)")
    ap.add_argument("--n-estimators", type=int, default=None, help="force a tree count (logged as such)")
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel CV folds (AdaBoost itself is sequential)")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline; output to a temp dir")
    a = ap.parse_args(argv)
    a.tag = ""
    if a.smoke:
        a.tune_n, a.fit_n, a.test_n = SMOKE["tune_n"], SMOKE["fit_n"], SMOKE["test_n"]
        a.depth_grid, a.trees_grid = SMOKE["depth_grid"], SMOKE["trees_grid"]
        a.tag = "_smoke"
        a.results_dir = a.results_dir or tempfile.mkdtemp(prefix="har_ab_smoke_")
    a.results_dir = a.results_dir or str(RESULTS_DIR)
    (Path(a.results_dir) / "metrics").mkdir(parents=True, exist_ok=True)
    return a


def main(argv=None):
    a = parse_args(argv)
    df, split = load_processed(), load_split()
    print("split fingerprint:", split_fingerprint(split), flush=True)
    for fs in (["full", "reduced"] if a.feature_set == "both" else [a.feature_set]):
        run_feature_set(a, df, split, fs)
    print("results dir:", a.results_dir)


if __name__ == "__main__":
    main()
