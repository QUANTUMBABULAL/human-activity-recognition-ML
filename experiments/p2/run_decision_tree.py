"""E4: Decision tree (Gini), max_depth by 5-fold CV on training rows. OWNER: Person 2.

    python -m experiments.p2.run_decision_tree --feature-set both
    python -m experiments.p2.run_decision_tree --smoke          # quick pipeline check, NOT a result

Options: --tune-n 200000 (CV rows), --rule one_sd|max, --fit-n (default: all training rows),
         --skip-tune (paper depth 15), --max-depth N (forced, no CV)
--smoke uses tiny row counts and writes to a temp directory (never results/), so its numbers
cannot be mistaken for real results.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from har.config import N_FOLDS, RESULTS_DIR
from har.data import get_xy, load_processed
from har.models_p2 import decision_tree as dt
from har.runner import run_experiment, subsample
from har.splits import load_split, split_fingerprint

SMOKE = {"tune_n": 5_000, "fit_n": 20_000, "test_n": 5_000, "depth_grid": [4, 10, 15]}


def run_feature_set(a, df, split, fs):
    tune_idx = subsample(split["train"], a.tune_n)          # TRAINING rows only
    cv_info = {"n_folds": N_FOLDS, "tune_rows": int(len(tune_idx)), "grid": a.depth_grid,
               "rule": a.rule}
    if a.max_depth is not None:
        depth, note = a.max_depth, f"max_depth forced to {a.max_depth} (no CV)"
        cv_info["rule"] = "forced"
    elif a.skip_tune:
        depth, note = dt.PAPER["max_depth"], "max_depth taken from paper (no CV)"
        cv_info["rule"] = "paper value"
    else:
        X, y = get_xy(df, fs, tune_idx)
        rows = dt.cv_grid(X, y, depth_grid=a.depth_grid, n_jobs=a.n_jobs)
        pd.DataFrame(rows).to_csv(Path(a.results_dir) / "metrics" / f"decision_tree_{fs}_cv.csv",
                                  index=False)
        depth = dt.select_depth(rows, a.rule)["max_depth"]
        cv_info["best_by_max_val"] = dt.select_depth(rows, "max")["max_depth"]
        note = f"max_depth chosen by {N_FOLDS}-fold CV ({a.rule} rule) on {len(tune_idx)} training rows"
    cv_info["chosen_max_depth"] = depth
    print(f"[{fs}] max_depth = {depth} (paper: {dt.PAPER['max_depth']})", flush=True)

    fit_note = (f"fit on {min(a.fit_n, len(split['train']))} sampled training rows" if a.fit_n
                else "final fit on all training rows")
    test_note = f"; test on a seeded {a.test_n}-row subset" if a.test_n else "; test on the full test split"
    return run_experiment(model_name="decision_tree", owner="P2", model=dt.build(depth), df=df,
                          split=split, feature_set=fs,
                          params={"criterion": dt.PAPER["criterion"], "max_depth": depth,
                                  "min_samples_leaf": 1},
                          paper_test_accuracy=dt.PAPER_TEST_ACC[fs], cv_info=cv_info,
                          fit_n=a.fit_n, test_n=a.test_n,
                          notes=f"{'SMOKE TEST - not a result; ' if a.smoke else ''}{note}; {fit_note}{test_note}",
                          post_fit=dt.tree_info, tag=a.tag, results_dir=a.results_dir)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["full", "reduced", "both"], default="both")
    ap.add_argument("--tune-n", type=int, default=200_000, help="training rows used for CV")
    ap.add_argument("--fit-n", type=int, default=None, help="final-fit rows (default: all training rows)")
    ap.add_argument("--test-n", type=int, default=None, help="evaluate on a seeded test subset (fallback)")
    ap.add_argument("--rule", choices=dt.RULES, default="one_sd", help="depth selection rule (paper: one_sd)")
    ap.add_argument("--skip-tune", action="store_true", help="use the paper's max_depth = 15")
    ap.add_argument("--max-depth", type=int, default=None, help="force a max_depth (logged as such)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline; output to a temp dir")
    a = ap.parse_args(argv)
    a.depth_grid, a.tag = dt.DEPTH_GRID, ""
    if a.smoke:
        a.tune_n, a.fit_n, a.test_n, a.depth_grid = (SMOKE["tune_n"], SMOKE["fit_n"], SMOKE["test_n"],
                                                     SMOKE["depth_grid"])
        a.tag = "_smoke"
        a.results_dir = a.results_dir or tempfile.mkdtemp(prefix="har_dt_smoke_")
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
