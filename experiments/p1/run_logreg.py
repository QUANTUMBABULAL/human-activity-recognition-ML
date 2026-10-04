"""E1: Logistic regression (L2, SAG), C by 5-fold CV on training rows. OWNER: Person 1.

    python -m experiments.p1.run_logreg --feature-set both
Options: --tune-n 200000 (CV rows), --max-iter 300 (CV) / --final-max-iter 1000, --skip-tune (paper C=0.01)
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from har.config import N_FOLDS, RESULTS_DIR
from har.data import get_xy, load_processed
from har.models_p1 import logreg
from har.runner import run_experiment, subsample
from har.splits import load_split, split_fingerprint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["full", "reduced", "both"], default="both")
    ap.add_argument("--tune-n", type=int, default=200_000, help="training rows used for CV")
    ap.add_argument("--max-iter", type=int, default=300, help="max_iter inside CV")
    ap.add_argument("--final-max-iter", type=int, default=1000, help="max_iter of the final fit")
    ap.add_argument("--skip-tune", action="store_true", help="use the paper's C = 0.01")
    ap.add_argument("--C", type=float, default=None, help="force a C (logged as such)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    a = ap.parse_args()

    df, split = load_processed(), load_split()
    print("split fingerprint:", split_fingerprint(split))
    sets = ["full", "reduced"] if a.feature_set == "both" else [a.feature_set]
    for fs in sets:
        tune_idx = subsample(split["train"], a.tune_n)      # TRAINING rows only
        cv_info = {"n_folds": N_FOLDS, "tune_rows": int(len(tune_idx)),
                   "grid": logreg.C_GRID, "rule": "max mean val accuracy"}
        if a.C is not None:
            C, note = a.C, f"C forced to {a.C} (no CV)"
            cv_info["rule"] = "forced"
        elif a.skip_tune:
            C, note = logreg.PAPER["C"], "C taken from paper (no CV)"
            cv_info["rule"] = "paper value"
        else:
            X, y = get_xy(df, fs, tune_idx)
            C, table = logreg.tune_C(X, y, max_iter=a.max_iter, n_jobs=a.n_jobs)
            pd.DataFrame(table).to_csv(f"{a.results_dir}/metrics/logreg_{fs}_cv.csv", index=False)
            print(pd.DataFrame(table).to_string(index=False))
            note = f"C chosen by {N_FOLDS}-fold CV on {len(tune_idx)} training rows"
        cv_info["chosen_C"] = C
        print(f"[{fs}] chosen C = {C} (paper: {logreg.PAPER['C']})")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            run_experiment(model_name="logreg", owner="P1", model=logreg.build(C, a.final_max_iter),
                           df=df, split=split, feature_set=fs,
                           params={"C": C, "solver": "sag", "penalty": "l2", "max_iter": a.final_max_iter},
                           paper_test_accuracy=logreg.PAPER_TEST_ACC[fs], cv_info=cv_info,
                           notes=f"{note}; final fit on all training rows",
                           results_dir=a.results_dir)
        if any(issubclass(x.category, ConvergenceWarning) for x in w):
            print(f"!! ConvergenceWarning in final fit ({fs}): rerun with a larger --final-max-iter "
                  "and log it in docs/decisions_p1.md")


if __name__ == "__main__":
    main()
