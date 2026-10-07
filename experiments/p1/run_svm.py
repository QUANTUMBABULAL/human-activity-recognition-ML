"""E2/E3: SVM. OWNER: Person 1.

Stage A (MUST): RBF only. 5-fold CV over C on --tune-n training rows; final fit on --fit-n rows;
                evaluated on the test split (optionally a seeded --test-n subset).
Stage B (SHOULD): linear/poly/rbf x C in {0.1,1,10,100}, CV only, on --tune-n rows (default 10k).
Stage C (OPTIONAL): Stage A with a larger --fit-n (e.g. 100000-175000).

    python -m experiments.p1.run_svm --stage A --feature-set both > svm_stageA.log 2>&1
    python -m experiments.p1.run_svm --stage B --feature-set both --tune-n 10000
    python -m experiments.p1.run_svm --quick       # cheap educational run (status QUICK)

--quick (decision P2-21 in docs/decisions_p2.md) is a fixed reduced-compute preset: Stage A only,
reduced features, RBF with the paper's C for that feature set (C = 1000 reduced / 100 full; no CV,
nothing tuned), final fit on a seeded 30,000-row subset of the TRAINING rows, full test split.
Written to results/metrics/svm_rbf_<fs>_quick.json with status QUICK - never a FULL result.
"""
from __future__ import annotations

import argparse

import pandas as pd

from experiments import quick_mode
from har.config import N_FOLDS, RESULTS_DIR
from har.data import get_xy, load_processed
from har.models_p1 import svm
from har.runner import run_experiment, subsample
from har.splits import load_split, split_fingerprint

QUICK = {"feature_set": "reduced", "fit_n": 30_000, "kernel": "rbf"}
QUICK_LOCKED = ("stage", "tune_n", "fit_n", "test_n", "skip_tune", "C")


def stage_a(a, df, split, fs):
    if a.quick:
        return run_quick(a, df, split, fs)
    tune_idx = subsample(split["train"], a.tune_n)           # TRAINING rows only
    cv_info = {"n_folds": N_FOLDS, "tune_rows": int(len(tune_idx)), "grid": svm.C_GRID,
               "rule": "max mean val accuracy"}
    if a.C is not None:
        C, note = a.C, f"C forced to {a.C} (no CV)"; cv_info["rule"] = "forced"
    elif a.skip_tune:
        C, note = svm.PAPER[fs]["C"], "C taken from paper (no CV)"; cv_info["rule"] = "paper value"
    else:
        X, y = get_xy(df, fs, tune_idx)
        best, rows = svm.cv_grid(X, y, kernels=("rbf",), C_grid=svm.C_GRID, n_jobs=a.n_jobs,
                                 cache_mb=a.cache_mb)
        pd.DataFrame(rows).to_csv(f"{a.results_dir}/metrics/svm_rbf_{fs}_cv.csv", index=False)
        C, note = best["C"], f"C chosen by {N_FOLDS}-fold CV on {len(tune_idx)} training rows"
    cv_info["chosen_C"] = C
    print(f"[{fs}] C = {C} (paper: {svm.PAPER[fs]['C']})", flush=True)
    model = svm.build("rbf", C, cache_size_mb=a.cache_mb * 4)
    fit_note = f"fit on {min(a.fit_n or 10**12, len(split['train']))} sampled training rows"
    test_note = f"; test on a seeded {a.test_n}-row subset" if a.test_n else "; test on the full test split"
    run_experiment(model_name="svm_rbf", owner="P1", model=model, df=df, split=split, feature_set=fs,
                   params={"kernel": "rbf", "C": C, "gamma": "auto"},
                   paper_test_accuracy=svm.PAPER_TEST_ACC[fs], cv_info=cv_info,
                   fit_n=a.fit_n, test_n=a.test_n, train_eval_n=20_000,
                   notes=f"{note}; {fit_note}{test_note}",
                   post_fit=lambda m: {"n_support_vectors": svm.n_support_vectors(m)},
                   results_dir=a.results_dir)


def run_quick(a, df, split, fs):
    """Fixed QUICK preset: paper C (no CV), fit on QUICK['fit_n'] seeded training rows, full test split."""
    C, fit_n = svm.PAPER[fs]["C"], QUICK["fit_n"]
    n_fit = min(fit_n, len(split["train"]))
    config = {"feature_set": fs, "kernel": "rbf", "C": C, "gamma": "auto", "C_source": "paper value (not tuned)",
              "fit_rows": n_fit, "test_rows": "full test split"}
    print(f"[{fs}] QUICK preset: rbf, C = {C} (paper value, no CV), fit on {n_fit} training rows", flush=True)
    return run_experiment(model_name="svm_rbf", owner="P1", model=svm.build("rbf", C, cache_size_mb=a.cache_mb * 4),
                          df=df, split=split, feature_set=fs,
                          params={"kernel": "rbf", "C": C, "gamma": "auto"},
                          paper_test_accuracy=svm.PAPER_TEST_ACC[fs], cv_info=quick_mode.cv_info(),
                          fit_n=fit_n, train_eval_n=20_000,
                          notes=f"{quick_mode.NOTE}; C taken from paper (no CV); fit on {n_fit} sampled "
                                "training rows (subsample); test on the full test split",
                          post_fit=lambda m: {"n_support_vectors": svm.n_support_vectors(m),
                                              **quick_mode.record(config)},
                          tag=quick_mode.TAG, results_dir=a.results_dir)


def stage_b(a, df, split, fs):
    tune_idx = subsample(split["train"], a.tune_n)
    X, y = get_xy(df, fs, tune_idx)
    _, rows = svm.cv_grid(X, y, kernels=("linear", "poly", "rbf"), C_grid=svm.KERNEL_C_GRID,
                          n_jobs=a.n_jobs, max_iter=a.max_iter, cache_mb=a.cache_mb)
    pd.DataFrame(rows).to_csv(f"{a.results_dir}/metrics/svm_kernels_{fs}_cv.csv", index=False)
    summarise_kernels(a.results_dir)


def summarise_kernels(results_dir):
    """Paper: RBF beats linear by ~15 and poly by ~10 points (validation accuracy, both sets averaged)."""
    best = {}
    for fs in ("full", "reduced"):
        try:
            t = pd.read_csv(f"{results_dir}/metrics/svm_kernels_{fs}_cv.csv")
        except FileNotFoundError:
            continue
        best[fs] = t.groupby("kernel")["val_mean"].max()
    if not best:
        return
    s = pd.DataFrame(best)
    s["mean_over_sets"] = s.mean(axis=1)
    out = pd.DataFrame({"comparison": ["rbf - linear (pp)", "rbf - poly (pp)"],
                        **{c: [100 * (s.loc["rbf", c] - s.loc["linear", c]),
                               100 * (s.loc["rbf", c] - s.loc["poly", c])] for c in s.columns}})
    out.to_csv(f"{results_dir}/metrics/svm_kernels_summary.csv", index=False)
    print("\nBest validation accuracy per kernel:\n", s.round(4).to_string())
    print("\nKernel gaps (paper: ~+15 vs linear, ~+10 vs poly):\n", out.round(2).to_string(index=False))


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], default="A")
    ap.add_argument("--feature-set", choices=["full", "reduced", "both"], default=None,
                    help="default: both (--quick: reduced)")
    ap.add_argument("--tune-n", type=int, default=None, help="CV rows (default 20000 for A, 10000 for B)")
    ap.add_argument("--fit-n", type=int, default=50_000, help="Stage A final-fit rows")
    ap.add_argument("--test-n", type=int, default=None, help="evaluate on a seeded test subset (fallback)")
    ap.add_argument("--skip-tune", action="store_true", help="Stage A: use the paper's C")
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--max-iter", type=int, default=200_000, help="Stage B cap for linear/poly (-1 = none)")
    ap.add_argument("--cache-mb", type=int, default=500, help="libsvm cache per CV worker (final fit uses 4x)")
    ap.add_argument("--n-jobs", type=int, default=N_FOLDS)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    quick_mode.add_argument(ap)
    a = ap.parse_args(argv)
    quick_mode.check_args(ap, a, QUICK_LOCKED, argv)
    a.feature_set = a.feature_set or (QUICK["feature_set"] if a.quick else "both")
    if a.tune_n is None:
        a.tune_n = 20_000 if a.stage == "A" else 10_000
    return a


def main(argv=None):
    a = parse_args(argv)
    df, split = load_processed(), load_split()
    print("split fingerprint:", split_fingerprint(split), flush=True)
    for fs in (["full", "reduced"] if a.feature_set == "both" else [a.feature_set]):
        (stage_a if a.stage == "A" else stage_b)(a, df, split, fs)


if __name__ == "__main__":
    main()
