"""E7: MLP (512-512 ReLU, dropout 0.5, softmax, categorical cross-entropy, SGD). OWNER: Person 2.

    python -m experiments.p2.run_mlp --feature-set both
    python -m experiments.p2.run_mlp --smoke          # quick pipeline check, NOT a result

Options: --epochs 100, --batch-size 32, --learning-rate 0.01, --momentum 0.0,
         --val-frac 0.1 (seeded slice of the TRAINING rows, monitoring only; 0 = fit on all of them),
         --patience N (early stopping on that slice; default off), --fit-n (default: all training rows)
The architecture is fixed to the paper's (no architecture search). The scaler, label encoding and
validation slice all live inside KerasMLP.fit, which only ever receives training rows.
--smoke uses tiny row counts and 2 epochs and writes to a temp directory (never results/).
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from har.config import RESULTS_DIR
from har.data import load_processed
from har.models_p2 import mlp
from har.runner import run_experiment
from har.splits import load_split, split_fingerprint

SMOKE = {"fit_n": 20_000, "test_n": 5_000, "epochs": 2}


def run_feature_set(a, df, split, fs):
    model = mlp.build(epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.learning_rate,
                      momentum=a.momentum, val_frac=a.val_frac, patience=a.patience,
                      verbose=a.verbose)
    fit_note = (f"fit on {min(a.fit_n, len(split['train']))} sampled training rows "
                f"(reproduced on a subsample)" if a.fit_n else "fit on all training rows")
    val_note = (f", of which a seeded {a.val_frac:.0%} slice is held out for validation monitoring"
                if a.val_frac else ", no validation slice")
    stop_note = f"; early stopping patience {a.patience}" if a.patience else f"; fixed {a.epochs} epochs"
    test_note = f"; test on a seeded {a.test_n}-row subset" if a.test_n else "; test on the full test split"
    print(f"[{fs}] {model.get_config()}", flush=True)
    rec = run_experiment(model_name="mlp", owner="P2", model=model, df=df, split=split, feature_set=fs,
                         params=model.get_config(), paper_test_accuracy=mlp.PAPER_TEST_ACC[fs],
                         cv_info={"selection": "none - paper architecture; validation slice from "
                                               "training rows used for monitoring only"},
                         fit_n=a.fit_n, test_n=a.test_n,
                         notes=f"{'SMOKE TEST - not a result; ' if a.smoke else ''}{fit_note}{val_note}"
                               f"{stop_note}{test_note}",
                         post_fit=mlp.mlp_info, tag=a.tag, results_dir=a.results_dir)
    hist = pd.DataFrame(model.history_)
    hist.insert(0, "epoch", range(1, len(hist) + 1))
    hist.to_csv(Path(a.results_dir) / "metrics" / f"mlp_{fs}_history.csv", index=False)
    return rec


def parse_args(argv=None):
    d = mlp.DEFAULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["full", "reduced", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=d["epochs"])
    ap.add_argument("--batch-size", type=int, default=d["batch_size"])
    ap.add_argument("--learning-rate", type=float, default=d["learning_rate"])
    ap.add_argument("--momentum", type=float, default=d["momentum"])
    ap.add_argument("--val-frac", type=float, default=d["val_frac"],
                    help="validation slice of the TRAINING rows (0 = none)")
    ap.add_argument("--patience", type=int, default=d["patience"], help="early stopping on val_loss (default off)")
    ap.add_argument("--fit-n", type=int, default=None, help="training rows (default: all; else labelled a subsample)")
    ap.add_argument("--test-n", type=int, default=None, help="evaluate on a seeded test subset (fallback)")
    ap.add_argument("--verbose", type=int, default=2, help="Keras verbosity (2 = one line per epoch)")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline; output to a temp dir")
    a = ap.parse_args(argv)
    a.tag = ""
    if a.smoke:
        a.fit_n, a.test_n, a.epochs = SMOKE["fit_n"], SMOKE["test_n"], SMOKE["epochs"]
        a.tag = "_smoke"
        a.results_dir = a.results_dir or tempfile.mkdtemp(prefix="har_mlp_smoke_")
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
