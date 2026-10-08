"""E7: MLP (512-512 ReLU, dropout 0.5, softmax, categorical cross-entropy, SGD). OWNER: Person 2.

    python -m experiments.p2.run_mlp --feature-set both
    python -m experiments.p2.run_mlp --smoke          # quick pipeline check, NOT a result
    python -m experiments.p2.run_mlp --quick          # cheap educational run (status QUICK)

Options: --epochs 100, --batch-size 32, --learning-rate 0.01, --momentum 0.0,
         --val-frac 0.1 (seeded slice of the TRAINING rows, monitoring only; 0 = fit on all of them),
         --patience N (early stopping on that slice; default off), --fit-n (default: all training rows)
The architecture is fixed to the paper's (no architecture search). The scaler, label encoding and
validation slice all live inside KerasMLP.fit, which only ever receives training rows.
--smoke uses tiny row counts and 2 epochs and writes to a temp directory (never results/).
--quick (P2-21) is a fixed reduced-compute preset: reduced features, same 512-512 network, 5 epochs,
batch 256, learning rate 0.08 (the pre-declared P2-20 fallback pair), 10% validation slice of the
training rows, no early stopping, no search; all training rows, full test split. Keras uses a GPU
automatically when TensorFlow sees one; the device is recorded. Written to
results/metrics/mlp_<fs>_quick.json (+ mlp_<fs>_quick_history.csv) with status QUICK - never FULL.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from experiments import quick_mode
from har.config import RESULTS_DIR
from har.data import load_processed
from har.models_p2 import mlp
from har.runner import run_experiment
from har.splits import load_split, split_fingerprint

SMOKE = {"fit_n": 20_000, "test_n": 5_000, "epochs": 2}
QUICK = {"feature_set": "reduced", "epochs": 5, "batch_size": 256, "learning_rate": 0.08,
         "val_frac": 0.1, "patience": None}
QUICK_LOCKED = ("epochs", "batch_size", "learning_rate", "momentum", "val_frac", "patience",
                "fit_n", "test_n")


def device_info() -> dict:
    """Devices TensorFlow can see (Keras places the model on the first GPU automatically if any)."""
    try:
        import tensorflow as tf
        gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception as e:                       # recording the device must never break a run
        return {"device": "unknown", "device_error": repr(e)}
    return {"device": "GPU" if gpus else "CPU", "gpus": gpus}


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
    prefix = "SMOKE TEST - not a result; " if a.smoke else ""
    cv_info = {"selection": "none - paper architecture; validation slice from training rows used for "
                            "monitoring only"}
    post_fit = mlp.mlp_info
    if a.quick:
        config = {"feature_set": fs, **{k: v for k, v in QUICK.items() if k != "feature_set"},
                  "hidden_units": list(model.hidden_units), "test_rows": "full test split",
                  "fit_rows": "all training rows minus the seeded validation slice"}
        prefix, cv_info = f"{quick_mode.NOTE}; ", {**cv_info, **quick_mode.cv_info()}

        def post_fit(m):
            return {**mlp.mlp_info(m), **device_info(), "history": m.history_, **quick_mode.record(config)}
    rec = run_experiment(model_name="mlp", owner="P2", model=model, df=df, split=split, feature_set=fs,
                         params=model.get_config(), paper_test_accuracy=mlp.PAPER_TEST_ACC[fs],
                         cv_info=cv_info, fit_n=a.fit_n, test_n=a.test_n,
                         notes=f"{prefix}{fit_note}{val_note}{stop_note}{test_note}",
                         post_fit=post_fit, tag=a.tag, results_dir=a.results_dir)
    hist = pd.DataFrame(model.history_)
    hist.insert(0, "epoch", range(1, len(hist) + 1))
    hist.to_csv(Path(a.results_dir) / "metrics" / f"mlp_{fs}{a.tag}_history.csv", index=False)
    return rec


def parse_args(argv=None):
    d = mlp.DEFAULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["full", "reduced", "both"], default=None,
                    help="default: both (--quick: reduced)")
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
    quick_mode.add_argument(ap)
    a = ap.parse_args(argv)
    quick_mode.check_args(ap, a, QUICK_LOCKED, argv)
    a.feature_set = a.feature_set or (QUICK["feature_set"] if a.quick else "both")
    a.tag = ""
    if a.quick:
        a.epochs, a.batch_size, a.learning_rate = QUICK["epochs"], QUICK["batch_size"], QUICK["learning_rate"]
        a.val_frac, a.patience = QUICK["val_frac"], QUICK["patience"]
        a.tag = quick_mode.TAG
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
