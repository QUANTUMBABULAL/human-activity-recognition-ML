"""OPTIONAL, run by hand: export one already-reported model for the dashboard's prediction demo.

    python -m app.export_demo_model --model decision_tree --feature-set full

The project saved metrics but no fitted models, so the demo has nothing to load. This script
refits ONE existing FULL configuration with the exact hyperparameters recorded in its result
JSON (results/metrics/<model>_<fs>.json), on the frozen training split, with SEED = 229. It then
predicts the frozen test split and saves the model ONLY IF the predictions reproduce the saved
result: identical confusion matrix (results/confusion/<stem>.csv, when it exists) and identical
test accuracy. Otherwise nothing is written.

What it never does: tune anything, write to results/, change the split, or run on dashboard start.
Output: artifacts/<stem>.joblib + artifacts/<stem>_metadata.json.

The saved object is the fitted estimator exactly as the experiment used it. Decision trees and
random forests are scale-invariant and were trained on raw features (no scaler, D-09), so the
estimator alone is everything needed for prediction; logreg is saved as built by its builder.

Approximate cost on the project laptop (from the recorded fit_seconds): decision_tree full ~2 min,
logreg full ~1 min; random_forest full ~23 min and a ~1 GB+ artifact, with no saved confusion
matrix to check against (accuracy check only).
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.data_layer import METADATA_SUFFIX, MODELS_DIR
from har.config import ACTIVITY_NAMES, FEATURE_SETS, LABEL_ORDER, RESULTS_DIR, SEED
from har.data import get_xy, load_processed
from har.metrics import confusion_df, evaluate
from har.models_p1 import logreg
from har.models_p2 import decision_tree, random_forest
from har.splits import load_split, split_fingerprint

# model -> (builder, recorded-param names the builder accepts)
BUILDERS = {
    "decision_tree": (decision_tree.build, ("max_depth", "criterion", "min_samples_leaf")),
    "logreg": (logreg.build, ("C", "max_iter")),
    "random_forest": (random_forest.build, ("n_estimators", "max_depth", "max_features", "criterion")),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=sorted(BUILDERS), default="decision_tree")
    ap.add_argument("--feature-set", choices=("full", "reduced"), default="full")
    a = ap.parse_args(argv)

    from experiments import compare_results as cr
    stem = f"{a.model}_{a.feature_set}"
    rec = json.loads((RESULTS_DIR / "metrics" / f"{stem}.json").read_text())
    status = cr.classify(rec, "", cr.load_split_counts(RESULTS_DIR))
    if status != "FULL":
        raise SystemExit(f"{stem} has status {status}; only FULL results can be exported")

    build, names = BUILDERS[a.model]
    params = {k: rec["params"][k] for k in names if k in rec["params"]}
    print(f"[{stem}] refitting with recorded params {params}")

    df, split = load_processed(), load_split()
    print("split fingerprint", split_fingerprint(split))
    X_tr, y_tr = get_xy(df, a.feature_set, split["train"])
    X_te, y_te = get_xy(df, a.feature_set, split["test"])
    del df
    model = build(**params).fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc = evaluate(y_te, y_pred)["accuracy"]
    ok_acc = math.isclose(acc, rec["test_accuracy"], rel_tol=0, abs_tol=1e-12)
    cm_path = RESULTS_DIR / "confusion" / f"{stem}.csv"
    ok_cm = None
    if cm_path.is_file():
        saved = pd.read_csv(cm_path, index_col=0)
        ok_cm = bool(np.array_equal(confusion_df(y_te, y_pred).to_numpy(), saved.to_numpy()))
    print(f"test accuracy {acc:.10f} vs recorded {rec['test_accuracy']:.10f} -> "
          f"{'MATCH' if ok_acc else 'DIFFERENT'}; confusion matrix: "
          f"{'not saved' if ok_cm is None else ('MATCH' if ok_cm else 'DIFFERENT')}")
    if not ok_acc or ok_cm is False:
        raise SystemExit("refit does not reproduce the reported result (e.g. different library "
                         "version); nothing was written")

    import joblib
    import sklearn
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"{stem}.joblib", compress=3)
    meta = {"stem": stem, "model": a.model, "feature_set": a.feature_set,
            "feature_count": len(FEATURE_SETS[a.feature_set]),
            "features": FEATURE_SETS[a.feature_set], "seed": SEED,
            "estimator": type(model).__name__,
            "training": {"rows": int(len(y_tr)), "split": "frozen train split (all rows)",
                         "preprocessing": ("StandardScaler inside the saved Pipeline"
                                           if hasattr(model, "steps") else
                                           "none (raw features; trees are scale-invariant)")},
            "class_labels": {int(c): ACTIVITY_NAMES[int(c)] for c in model.classes_},
            "source_result": f"results/metrics/{stem}.json", "params": params,
            "verified": {"test_accuracy": acc, "accuracy_matches": ok_acc,
                         "confusion_matrix_matches": ok_cm, "n_test": int(len(y_te))},
            "split_fingerprint": split_fingerprint(split), "sklearn": sklearn.__version__,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    assert [int(c) for c in model.classes_] == LABEL_ORDER
    (MODELS_DIR / f"{stem}{METADATA_SUFFIX}").write_text(json.dumps(meta, indent=2))
    print(f"wrote {MODELS_DIR / (stem + '.joblib')}")


if __name__ == "__main__":
    main()
