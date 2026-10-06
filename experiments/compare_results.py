"""Final comparison of every saved result. OWNER: Person 2.

    python -m experiments.compare_results                     # read results/metrics/*.json
    python -m experiments.compare_results --results-dir DIR --no-write

Reads only result JSON files written by har.metrics.save_result (never the dataset, never trains
anything) and prints:
  1. OUR RESULTS: one row per non-smoke result file + one NOT RUN row per missing (model, feature set)
  2. rankings by accuracy / macro F1 / weighted F1 (FULL-data runs, then FULL / SUBSAMPLE runs,
     then QUICK runs in a separate pool of their own)
  3. full (31) vs reduced (11) features per model, only where both results exist
  4. training / prediction time
  5. PAPER REFERENCE (Athens et al. 2018, report text) - a separate table, never our numbers
Writes results/comparison.csv (our results only; missing metrics are empty cells, never 0) unless
--no-write or there is no real result yet.

Status of a result file:
  SMOKE      tag "_smoke" or notes starting "SMOKE TEST" -> excluded from every table and ranking
  QUICK      --quick run (JSON "status": "QUICK", tag "_quick" or notes starting "QUICK"): a fixed
             reduced-compute educational preset (P2-21). Shown in the table and timings, ranked only
             against other QUICK runs, never in FULL rankings, full-vs-reduced or OURS vs PAPER,
             whatever its row counts
  FULL       n_train / n_test equal the frozen split's counts in results/split_meta.json
  SUBSAMPLE  fewer training or test rows than the frozen split ("reproduced on a subsample")
  UNVERIFIED split_meta.json missing, so FULL cannot be checked (never ranked as FULL)
  MALFORMED  unreadable JSON or missing required keys (listed under problems, no metrics shown)
  NOT RUN    no result file for an expected (model, feature set)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from har.config import RESULTS_DIR
from har.metrics import REQUIRED_KEYS
from har.models_p1 import logreg, svm
from har.models_p2 import adaboost, decision_tree, mlp, random_forest

# file stem -> display name, in report order; PAPER_TEST_ACC comes from each model module
MODELS = {"logreg": "Logistic Regression", "svm_rbf": "SVM (RBF)", "decision_tree": "Decision Tree",
          "random_forest": "Random Forest", "adaboost": "AdaBoost", "mlp": "MLP"}
FEATURE_SETS = ("reduced", "full")
PAPER_TEST_ACC = {"logreg": logreg.PAPER_TEST_ACC, "svm_rbf": svm.PAPER_TEST_ACC,
                  "decision_tree": decision_tree.PAPER_TEST_ACC,
                  "random_forest": random_forest.PAPER_TEST_ACC, "adaboost": adaboost.PAPER_TEST_ACC,
                  "mlp": mlp.PAPER_TEST_ACC}
PAPER_NOTES = {("mlp", "reduced"): "report text 81.4; Figure 2 shows 84.0 (docs/paper_numbers.md)"}
MAIN_PROTOCOL = "random_85_15"
METRICS = (("test_accuracy", "Accuracy"), ("test_macro_f1", "Macro F1"),
           ("test_weighted_f1", "Weighted F1"))
CSV_COLUMNS = ["model", "model_name", "feature_set", "status", "stem", "primary", "protocol",
               "test_accuracy", "test_macro_f1", "test_weighted_f1", "train_accuracy",
               "fit_seconds", "predict_seconds", "n_train", "n_test", "params", "notes", "source"]
MISSING = "-"   # how a missing value is shown in the terminal (CSV: empty cell)
QUICK_TAG = "_quick"
NON_QUICK = ["FULL", "SUBSAMPLE", "UNVERIFIED"]


# ----------------------------------------------------------------------------- loading

def _num(v):
    """float, or None for missing / non-numeric / NaN values. Never turns missing into 0."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _split_stem(stem: str):
    """'<model>_<feature_set><tag>' -> (model, feature_set, tag); (None, None, None) if no match."""
    for fs in FEATURE_SETS:
        marker = f"_{fs}"
        i = stem.find(marker)
        if i > 0:
            return stem[:i], fs, stem[i + len(marker):]
    return None, None, None


def load_split_counts(results_dir) -> dict | None:
    p = Path(results_dir) / "split_meta.json"
    try:
        meta = json.loads(p.read_text())
        return {"n_train": int(meta["n_train"]), "n_test": int(meta["n_test"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def classify(rec: dict, tag: str, split_counts: dict | None) -> str:
    if "smoke" in tag.lower() or str(rec.get("notes", "")).upper().startswith("SMOKE TEST"):
        return "SMOKE"
    if (rec.get("status") == "QUICK" or "quick" in tag.lower()
            or str(rec.get("notes", "")).upper().startswith("QUICK")):
        return "QUICK"                      # checked BEFORE row counts: a quick run can use all rows
    if split_counts is None:
        return "UNVERIFIED"
    n_train, n_test = _num(rec.get("n_train")), _num(rec.get("n_test"))
    if n_train is None or n_test is None:
        return "UNVERIFIED"
    if n_train >= split_counts["n_train"] and n_test >= split_counts["n_test"]:
        return "FULL"
    return "SUBSAMPLE"


def discover(results_dir=RESULTS_DIR):
    """Read results_dir/metrics/*.json. Returns (rows, problems); never raises on a bad file."""
    results_dir = Path(results_dir)
    split_counts = load_split_counts(results_dir)
    rows, problems = [], []
    for path in sorted((results_dir / "metrics").glob("*.json")):
        stem = path.stem
        try:
            rec = json.loads(path.read_text())
            if not isinstance(rec, dict):
                raise ValueError("top level is not a JSON object")
            missing = REQUIRED_KEYS - rec.keys()
            if missing:
                raise ValueError(f"missing required keys {sorted(missing)}")
        except (OSError, ValueError) as e:            # JSONDecodeError is a ValueError
            model, fs, _ = _split_stem(stem)
            problems.append({"source": str(path), "stem": stem, "model": model, "feature_set": fs,
                             "error": str(e)})
            continue
        model, fs = str(rec["model"]), str(rec["feature_set"])
        prefix = f"{model}_{fs}"
        tag = stem[len(prefix):] if stem.startswith(prefix) else stem
        status = classify(rec, tag, split_counts)
        rows.append({
            "model": model, "model_name": MODELS.get(model, model), "feature_set": fs,
            "status": status, "stem": stem,
            # primary = the canonical file of its mode: <model>_<fs> or, for QUICK, <model>_<fs>_quick
            "primary": stem == prefix or (status == "QUICK" and tag == QUICK_TAG),
            "protocol": rec.get("protocol"),
            **{k: _num(rec.get(k)) for k in ("test_accuracy", "test_macro_f1", "test_weighted_f1",
                                             "train_accuracy", "fit_seconds", "predict_seconds")},
            "n_train": _num(rec.get("n_train")), "n_test": _num(rec.get("n_test")),
            "params": json.dumps(rec.get("params"), sort_keys=True, default=str),
            "notes": rec.get("notes", ""), "source": str(path)})
    return rows, problems


def build_table(rows, problems) -> pd.DataFrame:
    """Non-smoke result rows + a NOT RUN / MALFORMED row for every expected pair without one."""
    real = [r for r in rows if r["status"] != "SMOKE"]
    have = {(r["model"], r["feature_set"]) for r in real if r["primary"]}
    bad = {(p["model"], p["feature_set"]) for p in problems}
    out = list(real)
    for m in MODELS:
        for fs in FEATURE_SETS:
            if (m, fs) not in have:
                out.append({"model": m, "model_name": MODELS[m], "feature_set": fs,
                            "status": "MALFORMED" if (m, fs) in bad else "NOT RUN",
                            "stem": f"{m}_{fs}", "primary": True})
    df = pd.DataFrame(out).reindex(columns=CSV_COLUMNS)
    order = {m: i for i, m in enumerate(MODELS)}
    df["_m"] = df["model"].map(order).fillna(len(order))
    df["_f"] = df["feature_set"].map({fs: i for i, fs in enumerate(FEATURE_SETS)}).fillna(9)
    df = df.sort_values(["_m", "_f", "primary", "stem"], ascending=[True, True, False, True])
    return df.drop(columns=["_m", "_f"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- analysis

def has_result(df: pd.DataFrame) -> pd.Series:
    return df["status"].isin(NON_QUICK + ["QUICK"])


def rankable(df: pd.DataFrame, statuses) -> pd.DataFrame:
    """Primary result rows of the main protocol with the given statuses."""
    return df[df["status"].isin(statuses) & df["primary"].astype(bool) & (df["protocol"] == MAIN_PROTOCOL)]


def rank(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Competition ranking (1, 1, 3) on the stored full-precision value; missing values are not ranked."""
    d = df[df[metric].notna()].copy()
    if d.empty:
        return d
    d["rank"] = d[metric].rank(method="min", ascending=False).astype(int)
    tied = d["rank"].duplicated(keep=False)
    d["rank_label"] = d["rank"].astype(str) + tied.map({True: "=", False: ""})
    return d.sort_values(["rank", "model", "feature_set"])


def feature_set_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """full - reduced per model, only when BOTH primary results exist (main protocol)."""
    d = rankable(df, NON_QUICK)               # QUICK never enters this comparison
    out = []
    for m in MODELS:
        r = d[(d["model"] == m) & (d["feature_set"] == "reduced")]
        f = d[(d["model"] == m) & (d["feature_set"] == "full")]
        if r.empty or f.empty:
            out.append({"model_name": MODELS[m], "basis": "cannot compare: "
                        + ", ".join(fs for fs, x in (("reduced", r), ("full", f)) if x.empty) + " missing"})
            continue
        r, f = r.iloc[0], f.iloc[0]
        row = {"model_name": MODELS[m],
               "basis": "both FULL" if r["status"] == f["status"] == "FULL"
               else f"reduced {r['status']} / full {f['status']} (not a full-data comparison)"}
        for k, label in METRICS:
            row[f"{label} (full - reduced)"] = (f[k] - r[k]) if pd.notna(f[k]) and pd.notna(r[k]) else None
        out.append(row)
    return pd.DataFrame(out)


def paper_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(paper reference table, paper-vs-ours gap table). Paper columns are labelled PAPER."""
    ref = pd.DataFrame([{"model_name": MODELS[m], "feature_set": fs,
                         "PAPER test accuracy": PAPER_TEST_ACC[m].get(fs),
                         "PAPER note": PAPER_NOTES.get((m, fs), "")}
                        for m in MODELS for fs in FEATURE_SETS])
    ours = rankable(df, NON_QUICK)            # QUICK runs are not paper reproductions
    gap = []
    for _, r in ours.iterrows():
        p = PAPER_TEST_ACC.get(r["model"], {}).get(r["feature_set"])
        if p is None or pd.isna(r["test_accuracy"]):
            continue
        gap.append({"model_name": r["model_name"], "feature_set": r["feature_set"],
                    "OURS test accuracy": r["test_accuracy"], "OURS status": r["status"],
                    "PAPER test accuracy": p, "OURS - PAPER": r["test_accuracy"] - p})
    return ref, pd.DataFrame(gap)


# ----------------------------------------------------------------------------- printing

def _fmt(v, kind="metric"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return MISSING
    if kind == "metric":
        return f"{v:.4f}"
    if kind == "delta":
        return f"{v:+.4f}"
    if kind == "seconds":
        return f"{v:,.2f}"
    if kind == "int":
        return f"{int(v):,}"
    return str(v)


def _print(title, df, formats=None):
    print(f"\n=== {title} ===")
    if df is None or df.empty:
        print("(none)")
        return
    d = df.copy()
    for col, kind in (formats or {}).items():
        if col in d:
            d[col] = d[col].map(lambda v, k=kind: _fmt(v, k))
    print(d.to_string(index=False))


def report(df: pd.DataFrame, rows, problems):
    real = df[has_result(df)]
    smoke = [r for r in rows if r["status"] == "SMOKE"]
    print("OUR RESULTS (from results/metrics/*.json; metrics are fractions, times in seconds)")
    if real.empty:
        print("\nNo final experiment results found: no final experiments have been run yet.")
    elif (real["status"] == "QUICK").all():
        print("\nOnly QUICK (reduced-compute educational) results so far: no full-scale experiment has "
              "been run yet.")
    main_cols = {"model_name": "Model", "feature_set": "Feature Set", "test_accuracy": "Accuracy",
                 "test_macro_f1": "Macro F1", "test_weighted_f1": "Weighted F1",
                 "fit_seconds": "Train Time (s)", "predict_seconds": "Predict Time (s)",
                 "n_train": "Train Rows", "n_test": "Test Rows", "status": "Status"}
    t = df.assign(status=df["status"] + df["primary"].astype(bool).map({True: "", False: " (extra run)"}))
    t = t[list(main_cols)].rename(columns=main_cols)
    _print("Comparison table (missing = '-', never 0)", t,
           {"Accuracy": "metric", "Macro F1": "metric", "Weighted F1": "metric",
            "Train Time (s)": "seconds", "Predict Time (s)": "seconds",
            "Train Rows": "int", "Test Rows": "int"})
    other = real[real["protocol"] != MAIN_PROTOCOL]
    if not other.empty:
        print(f"Note: {len(other)} result(s) use a protocol other than {MAIN_PROTOCOL}; "
              "they are excluded from rankings and comparisons.")

    for label, statuses in (("FULL-data runs only", ["FULL"]),
                            ("FULL + SUBSAMPLE / UNVERIFIED runs (no QUICK)", NON_QUICK),
                            ("QUICK runs only (reduced-compute educational preset, NOT comparable to "
                             "FULL)", ["QUICK"])):
        pool = rankable(df, statuses)
        for k, name in METRICS:
            r = rank(pool, k)
            if not r.empty:
                r = r[["rank_label", "model_name", "feature_set", k, "status"]].rename(
                    columns={"rank_label": "Rank", "model_name": "Model", "feature_set": "Feature Set",
                             k: name, "status": "Status"})
            _print(f"Ranking by {name} - {label} ('=' marks a tie)", r, {name: "metric"})

    fsc = feature_set_comparison(df)
    deltas = [c for c in fsc.columns if c.endswith("(full - reduced)")]
    _print("Full (31) vs reduced (11) features - OURS, positive = full is better", fsc,
           {c: "delta" for c in deltas})
    if deltas:
        both = fsc[fsc["basis"] == "both FULL"]
        acc = "Accuracy (full - reduced)"
        if not both.empty:
            n_better = int((both[acc] > 0).sum())
            print(f"Full features give higher test accuracy for {n_better} of {len(both)} model(s) "
                  "with FULL-data results on both sets.")
        else:
            print("No model has FULL-data results on both feature sets yet: no full-vs-reduced claim.")

    if not real.empty:
        tm = real.sort_values("fit_seconds", na_position="last")[
            ["model_name", "feature_set", "fit_seconds", "predict_seconds", "n_train", "n_test", "status"]]
        _print("Training / prediction time - OURS (depends on machine, rows and preset used; compare "
               "SUBSAMPLE and QUICK timings with care)", tm,
               {"fit_seconds": "seconds", "predict_seconds": "seconds", "n_train": "int", "n_test": "int"})

    ref, gap = paper_table(df)
    _print("PAPER REFERENCE - Athens et al. 2018, report-text test accuracy (NOT our results)", ref,
           {"PAPER test accuracy": "metric"})
    print("The paper reports accuracy only: no paper macro F1, weighted F1 or timings exist.")
    print("QUICK results are never compared with the paper: they are not paper reproductions.")
    _print("OURS vs PAPER test accuracy (only where we have a result)", gap,
           {"OURS test accuracy": "metric", "PAPER test accuracy": "metric", "OURS - PAPER": "delta"})

    missing = df[df["status"] == "NOT RUN"]
    print(f"\nMissing configurations (NOT RUN): {len(missing)} of {len(MODELS) * len(FEATURE_SETS)}")
    for _, r in missing.iterrows():
        print(f"  - {r['model_name']} / {r['feature_set']}")
    prim = df[df["primary"].astype(bool)]
    non_quick = set(zip(*[prim.loc[prim["status"].isin(NON_QUICK), c] for c in ("model", "feature_set")]))
    quick_only = prim[(prim["status"] == "QUICK")
                      & ~pd.Series(list(zip(prim["model"], prim["feature_set"])), index=prim.index).isin(non_quick)]
    if not quick_only.empty:
        print(f"\nOnly a QUICK result (full-scale experiment not run): {len(quick_only)}")
        for _, r in quick_only.iterrows():
            print(f"  - {r['model_name']} / {r['feature_set']}")
    if smoke:
        print(f"\nExcluded SMOKE result files ({len(smoke)}; never part of the comparison):")
        for r in smoke:
            print(f"  - {r['source']}")
    if problems:
        print(f"\nProblems ({len(problems)} unreadable result file(s), not used):")
        for p in problems:
            print(f"  - {p['source']}: {p['error']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out", default=None, help="CSV path (default: <results-dir>/comparison.csv)")
    ap.add_argument("--no-write", action="store_true", help="print only, write no file")
    a = ap.parse_args(argv)

    rows, problems = discover(a.results_dir)
    df = build_table(rows, problems)
    report(df, rows, problems)

    out = Path(a.out or Path(a.results_dir) / "comparison.csv")
    if a.no_write:
        print("\n--no-write: no file written.")
    elif not has_result(df).any():
        print(f"\nNo real results, so {out} was not written.")
    else:
        df.to_csv(out, index=False)          # NaN -> empty cell, never 0
        print(f"\nWrote {out}")
    return df


if __name__ == "__main__":
    main()
