"""Task P1-2: audit the 9 raw PAMAP2 Protocol files -> docs/data_audit.md. OWNER: Person 1.

    python -m scripts.audit_raw [--raw-dir PATH] [--out docs/data_audit.md]

Checks (each gets PASS/FAIL/INFO in the report): 9 files, 54 columns, timestamps strictly
increasing, no duplicated rows, only labels {0} + the 12 protocol IDs, heart-rate missing
fraction near 0.9, IMU missing fractions small. Also writes per-subject activity counts
and the total number of protocol-labelled rows (report says "about 1.9 million").
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from har.config import (ACTIVITY_NAMES, DOCS_DIR, FEATURE_SETS, LABEL_ORDER, RAW_COLUMNS,
                        RAW_DIR, TRANSIENT_LABEL)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def audit_file(path: Path) -> dict:
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype="float64")
    n_cols = df.shape[1]
    out = {"file": path.name, "rows": len(df), "n_cols": n_cols}
    if n_cols != 54:
        out["error"] = f"expected 54 columns, found {n_cols}"
        return out
    df.columns = RAW_COLUMNS
    ts = df["timestamp"].to_numpy()
    out["ts_strictly_increasing"] = bool(np.all(np.diff(ts) > 0))
    out["ts_non_decreasing"] = bool(np.all(np.diff(ts) >= 0))
    out["duration_s"] = float(ts[-1] - ts[0])
    out["median_dt_s"] = float(np.median(np.diff(ts)))
    out["dup_rows"] = int(df.duplicated().sum())
    out["dup_timestamps"] = int(df["timestamp"].duplicated().sum())
    labels = df["activity_id"].astype(int)
    out["labels_present"] = sorted(set(labels.unique()))
    out["unexpected_labels"] = sorted(set(labels.unique()) - set(LABEL_ORDER) - {TRANSIENT_LABEL})
    out["hr_nan_frac"] = float(df["heart_rate"].isna().mean())
    imu_cols = [c for c in FEATURE_SETS["full"] if c != "heart_rate"]
    nan_frac = df[imu_cols].isna().mean()
    out["imu_nan_frac_max"] = float(nan_frac.max())
    out["imu_nan_frac_max_col"] = str(nan_frac.idxmax())
    out["imu_nan_rows_any"] = int(df[imu_cols].isna().any(axis=1).sum())
    counts = labels.value_counts()
    out["counts"] = {int(k): int(v) for k, v in counts.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--out", default=str(DOCS_DIR / "data_audit.md"))
    a = ap.parse_args()

    files = sorted(Path(a.raw_dir).glob("subject1*.dat"))
    if not files:
        raise SystemExit(f"No subject1*.dat files in {a.raw_dir}. See README 'Where the data lives'.")
    res = []
    for f in files:
        print("auditing", f.name, flush=True)
        res.append(audit_file(f))
    bad = [r for r in res if "error" in r]
    if bad:
        raise SystemExit(f"column-count problem: {bad}")

    summary = pd.DataFrame([{
        "file": r["file"], "rows": f"{r['rows']:,}", "cols": r["n_cols"],
        "t strictly incr.": r["ts_strictly_increasing"], "median dt (s)": round(r["median_dt_s"], 4),
        "dup rows": r["dup_rows"], "HR missing": f"{r['hr_nan_frac']:.3f}",
        "worst IMU col missing": f"{r['imu_nan_frac_max']:.4f} ({r['imu_nan_frac_max_col']})",
        "rows w/ any IMU NaN": f"{r['imu_nan_rows_any']:,}",
    } for r in res])

    act = pd.DataFrame({r["file"].replace(".dat", ""): pd.Series(r["counts"]) for r in res}).fillna(0).astype(int)
    act = act.reindex([TRANSIENT_LABEL] + LABEL_ORDER).fillna(0).astype(int)
    act.insert(0, "activity", ["(transient)"] + [ACTIVITY_NAMES[c] for c in LABEL_ORDER])
    act.insert(0, "id", act.index)
    act["total"] = act.drop(columns=["id", "activity"]).sum(axis=1)

    total_rows = sum(r["rows"] for r in res)
    protocol_rows = int(act.loc[LABEL_ORDER, "total"].sum())
    all12 = [r["file"].replace(".dat", "") for r in res if set(LABEL_ORDER) <= set(r["counts"])]
    unexpected = sorted({l for r in res for l in r["unexpected_labels"]})
    hr_mean = float(np.mean([r["hr_nan_frac"] for r in res]))
    imu_worst = max(r["imu_nan_frac_max"] for r in res)

    def status(ok): return "PASS" if ok else "**FAIL - investigate**"
    checks = pd.DataFrame([
        {"check": "9 files found", "result": status(len(res) == 9), "detail": f"{len(res)} files"},
        {"check": "every file has 54 columns", "result": status(all(r['n_cols'] == 54 for r in res)), "detail": ""},
        {"check": "timestamps strictly increase", "result": status(all(r['ts_strictly_increasing'] for r in res)),
         "detail": "; ".join(r['file'] for r in res if not r['ts_strictly_increasing']) or "all files"},
        {"check": "no duplicated rows", "result": status(all(r['dup_rows'] == 0 for r in res)),
         "detail": f"{sum(r['dup_rows'] for r in res)} duplicated rows in total"},
        {"check": "only label 0 + the 12 protocol IDs", "result": status(not unexpected),
         "detail": f"unexpected: {unexpected}" if unexpected else "none"},
        {"check": "heart-rate missing fraction near 0.9", "result": status(0.8 < hr_mean < 0.95),
         "detail": f"mean over subjects {hr_mean:.3f}"},
        {"check": "IMU missing fractions small (<2% per column)", "result": status(imu_worst < 0.02),
         "detail": f"worst column fraction {imu_worst:.4f}"},
        {"check": "sampling interval about 0.01 s", "result": status(all(abs(r['median_dt_s'] - 0.01) < 0.002 for r in res)),
         "detail": f"median dt {np.median([r['median_dt_s'] for r in res]):.4f} s"},
        {"check": "rows with a protocol label vs report (~1.9 M)", "result": "INFO",
         "detail": f"{protocol_rows:,} rows ({protocol_rows / 1.9e6:.2f} x 1.9M); total raw rows {total_rows:,}"},
    ])

    md = ["# Data audit (P1-2)", "",
          "Generated by `python -m scripts.audit_raw`. **Do not hand-edit the tables.** "
          "Add interpretation under *Notes* and surprises to `docs/decisions_p1.md`.", "",
          "## Checks", md_table(checks), "",
          "## Per-file summary", md_table(summary), "",
          "## Rows per activity per subject (raw, before cleaning)", md_table(act), "",
          f"Subjects that performed all 12 protocol activities: {', '.join(all12) or 'none'} "
          "(use these as test subjects for the optional subject-wise extension X1).", "",
          "## Manual checks still to do (script cannot do these)", "",
          "- [ ] Open the dataset's own readme and confirm the column order in `har/config.py` "
          "(timestamp, activity ID, heart rate, then hand / chest / ankle blocks of 17: temperature, "
          "+-16 g acc, +-6 g acc, gyro, magnetometer, orientation).",
          "- [ ] Compare the protocol-labelled row count above with the report's 'about 1.9 million'; "
          "if it is far off, stop and investigate.",
          "- [ ] Write every surprise as a new line in `docs/decisions_p1.md`.", "",
          "## Notes", "", "_(write your interpretation here)_", ""]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(md_table(checks))
    print("wrote", out)


if __name__ == "__main__":
    main()
