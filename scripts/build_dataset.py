"""Task P1-3: build the cleaned dataset + frozen split. OWNER: Person 1.

    python -m scripts.build_dataset                      # main protocol (linear HR interpolation)
    python -m scripts.build_dataset --hr-method ffill    # optional extension X2 (separate files)

Writes (NOT committed): data/processed/pamap2_protocol_clean[_ffill].parquet,
                        data/processed/split_random_seed229[_ffill].npz
Writes (committed):     results/split_meta[_ffill].json, results/cleaning_log[_ffill].json
Both laptops must print the SAME split fingerprint AND rows fingerprint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from har.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from har.data import PROCESSED_FILE, build_dataset, rows_fingerprint
from har.splits import SPLIT_FILE, make_random_split, save_split, split_fingerprint, write_split_meta

PAPER_ROWS = 1.9e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--out-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--hr-method", choices=["linear", "ffill"], default="linear")
    a = ap.parse_args()
    suffix = "" if a.hr_method == "linear" else f"_{a.hr_method}"

    df, log = build_dataset(a.raw_dir, hr_method=a.hr_method)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / PROCESSED_FILE.replace(".parquet", f"{suffix}.parquet")
    df.to_parquet(pq, index=False)

    split = make_random_split(len(df))
    sp = save_split(split, out_dir / SPLIT_FILE.replace(".npz", f"{suffix}.npz"))
    rfp = rows_fingerprint(df)
    meta = write_split_meta(split, len(df), Path(a.results_dir) / f"split_meta{suffix}.json",
                            extra={"hr_method": a.hr_method, "rows_fingerprint": rfp})
    (Path(a.results_dir) / f"cleaning_log{suffix}.json").write_text(json.dumps(log, indent=2))

    t = pd.DataFrame(log["subjects"])
    t["share_dropped_%"] = (100 * (t["rows_raw"] - t["rows_clean"]) / t["rows_raw"]).round(1)
    print("\nPer-subject cleaning log (check that no subject loses an unexpected share):")
    print(t[["subject_id", "rows_raw", "rows_label0", "rows_dropped_nan", "rows_clean", "share_dropped_%"]]
          .to_string(index=False))
    print("\nClass counts:", log["class_counts"])
    ratio = len(df) / PAPER_ROWS
    print(f"\nCleaned rows: {len(df):,}  ({ratio:.2f} x the report's ~1.9 M)")
    if not 0.85 <= ratio <= 1.15:
        print("WARNING: row count differs from the report by >15%. Investigate and log in decisions_p1.md.")
    print(f"Train/test rows: {meta['n_train']:,} / {meta['n_test']:,}")
    print(f"\nSPLIT FINGERPRINT: {meta['fingerprint']}")
    print(f"ROWS  FINGERPRINT: {rfp}")
    print("-> both laptops must print these two values identically.")
    print("wrote", pq, "and", sp)


if __name__ == "__main__":
    main()
