"""Data loading / cleaning tests. OWNER: Person 1. They use FAKE files: only the real-data
audit (scripts/audit_raw.py) proves the loader is right for the real PAMAP2 files."""
import numpy as np
import pandas as pd

from har.config import FEATURE_SETS, FORBIDDEN_FEATURES, LABEL_ORDER, RAW_COLUMNS
from har.data import (KEEP_RAW, build_dataset, clean_subject, fill_heart_rate, get_xy,
                      load_subject, rows_fingerprint)
from har.synthetic import fake_raw_subject, write_dat, write_fake_dat_files


def test_load_keeps_first_row_and_needed_columns(tmp_path):
    raw = fake_raw_subject(500)
    p = write_dat(raw, tmp_path / "subject101.dat")
    df = load_subject(p)
    assert len(df) == 500                                   # header=None: no row swallowed as header
    assert abs(df["timestamp"].iloc[0] - raw["timestamp"].iloc[0]) < 1e-6
    assert set(df.columns) == set(KEEP_RAW)                 # unused columns never loaded
    assert not any(c.endswith(("acc6_x", "ori_0")) for c in df.columns)


def test_feature_sets_sizes_and_no_leakage():
    red, full = FEATURE_SETS["reduced"], FEATURE_SETS["full"]
    assert len(red) == 11 and len(full) == 31
    assert set(red) < set(full)
    assert not FORBIDDEN_FEATURES & set(full)
    assert not [c for c in full if "acc6" in c or "ori_" in c]
    assert all(c in RAW_COLUMNS for c in full)


def _hr_frame(values):
    return pd.DataFrame({"timestamp": np.arange(len(values)) * 0.01, "heart_rate": values})


def test_linear_interpolation_values_and_boundaries():
    nan = np.nan
    df = _hr_frame([nan, nan, 60, nan, nan, nan, 100, nan, nan])
    out = fill_heart_rate(df, "linear")["heart_rate"].to_numpy()
    assert np.isnan(out[:2]).all() and np.isnan(out[7:]).all()       # no extrapolation at the edges
    np.testing.assert_allclose(out[2:7], [60, 70, 80, 90, 100], rtol=1e-6)
    ff = fill_heart_rate(df, "ffill")["heart_rate"].to_numpy()
    assert np.isnan(ff[:2]).all()
    np.testing.assert_allclose(ff[2:], [60, 60, 60, 60, 100, 100, 100])
    try:
        fill_heart_rate(df, "cubic"); assert False
    except ValueError:
        pass


def test_clean_subject_policy():
    raw = fake_raw_subject(2000)
    clean, log = clean_subject(raw, 101)
    assert set(clean["activity_id"].unique()) <= set(LABEL_ORDER)    # label 0 gone
    assert not clean[FEATURE_SETS["full"]].isna().any().any()        # no NaN in features
    assert log["rows_raw"] == 2000
    assert log["rows_label0"] == int((raw["activity_id"] == 0).sum())
    assert log["rows_clean"] == len(clean)
    assert log["rows_after_label_filter"] - log["rows_dropped_nan"] == log["rows_clean"]
    assert (clean["subject_id"] == 101).all()


def test_build_dataset_and_get_xy(tmp_path):
    write_fake_dat_files(tmp_path, n_subjects=3, rows=2000)
    df, log = build_dataset(tmp_path)
    assert log["rows_total"] == len(df) == sum(s["rows_clean"] for s in log["subjects"])
    assert list(df["row_id"]) == list(range(len(df)))
    assert df["subject_id"].nunique() == 3
    for fs, d in (("full", 31), ("reduced", 11)):
        X, y = get_xy(df, fs)
        assert X.shape == (len(df), d) and X.dtype == np.float32 and y.dtype == np.int64
        assert np.isfinite(X).all()
        assert set(np.unique(y)) <= set(LABEL_ORDER)
    X, y = get_xy(df, "reduced", np.array([0, 5, 7]))
    assert X.shape == (3, 11) and len(y) == 3
    assert rows_fingerprint(df) == rows_fingerprint(build_dataset(tmp_path)[0])


def test_interpolation_does_not_cross_subjects(tmp_path):
    a, b = fake_raw_subject(1500, seed=1, with_nans=False), fake_raw_subject(1500, seed=2, with_nans=False)
    a["activity_id"] = 1; b["activity_id"] = 1                       # keep every row
    a["heart_rate"] = np.nan; b["heart_rate"] = np.nan
    a.loc[100, "heart_rate"] = 60.0; a.loc[200, "heart_rate"] = 70.0  # A: valid readings only in rows 100..200
    b.loc[900, "heart_rate"] = 200.0; b.loc[1000, "heart_rate"] = 210.0
    write_dat(a, tmp_path / "subject101.dat"); write_dat(b, tmp_path / "subject102.dat")
    df, _ = build_dataset(tmp_path)
    da, db = df[df.subject_id == 101], df[df.subject_id == 102]
    assert len(da) == 101 and len(db) == 101                          # only the 'inside' rows survive
    assert da["heart_rate"].between(60, 70).all()                     # never pulled towards 200
    assert db["heart_rate"].between(200, 210).all()
