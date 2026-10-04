# Decisions and deviations log - Person 1

One line per decision. Add a new row for every surprise found in the audit (P1-2).
IDs D-01..D-10 follow the playbook's Section 2.3; D-11 onwards are P1-specific.

| ID | Topic | Paper / source says | We do | Why / likely effect |
|---|---|---|---|---|
| D-01 | Number of classes | Report: 12 activities; poster: "18 IDs"; authors' code: 25 outputs | 12 protocol labels {1,2,3,4,5,6,7,12,13,16,17,24}; label 0 dropped | Report is primary; the poster's 18 is the ID count defined for the whole dataset |
| D-02 | Heart-rate filling | Report: linear interpolation; authors' code: forward-fill | Linear, per subject, on the full time series before dropping label 0; no extrapolation at edges (rows dropped) | Report is primary; `--hr-method ffill` kept for optional X2 |
| D-03 | Other missing values | Not described | Drop rows with NaN in any of the 31 full-set columns, so both feature sets use identical rows | Matches authors' code; affects a tiny share of rows (see `results/cleaning_log.json`) |
| D-04 | Feature columns | Report: 52 features; Fig. 1: n = 12-52 | reduced = 11, full = 31 (temp, +-16 g acc, gyro, mag per IMU + HR). Never +-6 g acc or orientation | From authors' feature_indices; orientation is documented as invalid, +-6 g saturates |
| D-05 | SVM training size | "~10,000 SVs = ~8% of training data" implies ~125k rows; authors' cap 175k | Tune on 20k rows, fit on 50k rows (Stage A) | Kernel SVM cost grows ~quadratically; expect lower accuracy than the paper |
| D-05b | SVM chosen C | Report text gives C = 1000 (reduced) but SVs quoted for C = 100 | Report CV-chosen C next to the paper values | Paper inconsistency; not a claim about their results |
| D-07 | LR loss | Eq. (1) is written as squared error + L2 | Library logistic regression (log-loss + L2) | Eq. (1) treated as a typo; authors' code uses sklearn LogisticRegression(solver='sag') |
| D-09 | Scaling | Report: LR, SVM, NN; poster: LR, SVM | StandardScaler inside the pipeline for LR and SVM (refit in every CV fold) | Prevents leakage; trees get raw values |
| D-10 | Other metrics | Intro names dataset size and speed | Record fit/predict seconds and support-vector count ourselves | No paper numbers exist to compare against |
| D-11 | File reading | Authors' loader used pandas default header | `header=None` so the first line of each file is kept | Authors' version silently loses one row per file |
| D-12 | Split | Random 85/15, no seed in authors' code | Seed 229, indices saved, fingerprint committed; also a rows fingerprint (subject, timestamp, label) | Split fingerprint alone depends only on the row count, so we also hash which rows survived cleaning |
| D-13 | Runner extension | n/a | `run_experiment(post_fit=...)` merges extra keys (e.g. `n_support_vectors`) into the JSON | Backward-compatible optional argument |
| D-14 | SVM CV | Report shows train + validation curves (Fig. 3) | `cv_grid` records train scores too and limits libsvm cache per worker (`--cache-mb`) | Needed for Fig.-3-style plots; avoids 5 x 2 GB memory use |
| D-15 | Log of label-0 rows | n/a | `rows_label0` counted on raw rows for both HR methods | Original appendix code counted it after dropping in the ffill branch (always 0) |
| D-16 | LR tuning size | Report does not state | CV on 200k training rows (`--tune-n`), final fit on all training rows | Compute; log actual rows in JSON |
| D-17 | Random split optimism | Report uses plain random row split | Reproduced as is; limitation stated in the report | Neighbouring 10 ms rows land in both train and test, inflating accuracy |

## Audit surprises (fill in after `python -m scripts.audit_raw`)

| Date | What the audit showed | Action |
|---|---|---|
| | | |
