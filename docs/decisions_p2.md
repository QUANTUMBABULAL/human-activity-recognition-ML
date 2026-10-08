# Decisions and deviations log - Person 2

P2 owns the tree ensembles and the neural network. This file has two parts: the architecture
contract P2 code follows (so it plugs into P1's frozen infrastructure without changes), and the
decisions/deviations table, filled in as real runs happen.

## 1. Ownership

| Owner | Files |
|---|---|
| **P2** | `har/models_p2/*`, `experiments/p2/*`, `experiments/compare_results.py`, `experiments/quick_mode.py`, `tests/test_p2_models.py`, `tests/test_compare_results.py`, `tests/test_quick_mode.py`, `docs/decisions_p2.md` |
| P1 (frozen, read-only for P2) | `har/{config,metrics,runner,synthetic,plots}.py` |
| P1 (used, not changed) | `har/{data,splits}.py`, `data/processed/*`, `results/split_meta.json` |

A change to a frozen P1 file needs a PR both people approve, and the reason is written here first.

## 2. Model modules (`har/models_p2/`)

| Module | Experiment | Paper setting (`docs/paper_numbers.md`) | Scaling |
|---|---|---|---|
| `decision_tree.py` | E4 | Gini, max depth 15 (one-standard-deviation rule) | none (raw values, D-09) |
| `random_forest.py` | E5 | 100 trees, sqrt(features) per split, max depth 20 | none |
| `adaboost.py` | E6 | default learning rate; 500 trees / depth 10 (reduced), 250 / depth 9 (full) | none |
| `mlp.py` | E7 | 512-512 ReLU, dropout 0.5, softmax, categorical cross-entropy, SGD | StandardScaler fit on training rows only |

Each module mirrors `har/models_p1/logreg.py` / `svm.py`:

- `build(**hyperparams)` returns an unfitted object with `.fit(X, y)` and `.predict(X)` - an
  sklearn estimator/Pipeline, or for the MLP a small `KerasMLP` wrapper class (the name
  `har/runner.py` already expects).
- `PAPER` - the paper's hyperparameters; `PAPER_TEST_ACC = {"full": ..., "reduced": ...}` copied
  from the report-text column of `docs/paper_numbers.md` (fractions, e.g. `0.927`).
- a grid constant (e.g. `DEPTH_GRID`) and a CV helper that returns `(best, rows)` like
  `svm.cv_grid`, using `KFold(N_FOLDS, shuffle=True, random_state=SEED)` on training rows only.
- `random_state=SEED` on every estimator; `n_jobs` exposed as an argument.

## 3. Experiment scripts (`experiments/p2/`)

`run_decision_tree.py`, `run_random_forest.py`, `run_adaboost.py`, `run_mlp.py`, plus
`experiments/compare_results.py` (reads every `results/metrics/*.json` into one table next to
`paper_test_accuracy`). Each script follows `experiments/p1/run_logreg.py`:

```
--feature-set {full,reduced,both}  --tune-n  --fit-n  --test-n  --skip-tune  --n-jobs  --results-dir
```

1. `df, split = load_processed(), load_split()`; print `split_fingerprint(split)`.
2. `tune_idx = subsample(split["train"], a.tune_n)`; CV on `get_xy(df, fs, tune_idx)` only.
3. Write the CV table to `results/metrics/<model>_<fs>_cv.csv`.
4. `run_experiment(model_name=..., owner="P2", model=build(best), df=df, split=split, ...)`.

## 4. How models consume X / y

- `get_xy(df, feature_set, idx)` -> `X float32 [n, d]`, `y int64 [n]`. `y` holds the **original
  activity IDs** `{1,2,3,4,5,6,7,12,13,16,17,24}` (`LABEL_ORDER`), not 0..11.
- `predict` must return the same IDs: `metrics.confusion_df` and `macro_f1` index by
  `LABEL_ORDER`, so 0..11 predictions would silently score wrong. sklearn trees do this
  automatically; the MLP wrapper must encode labels to 0..11 for Keras and decode `argmax` back
  with the encoding learnt in `fit`.
- Columns come only from `FEATURE_SETS`; `get_xy` asserts no `FORBIDDEN_FEATURES`
  (`timestamp`, `activity_id`, `subject_id`, `row_id`).

## 5. Feature sets

| Name | d | Columns |
|---|---|---|
| `reduced` ("limited" in the paper) | 11 | heart rate + hand IMU (temp, +-16 g acc, gyro, mag) |
| `full` | 31 | heart rate + hand, chest, ankle IMUs |

Both sets use identical rows (D-03). Every P2 model is run on both, giving
`<model>_reduced.json` and `<model>_full.json`.

## 6. Train/test split reuse

P2 never creates a split. It loads `data/processed/split_random_seed229.npz` with `load_split()`.
Verified on P2's machine (2026-10-05):

| Check | Value |
|---|---|
| cleaned rows | 1,921,420 (`row_id` = 0..n-1) |
| train / test rows | 1,633,207 / 288,213, disjoint, cover all rows |
| SPLIT FINGERPRINT | `204cf31f6f415437` (matches `results/split_meta.json`) |
| ROWS FINGERPRINT | `46d008ac842d2c8f` (matches `results/split_meta.json`) |
| classes | all 12 in both train and test |

If either fingerprint differs on any machine, stop and rebuild with `python -m scripts.build_dataset`
before running anything.

Subsampling for compute goes through `har.runner.subsample` (seeded, deterministic). Subsampled
fits are reported as such in `notes` and logged in section 9.

## 7. How metrics are saved

`run_experiment` fits, times, predicts and calls `metrics.save_result`, which writes:

- `results/metrics/<model>_<feature_set>[tag].json` with `REQUIRED_KEYS` (model, owner, feature_set,
  protocol, seed, n_train, n_test, train/test accuracy, test macro-F1, params, fit/predict seconds,
  paper_test_accuracy, notes) plus `cv`, `versions`, `timestamp_utc`.
- `results/confusion/<model>_<feature_set>.csv` (12 x 12, `LABEL_ORDER`).

Model names (file stems): `decision_tree`, `random_forest`, `adaboost`, `mlp`. Extra facts
(e.g. tree depth reached, MLP epochs run) go through `post_fit=lambda m: {...}` (D-13).
Figures: `python -m har.plots --cm <stem> --curve <stem>:<param>[:log]`; `plot_curve` needs
columns `<param>, val_mean, val_std` (+ `train_mean`) in the `_cv.csv`.

## 8. Rules

**Reproducibility.** `SEED = 229` for every estimator, `KFold` and subsample. MLP: call
`keras.utils.set_random_seed(SEED)` before building; record that GPU/oneDNN runs may still differ
slightly. `versions()` is saved in every JSON automatically. Commands used for each reported
number are written in section 9.

**No test-set tuning.** Hyperparameters, depth, number of trees, epochs and early stopping are
chosen only from CV/validation folds inside `split["train"]`. MLP early stopping uses a validation
slice of the training rows, never test rows. `split["test"]` appears only inside `run_experiment`;
`tests/test_p2_models.py` enforces this for `experiments/p2/run_*.py`. Each final model is evaluated
on the test set once; no re-running with new settings after looking at test accuracy.

**Honest numbers.** Every accuracy, F1, time or count in the report comes from a JSON written by a
real run on the real processed data. Numbers from `har/synthetic.py` (used in tests) are never
reported. A model not run on full data is reported as "reproduced on a subsample" or "not
reproduced (deferred: reason)".

## 9. Testing strategy

`tests/test_p2_models.py` (fast, synthetic data only, no accuracy claims about PAMAP2):

- now: the runner contract with a P2 owner/name (JSON keys, 12 x 12 confusion, original label
  IDs), P2 package imports, and the no-test-rows check on `experiments/p2/run_*.py`.
- per model, once implemented: `build()` fits/predicts on `fake_processed_df`, predictions are
  a subset of `LABEL_ORDER`, determinism (same seed -> same predictions), CV helper returns a
  grid value and a table with `val_mean/val_std`, MLP test auto-skips without TensorFlow.
- Synthetic data is separable, so "better than chance" is a smoke check only, not a result.

Run: `python -m pytest -q`.

## 10. Decisions / deviations

Fill in one row per decision as runs happen (continue P1's numbering style with a `P2-` prefix).

| ID | Topic | Paper / source says | We do | Why / likely effect |
|---|---|---|---|---|
| P2-01 | Infrastructure | n/a | Reuse P1's `get_xy`, `load_split`, `run_experiment`, `save_result` unchanged | One split and one metrics format for both people |
| P2-02 | DT depth selection | Gini, depth 15 by the "one-standard-deviation rule" | 5-fold CV over `DEPTH_GRID` = {2,4,6,8,10,12,15,18,20,25,30}; shallowest depth within 1 SD of the best mean val. accuracy (`--rule one_sd`, default). Max-val depth also stored as `cv.best_by_max_val` | Paper rule; the grid range is our choice (report does not give one) |
| P2-03 | DT tuning size | Report does not state | CV on 200k training rows (`--tune-n`), final fit on all 1,633,207 training rows | Compute; actual rows logged in JSON (`cv.tune_rows`, `n_train`) |
| P2-04 | DT other settings | Not stated | `min_samples_leaf=1`, no pruning, no class weights, `random_state=229`, raw (unscaled) features | sklearn defaults; trees are scale-invariant |
| P2-05 | RF tuning | 100 trees, sqrt features, max depth 20 (selection method not stated) | 5-fold CV over `max_depth` in {10,15,20,25,30} at 100 trees, sqrt features; highest mean val. accuracy (`--rule max`). `--trees-grid 50 100 200` widens the search if time allows | Checks the paper's depth instead of assuming it; tree count kept at the paper value by default for compute |
| P2-06 | RF tuning size | Not stated | CV on 200k training rows (`--tune-n`), final fit on all 1,633,207 training rows | Probe (10 trees, 400k rows): 2-6 MB per tree, so 100 trees on full data is roughly 1-1.5 GB (extrapolated). CV folds run sequentially to keep only one forest in memory |
| P2-07 | RF determinism | n/a | `random_state=229`, `bootstrap=True`; trees and predictions identical for any `n_jobs`; `predict_proba` can differ by ~1e-16 between `n_jobs` values (threaded summation order) | Verified in `tests/test_p2_models.py` |
| P2-08 | AdaBoost API / algorithm | "default learning rate" (2018 sklearn: `base_estimator=`, default `algorithm="SAMME.R"`) | sklearn 1.9.1 `AdaBoostClassifier(estimator=DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=229), n_estimators=n, learning_rate=1.0, random_state=229)`. Discrete **SAMME** only | `base_estimator` was removed in sklearn 1.4 and `algorithm`/SAMME.R in 1.6, so the paper's likely SAMME.R cannot be reproduced on the installed version. SAMME uses hard tree votes rather than class probabilities, which may change accuracy relative to the paper (effect unknown until run) |
| P2-09 | AdaBoost tuning | Chosen 500 trees / depth 10 (reduced), 250 trees / depth 9 (full); selection method not stated | 5-fold CV over `max_depth` in {6,8,9,10,12} x `n_estimators` in {50,100,250,500}, `learning_rate` fixed at 1.0; highest mean val. accuracy (`--rule max`, same as RF); `--rule one_sd` picks the fewest trees then shallowest within 1 SD. Paper configs are in the grid and stay available via `--skip-tune` (logged as "paper value") | Checks the paper's choice instead of assuming it is optimal. CV may pick a different config per feature set; whatever it picks is reported, not overridden |
| P2-10 | AdaBoost CV implementation | n/a | One 500-tree fit per (depth, fold); counts 50/100/250 are scored with `staged_predict`. Folds run in parallel (`--n-jobs`); AdaBoost itself has no `n_jobs` | The first k trees of an n-tree AdaBoost fit are exactly a k-tree fit (same seeded RNG sequence), checked in `test_adaboost_staged_scores_match_a_fresh_smaller_fit`. Cuts CV cost about 4x versus refitting each count |
| P2-11 | AdaBoost tuning size | Not stated | CV on **100k** training rows (`--tune-n`, half the DT/RF default), final fit on all 1,633,207 training rows | Compute (see section 11). Actual rows logged in JSON (`cv.tune_rows`, `n_train`) |
| P2-12 | AdaBoost early stopping / determinism | n/a | sklearn stops boosting if a tree has zero weighted error (or is no better than chance). The JSON records `n_trees_fitted`, `stopped_early` and `last_estimator_error`. `random_state=229` gives identical trees and predictions on reruns | So a "500-tree" result that actually used fewer trees is visible. Determinism checked in `tests/test_p2_models.py` |
| P2-13 | MLP architecture | 512-512 ReLU, dropout 0.5, softmax, categorical cross-entropy, SGD | `Input(d) -> Dense(512, relu) -> Dropout(0.5) -> Dense(512, relu) -> Dropout(0.5) -> Dense(12, softmax)`; loss `categorical_crossentropy` on one-hot targets; d = 11 (reduced) / 31 (full); 274,956 / 285,196 parameters | As reported. Dropout after each hidden layer is our reading of "dropout 0.5" (placement not stated). Architecture is fixed: no architecture search, so no CV helper/grid for the MLP (unlike section 2's per-model template) |
| P2-14 | MLP optimiser / batch | SGD; learning rate, momentum and batch size not stated | `SGD(learning_rate=0.01, momentum=0.0)`, `batch_size=32` | Keras defaults of the report's era (plain SGD, `fit` batch 32), the most likely unstated values. All three are CLI flags; any change is logged in `params` |
| P2-15 | MLP epochs / validation | "100 epochs shown for reduced set"; full-set epochs and validation scheme not stated | Fixed 100 epochs for both sets. Inside `KerasMLP.fit`, a seeded 10% slice of the rows passed to `fit` (training rows only) is held out as Keras `validation_data` for monitoring; per-epoch loss/accuracy saved to `results/metrics/mlp_<fs>_history.csv`. Early stopping off by default (`--patience N` turns it on, `restore_best_weights=True`, monitored on that slice). `--val-frac 0` fits on all training rows | Keeps the paper's fixed schedule while giving a test-free learning curve. Cost: the network sees 90% of training rows (1,469,886 of 1,633,207). The JSON's `n_train` is the rows given to `fit`; `n_fit_rows` / `n_val_rows` give the actual split |
| P2-16 | MLP scaling | Not stated for the MLP | `StandardScaler` fitted inside `KerasMLP.fit` on the fitting rows only (not the validation slice, never test); `predict` only calls `transform` | No leakage; tested (scaler mean == fitting-row mean, `n_samples_seen_` == fitting rows, predict on shifted data with `fit` patched to raise) |
| P2-17 | MLP labels | n/a | Fixed map from `LABEL_ORDER`: ID -> position 0..11 -> one-hot; prediction = `LABEL_ORDER[argmax]`. Unknown IDs (e.g. 0) raise. Output layer always has 12 units, even if a subsample lacks a class | Runner/metrics index by original IDs; tested that softmax index i decodes to `LABEL_ORDER[i]` |
| P2-18 | MLP reproducibility | n/a | `keras.utils.set_random_seed(229)` (Python, NumPy, TF) at the start of every `fit`, `Dropout(seed=229)`, and `tf.config.experimental.enable_op_determinism()` (process-wide). Same machine + versions (TF 2.21.0, Keras 3.15.1, CPU) -> bit-identical probabilities (tested) | Not guaranteed across machines, CPU instruction sets, thread counts or TF/oneDNN versions. TF on native Windows has no GPU support, so runs are CPU only |
| P2-19 | AdaBoost final selection rule | Chosen 500 trees / depth 10 (reduced), 250 / depth 9 (full); selection method not stated | Final real-data runs use `--rule one_sd`. The CV grid (depth {6,8,9,10,12} x trees {50,100,250,500}, 5 folds, 100k training rows) is unchanged. One-SD picks the cheapest config (fewest trees, then shallowest) whose mean val. accuracy is within one SD of the best. Selection uses CV on training rows only. `cv.best_by_max_val` still records the max-rule pick. Decided 2026-10-05, before any AdaBoost test evaluation | CV differences inside one SD are noise. Without this rule, noise could select a much more expensive config (e.g. 500 trees / depth 12 on `full`, final fit roughly 11 h, section 11) for no real gain. Same principle as the paper's DT rule (P2-02). Differs from RF's `max` rule (P2-05), where the tree count is fixed and cost is not the issue. SAMME instead of SAMME.R remains an unavoidable sklearn API deviation (P2-08) |
| P2-20 | MLP batch size / learning rate: pre-declared fallback | Paper states SGD only; no batch size or learning rate (P2-14) | Default plan: batch 32, learning rate 0.01, 100 epochs (P2-14/15). Before the final MLP runs, time one epoch on the training rows (no test evaluation). If the timing implies that 100 epochs would exceed **about 10 h per feature set**, both sets use the fallback: **batch 256, learning rate 0.08**, same architecture, same 100 epochs unless the timing/implementation requires otherwise. The fallback is a **computational deviation**. If used, it is recorded automatically in the result JSON's `params` (`batch_size`, `learning_rate`) and must be written here as a dated row with the measured epoch time. Decided 2026-10-05, before any MLP test evaluation | The trigger is runtime only, never test performance. Batch 256 at learning rate 0.01 would make 8x fewer updates (574,200 vs 4,593,400 over 100 epochs), roughly the progress of ~12-13 batch-32 epochs, so "100 epochs" would no longer mean the same training. Scaling the learning rate with the batch (x8 -> 0.08, a heuristic) keeps progress per epoch comparable. Gradient noise drops ~8x (less implicit regularisation). Both values remain our assumptions, not paper values |
| P2-21 | Resource-constrained educational experiment mode (`--quick`) | Paper configs: RF 100 trees / depth 20; AdaBoost 250-500 trees / depth 9-10 chosen from a CV grid; MLP 100 epochs; SVM C by 5-fold CV over kernels x C | A separate, fixed, cheap preset in each expensive script, next to the unchanged normal mode: **RF** reduced, 20 trees, depth 10, `n_jobs` 2, no CV; **AdaBoost** reduced, 20 trees, depth 6, learning rate 1.0, no CV grid; **MLP** reduced, same 512-512 network, 5 epochs, batch 256, learning rate 0.08 (the P2-20 fallback pair), 10% validation slice, no early stopping, no search; **SVM** Stage A, reduced, RBF with the paper's C (1000 reduced / 100 full), 30,000 seeded training rows, no CV. All fit once on the frozen training split (SVM: a seeded subset of it) and are evaluated once on the full frozen test split. Output `<model>_<fs>_quick.json`, JSON `status: "QUICK"`, `quick_config`, `hyperparameter_tuning: "skipped ..."`, notes starting `QUICK`. Decided 2026-10-06, before any RF / AdaBoost / MLP / SVM test evaluation | Project priority is learning and comparing algorithms, not maximising accuracy, and compute is limited (sections 11-12: full AdaBoost and MLP runs take hours per feature set on this 15 W laptop CPU). Effect: QUICK numbers are expected to be lower than, and are not comparable to, full or paper numbers. They are kept apart from FULL everywhere (section 14) |
| | | | | |

## 11. AdaBoost compute notes

Probe on real training rows (2026-10-05, this machine: Intel i5-1035G1, 15 W, 4 physical cores / 8 logical
threads; one weighted tree, single thread):

| Set / depth | 200k rows | 1,633,207 rows |
|---|---|---|
| reduced / 10 | 2.6 s | 22.5 s |
| full / 9 | 6.8 s | 59.0 s |

Extrapolations only (assumed linear scaling; not measured end to end):

- final fit at the paper configs: reduced 500 trees, about 3 h; full 250 trees, about 4 h. If CV picks 500 trees
  for `full`, about 8 h. If CV picks 500 trees at depth 12 for `full`, roughly 11 h (assumes cost grows
  linearly with depth: 59 s x 12/9 per tree). Trees are fitted sequentially, so `--n-jobs` does not help here.
- CV at the defaults (100k rows, 80k per training fold, 5 folds in parallel): roughly **1-1.5 h (reduced)** and
  **2.5-3.5 h (full)** for the 5-depth grid. Corrected 2026-10-05: an earlier estimate (45 min / 2 h) assumed
  5x speed-up from 5 parallel folds, but this machine has 4 physical cores. CV fits 12,500 trees on 80k rows per
  set, about as much total work as a 500-tree final fit on all rows.
- The probe timed single short fits at boost clock. Multi-hour runs on this 15 W CPU are likely to throttle, so
  all estimates here are lower bounds rather than expected values.
- Memory is small: depth <= 12 trees have <= 8191 nodes each.

Run one feature set at a time. If a full fit is not feasible, use `--fit-n` and report the result as "reproduced on
a subsample". Smoke run (`--smoke`, 3k CV rows, 10k fit rows, 5k test rows, temp dir) takes about 30 s.

## 12. MLP compute notes

Probe on real training rows (2026-10-05, this machine, CPU only, 1 epoch on 300k rows, no validation):

| Set | batch 32 (default) | batch 256 |
|---|---|---|
| full | 43.6 s | 10.7 s |
| reduced | 31.1 s | 14.9 s |

Extrapolations only (linear in rows; per-epoch validation and the final predict add a little; probe timings are
noisy, e.g. the first fit includes TF warm-up):

- 100 epochs on 1,469,886 fitting rows at batch 32: the linear extrapolation gives about **6 h (full)** and
  **4 h (reduced)**. The gap is probably a probe artifact (the full-set fit ran first and included TF warm-up).
  Reduced and full use the **same rows** (D-03: 1,469,886 fitting + 163,321 validation rows) and almost the
  **same parameter count**: 274,956 vs 285,196 (+3.7%, first layer only; the 512x512 layer, 262,656 parameters,
  dominates). So **both sets are expected to take similar time**, roughly 4-6 h each, about 8-12 h for
  `--feature-set both`, before validation overhead and throttling.
- Updates: batch 32 -> 45,934 per epoch, 4,593,400 over 100 epochs; batch 256 -> 5,742 per epoch, 574,200 total.
  Batch 32 is slow because per-step overhead dominates the arithmetic: measured ~4.65 ms/step and ~6.9k samples/s
  (full), against ~9.1 ms/step and ~28k samples/s at batch 256.
- **Validation uses the training batch size**: `KerasMLP.fit` passes no `validation_batch_size`, so Keras
  evaluates the 163,321-row validation slice at batch 32, i.e. 5,104 forward-only steps after every epoch (638 at
  batch 256). The probe above ran without validation. Inference: this adds roughly 10-20% per epoch at batch 32.
- The one-epoch timing check and fallback rule before the final runs are in P2-20.
- At `--batch-size 256` about 1.5-2 h per set, but that is a deviation from P2-14 and changes the optimisation
  (8x fewer SGD updates per epoch at the same learning rate), so it must be logged as such.
- Memory: about 0.5 GB for the full-set arrays (X, scaled X, one-hot Y); the network is about 1 MB.

Smoke run (`--smoke`: real data and frozen split, 20k fit rows, 5k test rows, 2 epochs, temp dir) takes about 20 s.
A run with `--fit-n` is labelled "reproduced on a subsample" in `notes`.

## 13. Final comparison (`experiments/compare_results.py`)

```
python -m experiments.compare_results                    # prints, writes results/comparison.csv
python -m experiments.compare_results --no-write         # print only
python -m experiments.compare_results --results-dir DIR  # another results tree (tests use temp dirs)
```

**What it reads.** Only `results/metrics/*.json` written by `har.metrics.save_result`, plus
`results/split_meta.json` for the frozen split's row counts. It never loads PAMAP2, never imports
TensorFlow and never fits a model. Model and feature set come from the JSON's own `model` /
`feature_set` fields. The file stem `<model>_<feature_set>` is the **primary** result for that
pair; `<model>_<feature_set>_quick` is the primary QUICK result for it (both can exist and are
shown as separate rows). A tagged stem such as `svm_rbf_full_stageC` is listed as an "extra run" and is not ranked.
`*_cv.csv`, `*_history.csv` and other non-JSON files are ignored.

**Expected grid.** 6 models (`logreg`, `svm_rbf`, `decision_tree`, `random_forest`, `adaboost`,
`mlp`) x 2 feature sets (`reduced`, `full`) = 12 rows, always printed.

**Status of each row.**

| Status | Rule | Used in |
|---|---|---|
| FULL | `n_train` >= 1,633,207 and `n_test` >= 288,213 (counts read from `split_meta.json`) | table, both rankings, full-vs-reduced, gap vs paper |
| SUBSAMPLE | fewer training or test rows (e.g. SVM Stage A fits 50k rows, or any `--fit-n` / `--test-n` run) | table, the "all non-smoke" ranking only; labelled "reproduced on a subsample" |
| UNVERIFIED | `split_meta.json` missing or row counts missing, so FULL cannot be checked | as SUBSAMPLE; never counted as FULL |
| QUICK | JSON `status` is `"QUICK"`, tag contains `quick`, or notes start with `QUICK` (P2-21). Checked **before** row counts, so a quick run on all rows is still QUICK | table, timing table and CSV; ranked **only** in a separate "QUICK runs only" pool; never in the FULL ranking, the FULL + SUBSAMPLE ranking, full-vs-reduced or OURS vs PAPER. A pair with only a QUICK result is listed under "Only a QUICK result (full-scale experiment not run)" |
| SMOKE | tag contains `smoke` or notes start with `SMOKE TEST` | **nothing**: listed under "Excluded SMOKE result files" only, not in the CSV |
| MALFORMED | invalid JSON, not an object, or missing a `REQUIRED_KEYS` field | listed under "Problems"; the pair shows MALFORMED with no metrics |
| NOT RUN | no usable primary file for the pair | table and CSV with empty metric cells |

The MLP holds out 10% of the training rows for validation inside `fit` (P2-15). Its `n_train` is
still the full training count, so it can be FULL; `n_fit_rows` in its JSON gives the exact number.

**Missing values.** A missing, `null`, non-numeric or NaN metric stays missing: `-` in the
terminal, an empty cell in the CSV. It is never 0, never ranked and never used in a difference.
`comparison.csv` is not written when there is no FULL / SUBSAMPLE / UNVERIFIED result, so no file
with only placeholders appears in `results/`.

**Rankings.** Accuracy, macro F1 and weighted F1 are ranked separately, highest first, using
competition ranking (1, 1, 3). Ranks use the stored full-precision values and ties are marked
`=`. Two pools are printed: FULL-data runs only, then all non-smoke runs with each row's status.
Only primary rows of the main protocol (`random_85_15`) are ranked.

**Full vs reduced.** For each model, `full - reduced` is shown for the three metrics only when
both primary results exist. The basis column says "both FULL" or names the statuses (then it is
not a full-data comparison). The one-line summary ("full is better for k of m models") counts
only "both FULL" models. With no such pair it says that no claim can be made.

**Timing.** `fit_seconds` / `predict_seconds` come from `run_experiment` (wall clock, this machine,
on the rows actually used). They are shown with `n_train` / `n_test` because SUBSAMPLE timings
are not comparable to full-data timings. Smoke timings never appear.

**Paper vs ours.** Paper numbers are the report-text test accuracies in each model module's
`PAPER_TEST_ACC` (copied from `docs/paper_numbers.md`; MLP reduced = 81.4 text, 84.0 in Figure 2
is noted). They appear only in two separately titled sections: "PAPER REFERENCE ... (NOT our
results)" and "OURS vs PAPER", whose columns are prefixed `PAPER` / `OURS`. The paper gives no
F1 or timing values, so none are shown. `comparison.csv` contains our results only.

Tests: `tests/test_compare_results.py` uses hand-made JSON fixtures in temp directories (written
with `save_result` so the format is real). Their numbers are test inputs, never results.

## 14. Resource-constrained educational experiment mode (`--quick`, P2-21)

**Why it exists.** The goal of this project is to learn how these algorithms work and to compare
them, not to get the highest accuracy. Our compute is limited: one 15 W laptop CPU (section 11), and
TensorFlow on native Windows has no GPU (P2-18). The full paper-style experiments are expensive. A
full AdaBoost CV grid plus final fit takes several hours per feature set (section 11), and 100 MLP
epochs at batch 32 take about 4-6 h per feature set (section 12). `--quick` gives every expensive
model one cheap run, so all six algorithms can be compared on the same frozen data. The normal
mode is unchanged and is still the only way to get a FULL result.

**What it changes.** Only the amount of computation: fewer trees, fewer epochs, no
hyperparameter search, and for the SVM fewer training rows.

| Script | Quick preset | Normal mode |
|---|---|---|
| `experiments.p2.run_random_forest --quick` | reduced, 20 trees, depth 10, `n_jobs` 2, no CV, all training rows | 5-fold CV over depth {10..30} at 100 trees, then fit |
| `experiments.p2.run_adaboost --quick` | reduced, 20 trees, depth 6, learning rate 1.0, no CV, all training rows | 5-fold CV over depth {6..12} x trees {50,100,250,500}, then fit |
| `experiments.p2.run_mlp --quick` | reduced, 512-512, 5 epochs, batch 256, lr 0.08, 10% validation slice, no early stopping | 100 epochs, batch 32, lr 0.01 |
| `experiments.p1.run_svm --quick` | Stage A, reduced, RBF, paper C, 30,000 seeded training rows, no CV | CV over C on 20k rows, fit on 50k rows |

`--feature-set full` or `both` may be given explicitly with `--quick`, for an educational
full-vs-reduced comparison. Any other flag that would change the preset (for example `--epochs`,
`--max-depth`, `--fit-n`, `--skip-tune`, `--C` or `--smoke`) is refused with an error, even when it is
passed with its default value. So a QUICK result always means exactly this preset. `--n-jobs`
is allowed for RF because it does not change the trees (P2-07). Keras uses a GPU automatically if
TensorFlow sees one. The MLP JSON records `device` / `gpus` and the per-epoch `history`.

**What it does not change.**

- The frozen dataset, the rows fingerprint `46d008ac842d2c8f`, the frozen split
  (`204cf31f6f415437`) and the original label IDs.
- Test rows are still used only inside `run_experiment`, once, after the single fit. Nothing
  is tuned on them. Quick mode tunes nothing at all. Its settings were fixed before any quick test
  evaluation, and the SVM's C is the paper value, not one chosen by us.
- Model code (`har/models_p2/*`, `har/models_p1/svm.py`), the runner and the metrics format are
  unchanged. The MLP still fits its `StandardScaler` on the fitting rows only, and every estimator
  still uses `SEED = 229`.
- Existing result files. Quick results are written to `<model>_<fs>_quick.json`,
  `confusion/<model>_<fs>_quick.csv` and, for the MLP, `mlp_<fs>_quick_history.csv`, so they can
  never overwrite a FULL result.

**How QUICK results must be reported.** Report them as "QUICK (reduced-compute educational run)"
with their preset, never as a paper reproduction and never side by side with paper numbers as if
they matched. `compare_results.py` enforces this (section 13): QUICK is ranked only against QUICK,
and it is left out of the FULL rankings, the full-vs-reduced comparison and OURS vs PAPER. FULL and
QUICK results for the same model are always separate rows.

**Why reducing computation openly is more honest.** A 5-epoch MLP or a 20-tree forest is a
different experiment from the paper's 100-epoch / 100-tree configuration. It is not a noisier
version of the same experiment. Two dishonest options are easy. One is to run the cheap
configuration and report it under the paper's name. The other is to keep the paper's name and
silently cut epochs or trees. Either way, the number would hide its own cause. A lower accuracy
would look like a failed reproduction, or a model would look worse than it is. Naming the
reduction instead means three things:

- every QUICK number carries its real configuration (`params`, `quick_config`, `status`), so a
  reader knows exactly what was measured;
- comparisons stay like-for-like: QUICK against QUICK under one fixed preset, FULL against FULL and
  against the paper;
- the gap between a QUICK result and the paper is reported as a consequence of less computation. It
  is not evidence about the algorithm or about the paper's claim.

A smaller experiment that is labelled correctly is a valid result about that smaller experiment.
An unlabelled one would be a misleading result about the paper's experiment.
