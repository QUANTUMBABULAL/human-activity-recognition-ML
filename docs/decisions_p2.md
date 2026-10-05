# Decisions and deviations log - Person 2

P2 owns the tree ensembles and the neural network. This file has two parts: the architecture
contract P2 code follows (so it plugs into P1's frozen infrastructure without changes), and the
decisions/deviations table, filled in as real runs happen.

## 1. Ownership

| Owner | Files |
|---|---|
| **P2** | `har/models_p2/*`, `experiments/p2/*`, `experiments/compare_results.py`, `tests/test_p2_models.py`, `docs/decisions_p2.md` |
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
| | | | | |

## 11. AdaBoost compute notes

Probe on real training rows (2026-10-05, this machine, 8 cores, one weighted tree, single thread):

| Set / depth | 200k rows | 1,633,207 rows |
|---|---|---|
| reduced / 10 | 2.6 s | 22.5 s |
| full / 9 | 6.8 s | 59.0 s |

Extrapolations only (assumed linear scaling; not measured end to end):

- final fit at the paper configs: reduced 500 trees, about 3 h; full 250 trees, about 4 h. If CV picks 500 trees
  for `full`, about 8 h. Trees are fitted sequentially, so `--n-jobs` does not help here.
- CV at the defaults (100k rows, 80k per training fold, 5 folds in parallel): roughly 45 min (reduced) and
  2 h (full) for the 5-depth grid.
- Memory is small: depth <= 12 trees have <= 8191 nodes each.

Run one feature set at a time. If a full fit is not feasible, use `--fit-n` and report the result as "reproduced on
a subsample". Smoke run (`--smoke`, 3k CV rows, 10k fit rows, 5k test rows, temp dir) takes about 30 s.
