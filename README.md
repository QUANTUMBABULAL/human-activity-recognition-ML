# PAMAP2 Human Activity Classification - reimplementation

Independent student reimplementation of Athens, Blum and Singh, *Human Activity Classification*,
CS229 report, Stanford, Fall 2018 (https://cs229.stanford.edu/proj2018/report/6.pdf).
We use their dataset, models and protocol; all code and results here are our own and every
deviation is listed in `docs/decisions_p1.md` / `docs/decisions_p2.md`.

Dataset: PAMAP2 (Reiss & Stricker, 2012), UCI: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

## Ownership
| Owner | Files |
|---|---|
| **P1** | `har/{config,data,splits,metrics,runner,synthetic,plots}.py`, `har/models_p1/*`, `scripts/*`, `experiments/p1/*`, `tests/test_{data,splits_leakage,integration,p1_models}.py`, `docs/{decisions_p1,data_audit,paper_numbers}.md`, README, requirements, .gitignore, pytest.ini |
| **P2** | `har/models_p2/*`, `experiments/p2/*`, `experiments/compare_results.py`, `tests/test_p2_models.py`, `docs/decisions_p2.md` |

Frozen after the Day-1 kickoff (change only by a PR both approve): `config.py, metrics.py, runner.py, synthetic.py, plots.py`.

## Setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate     # Windows: py -3.11 -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements.txt                           # P1 may skip tensorflow
python -m pytest -q
```

## Where the data lives
Put the 9 files at `data/raw/PAMAP2_Dataset/Protocol/subject101.dat ... subject109.dat`, or set
`HAR_RAW_DIR` (e.g. `export HAR_RAW_DIR=~/datasets/PAMAP2_Dataset/Protocol`). Data never goes into Git.

## Person 1 run order
```bash
# P1-2  audit (writes docs/data_audit.md) - read every PASS/FAIL line
python -m scripts.audit_raw
# P1-3  clean + split + fingerprints (writes data/processed/*, results/split_meta.json, results/cleaning_log.json)
python -m scripts.build_dataset
# P1-4  logistic regression (E1)
python -m experiments.p1.run_logreg --feature-set reduced
python -m experiments.p1.run_logreg --feature-set full
python -m har.plots --curve logreg_full:C:log logreg_reduced:C:log --cm logreg_full logreg_reduced
# P1-5  SVM timing probe first (see below), then Stage A overnight (E2)
python -m experiments.p1.run_svm --stage A --feature-set both > svm_stageA.log 2>&1
python -m har.plots --curve svm_rbf_full:C:log svm_rbf_reduced:C:log --cm svm_rbf_full svm_rbf_reduced
# P1-6  Stage B kernel comparison (E3, SHOULD)
python -m experiments.p1.run_svm --stage B --feature-set both --tune-n 10000
```

### SVM timing probe (run before any long job)
```python
import time
from har.data import get_xy, load_processed
from har.runner import subsample
from har.splits import load_split
from har.models_p1 import svm
df, split = load_processed(), load_split()
for n in (5_000, 10_000, 20_000):
    X, y = get_xy(df, "reduced", subsample(split["train"], n))
    t = time.perf_counter(); svm.build("rbf", C=100).fit(X, y)
    print(f"{n:>7,} rows: {time.perf_counter()-t:6.1f} s")   # kernel SVM: ~3-4x per doubling
```

### Fallbacks (log each in `docs/decisions_p1.md`)
| Problem | Flag |
|---|---|
| SVM CV too slow | `--tune-n 10000`, then `--skip-tune` (paper C), `--fit-n 20000` |
| SVM prediction too slow | `--test-n 50000` |
| LR ConvergenceWarning | `--final-max-iter 5000`; `--tune-n 100000` |
| Low RAM during SVM CV | `--n-jobs 2 --cache-mb 300` |

## Streamlit Dashboard
```bash
python -m streamlit run app/streamlit_app.py
```
The Streamlit dashboard visualizes existing experiment results and provides an interactive prediction
demo using held-out PAMAP2 samples. It does not retrain models.

Pages: Overview · Models · Confusion Matrix · Predict · About. Only completed experiments are shown.
The Predict page lists only models with a saved artifact in `artifacts/` (`<stem>.joblib` +
`<stem>_metadata.json`; the experiments themselves save none). The FULL Decision Tree artifact is
exported by hand with `python -m app.export_demo_model --model decision_tree --feature-set full`.
This refits the reported configuration on the frozen training split and saves it only if it
reproduces the saved result exactly. Tests: `python -m pytest tests/test_dashboard.py -q`.

## Checking both laptops use identical data
`build_dataset` prints `SPLIT FINGERPRINT` and `ROWS FINGERPRINT`. Both must match across laptops
(and match `results/split_meta.json` on main).

## Git workflow (P1)
```bash
git init -b main && git add . && git status      # must NOT list data/, .venv/, *.parquet
git commit -m "chore: project skeleton, shared interface, tests"
git remote add origin https://github.com/<P1-username>/pamap2-har-reimplementation.git && git push -u origin main
git switch -c p1/data-audit    # then p1/logreg, p1/svm; one PR per branch, P2 reviews
```

## Honesty rules
Every number you report comes from your runs on the real data. Never report numbers produced by
`har/synthetic.py`. Say "reproduced", "reproduced on a subsample", or "not reproduced (deferred: reason)".
