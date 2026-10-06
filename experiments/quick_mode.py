"""Shared helpers for the --quick experiment mode (decision P2-21, docs/decisions_p2.md).

--quick is a FIXED, computationally reduced preset for learning and algorithm comparison under
limited compute. It is NOT a paper reproduction and never counts as a FULL result:
  - same frozen dataset and frozen train/test split; test rows are still only used inside
    run_experiment, and nothing is tuned on them (quick mode tunes nothing at all);
  - the result file gets the tag "_quick" (never overwrites <model>_<fs>.json);
  - the JSON records status = "QUICK", the actual preset and that tuning was skipped;
  - notes start with NOTE, so the status is visible even without the "status" field.
"""
from __future__ import annotations

import copy

STATUS = "QUICK"
TAG = "_quick"
NOTE = ("QUICK - resource-constrained educational run (fixed reduced-compute preset, "
        "hyperparameter tuning skipped); NOT a paper reproduction and NOT comparable to FULL results")
TUNING = "skipped (quick mode: fixed preset, no cross-validation, no search)"


def add_argument(ap):
    ap.add_argument("--quick", action="store_true",
                    help="cheap fixed preset for educational comparison (status QUICK, tag _quick); "
                         "not a paper reproduction")


def check_args(ap, a, locked, argv=None):
    """--quick is a fixed preset: refuse flags that would silently change it (and --smoke).

    A flag counts if it was given at all, even with its default value (e.g. --epochs 100).
    """
    if not a.quick:
        return
    if getattr(a, "smoke", False):
        ap.error("--quick and --smoke cannot be combined")
    unset = object()                               # non-string, so argparse never type-converts it
    probe = copy.deepcopy(ap)
    probe.set_defaults(**{n: unset for n in locked})
    given = vars(probe.parse_args(argv))
    bad = [n for n in locked if given[n] is not unset]
    if bad:
        ap.error("--quick is a fixed preset and cannot be combined with: "
                 + ", ".join("--" + n.replace("_", "-") for n in bad))


def cv_info() -> dict:
    return {"performed": False, "rule": "quick preset (no CV)", "tuning": TUNING}


def record(config: dict) -> dict:
    """Extra keys merged into the result JSON (via run_experiment(post_fit=...))."""
    return {"status": STATUS, "experiment_mode": "quick", "quick_config": dict(config),
            "hyperparameter_tuning": TUNING, "quick_note": NOTE}
