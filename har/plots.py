"""Confusion-matrix + validation-curve figures. OWNER: Person 1 (frozen after kickoff).

    python -m har.plots --cm logreg_full logreg_reduced \
                        --curve logreg_full:C:log logreg_reduced:C:log

--cm <stem>              reads results/confusion/<stem>.csv  -> results/figures/<stem>_cm.png
--curve <stem>:<param>[:log]
                         reads results/metrics/<stem>_cv.csv -> results/figures/<stem>_valcurve.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from har.config import ACTIVITY_NAMES, LABEL_ORDER, RESULTS_DIR


def plot_confusion(stem: str, results_dir: Path = RESULTS_DIR) -> Path:
    cm = pd.read_csv(Path(results_dir) / "confusion" / f"{stem}.csv", index_col=0).to_numpy(float)
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)   # row-normalised
    names = [ACTIVITY_NAMES[c] for c in LABEL_ORDER]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{stem}: row-normalised confusion")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = Path(results_dir) / "figures" / f"{stem}_cm.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_curve(stem: str, param: str, log: bool = False, results_dir: Path = RESULTS_DIR) -> Path:
    df = pd.read_csv(Path(results_dir) / "metrics" / f"{stem}_cv.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = df.groupby("kernel") if "kernel" in df.columns else [(None, df)]
    for name, g in groups:
        g = g.sort_values(param)
        label = f"{name} " if name else ""
        if "train_mean" in g:
            ax.plot(g[param], g["train_mean"], "--", marker="o", label=f"{label}train")
        ax.plot(g[param], g["val_mean"], marker="o", label=f"{label}validation")
        ax.fill_between(g[param], g["val_mean"] - g["val_std"], g["val_mean"] + g["val_std"], alpha=0.15)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(param); ax.set_ylabel("accuracy (5-fold CV, training rows)")
    ax.set_title(stem); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(results_dir) / "figures" / f"{stem}_valcurve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cm", nargs="*", default=[], metavar="STEM")
    ap.add_argument("--curve", nargs="*", default=[], metavar="STEM:PARAM[:log]")
    a = ap.parse_args()
    for stem in a.cm:
        print("wrote", plot_confusion(stem))
    for spec in a.curve:
        parts = spec.split(":")
        print("wrote", plot_curve(parts[0], parts[1], log=len(parts) > 2 and parts[2] == "log"))


if __name__ == "__main__":
    main()
