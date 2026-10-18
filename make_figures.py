"""Produce the results table and figures used in the report from results/*.jsonl.

    python make_figures.py --tag ETTh1_H24 --out ../report/figures
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
COLORS = {"linear": "#2a78d6", "transformer": "#eb6834", "naive": "#52514e"}
LABELS = {"linear": "Linear", "transformer": "Patch Transformer", "naive": "Seasonal naive"}


def load(tag):
    return [json.loads(l) for l in open(RESULTS / f"{tag}.jsonl")]


def table(recs):
    """Print a LaTeX-ready summary (mean over seeds) of test MAE / RMSE."""
    agg = defaultdict(list)
    for r in recs:
        agg[(r["model"], r["frac"])].append(r)
    print("model & frac & windows & params & epochs & time(s) & MAE & RMSE")
    for (m, f), rs in sorted(agg.items()):
        mae = np.mean([r["test"]["mae"] for r in rs])
        rmse = np.mean([r["test"]["rmse"] for r in rs])
        print(f"{m} & {int(f*100)}% & {rs[0]['n_train_windows']} & {rs[0]['params']} & "
              f"{np.mean([r['epochs'] for r in rs]):.0f} & "
              f"{np.mean([r['train_seconds'] for r in rs]):.1f} & {mae:.3f} & {rmse:.3f}")


def learning_curve(recs, out):
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for m in ["linear", "transformer", "naive"]:
        rs = sorted([r for r in recs if r["model"] == m], key=lambda r: r["frac"])
        x = [100 * r["frac"] for r in rs]
        y = [r["test"]["mae"] for r in rs]
        if m == "naive":
            ax.plot(x, y, "--", color=COLORS[m], lw=1.2, label=LABELS[m])
        else:
            ax.plot(x, y, "-o", color=COLORS[m], lw=2, ms=4, label=LABELS[m])
    ax.set_xticks([25, 50, 100])
    ax.set_xticklabels(["25%", "50%", "100%"])
    ax.set_xlabel("Training history used (fraction of 12 months)")
    ax.set_ylabel("Test MAE (°C)")
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.66))
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "learning_curve_ETTh1_H24.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ETTh1_H24")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = load(args.tag)
    table(recs)
    learning_curve(recs, out)


if __name__ == "__main__":
    main()
