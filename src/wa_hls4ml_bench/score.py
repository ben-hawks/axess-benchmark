#!/usr/bin/env python3
"""Score predictions against the wa-hls4ml benchmark's 6 regression targets.

Implements paper Section 3.2, Eq. 1-4 (Hawks et al., ACM TRETS 2026, doi:10.1145/3787490): R^2, SMAPE (epsilon = 1, the
smallest strictly positive value the integer resource/latency counts can take), RMSE, and
a per-target relative-percent-error (RPE) box plot.

    python -m wa_hls4ml_bench.score --pred predictions.csv --truth truth.csv --out results/

Both CSVs have a ``sample_id`` column plus BRAM, DSP, FF, LUT, cycles_max, interval_max.
If the truth CSV has a ``subset`` column, metrics are also broken down the way paper
Tables 4-5 are: dense / conv1d / conv2d for the test set, per architecture for the
exemplar set. Truth samples with no prediction are reported as missing coverage (a
complete submission predicts every sample) and are excluded from the metrics.

Outputs in --out: METRICS.md, metrics.json, rpe_boxplot.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["BRAM", "DSP", "FF", "LUT", "cycles_max", "interval_max"]
SMAPE_EPSILON = 1.0
SUBSET_GROUP = {"2_20": "dense", "2layer": "dense", "3layer": "dense", "latency": "dense",
                "resource": "dense", "conv1d": "conv1d", "conv2d": "conv2d"}


def r_squared(y_true, y_pred):
    """Eq. 1. NaN when y_true has zero variance (paper Table 5's "N/A" with footnote a)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)


def smape(y_true, y_pred, epsilon=SMAPE_EPSILON):
    """Eq. 2, in percent (0-200)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(200.0 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)))


def rmse(y_true, y_pred):
    """Eq. 3, native units."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def relative_percent_error(y_true, y_pred, epsilon=1.0):
    """Eq. 4, per sample, signed; positive = under-prediction. Visualization only."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return (y_true - y_pred) / (y_true + epsilon) * 100.0


def evaluate(truth: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """{target: {r_squared, smape, rmse, n}} for already-aligned frames."""
    return {t: {"r_squared": r_squared(truth[t], pred[t]), "smape": smape(truth[t], pred[t]),
                "rmse": rmse(truth[t], pred[t]), "n": int(len(truth))} for t in TARGETS}


def group_of(subset: str) -> str:
    return subset if subset.startswith("exemplar/") else SUBSET_GROUP.get(subset, subset)


def score(truth: pd.DataFrame, pred: pd.DataFrame) -> dict:
    missing = sorted(set(truth.sample_id) - set(pred.sample_id))
    extra = len(set(pred.sample_id) - set(truth.sample_id))
    merged = truth.merge(pred[["sample_id"] + TARGETS], on="sample_id", suffixes=("", "_pred"))
    if merged.empty:
        raise ValueError("no overlapping sample_id values between truth and predictions")
    t = merged[TARGETS]
    p = merged[[c + "_pred" for c in TARGETS]].set_axis(TARGETS, axis=1)
    bad = ~np.isfinite(p.to_numpy()).all(axis=1)
    if bad.any():
        raise ValueError(f"{bad.sum()} predictions are NaN/inf; a submission must give finite values")

    out = {
        "coverage": {"n_truth": int(len(truth)), "n_scored": int(len(merged)),
                     "n_truth_without_prediction": len(missing),
                     "n_predictions_without_truth": int(extra)},
        "groups": {"all": evaluate(t, p)},
    }
    if "subset" in merged.columns:
        groups = merged["subset"].map(group_of)
        for g in sorted(groups.unique()):
            m = (groups == g).to_numpy()
            if g != "all" and m.sum() > 0 and groups.nunique() > 1:
                out["groups"][g] = evaluate(t[m], p[m])
    out["_rpe"] = {k: relative_percent_error(t[k], p[k]) for k in TARGETS}
    return out


def render_markdown(result: dict, title: str) -> str:
    cov = result["coverage"]
    lines = [f"# {title}", "",
             f"Scored {cov['n_scored']} of {cov['n_truth']} ground-truth samples "
             f"({cov['n_truth_without_prediction']} without a prediction).", ""]
    for g, res in result["groups"].items():
        lines += [f"## {g}", "", "| Target | R^2 | SMAPE [%] | RMSE | n |", "|---|---|---|---|---|"]
        for t in TARGETS:
            r = res[t]
            r2 = "N/A*" if np.isnan(r["r_squared"]) else f"{r['r_squared']:.3f}"
            lines.append(f"| {t} | {r2} | {r['smape']:.2f} | {r['rmse']:.1f} | {r['n']} |")
        r2s = [res[t]["r_squared"] for t in TARGETS if not np.isnan(res[t]["r_squared"])]
        lines += ["", f"Mean R^2 over targets: {np.mean(r2s):.3f}" if r2s else "", ""]
    lines.append("*N/A: ground truth has zero variance for this target in this group (R^2 undefined).")
    return "\n".join(lines)


def plot_rpe_boxplot(errors_by_target, out_path, title="Relative Percent Error", symlog=True):
    """Paper Figures 7-12 style: one box per target, median and mean marked, symlog axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(errors_by_target)
    data = [np.asarray(errors_by_target[k]) for k in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    try:
        bp = ax.boxplot(data, tick_labels=labels, showmeans=True, meanline=True, patch_artist=True)
    except TypeError:  # matplotlib < 3.9
        bp = ax.boxplot(data, labels=labels, showmeans=True, meanline=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a6c8ff")
    for line, color in [(m, "orange") for m in bp["medians"]] + [(m, "green") for m in bp["means"]]:
        line.set_color(color)
        line.set_linestyle("--")
    if symlog:
        ax.set_yscale("symlog")
    ax.set_ylabel("Relative Percent Error [%]")
    ax.set_title(title)
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    truth = pd.read_csv(args.truth, dtype={"sample_id": str})
    pred = pd.read_csv(args.pred, dtype={"sample_id": str})
    for name, df in (("truth", truth), ("pred", pred)):
        missing_cols = [c for c in ["sample_id"] + TARGETS if c not in df.columns]
        if missing_cols:
            sys.exit(f"{name} CSV is missing columns: {missing_cols}")

    result = score(truth, pred)
    rpe = result.pop("_rpe")
    title = args.title or f"{args.pred.name} vs {args.truth.name}"
    md = render_markdown(result, title)
    (args.out / "METRICS.md").write_text(md, encoding="utf-8")
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_rpe_boxplot(rpe, args.out / "rpe_boxplot.png", title=f"Relative Percent Error: {title}")
    print(md)
    if result["coverage"]["n_truth_without_prediction"]:
        print(f"\nWARNING: {result['coverage']['n_truth_without_prediction']} ground-truth samples "
              "have no prediction (excluded from metrics).", file=sys.stderr)


if __name__ == "__main__":
    main()
