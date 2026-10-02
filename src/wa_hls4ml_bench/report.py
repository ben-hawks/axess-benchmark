"""Collect every <results>/<split>/<model>/metrics.json into one LEADERBOARD.md.

    python -m wa_hls4ml_bench.report --results $WA_RESULTS --out $WA_RESULTS/LEADERBOARD.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

from .score import TARGETS

# Scored for comparison only; not reference solutions (reference_solution/README.md).
AUXILIARY = {"rule4ml_gnn"}


def _fmt(x, spec):
    return "N/A" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def build(results_dir: str) -> str:
    lines = ["# wa-hls4ml leaderboard", "",
             "Ground truth: post-logic-synthesis `resource_report` (BRAM/DSP/FF/LUT) + "
             "HLS-estimate `latency_report` (cycles, II).",
             "R^2 per target (higher is better) and SMAPE [%] (lower is better); "
             "the full per-group tables are in each `<split>/<model>/METRICS.md`.",
             "Rows in *italics* are auxiliary comparison models, not reference solutions.", ""]
    for split_dir in sorted(glob.glob(os.path.join(results_dir, "*", ""))):
        split = os.path.basename(os.path.normpath(split_dir))
        runs = sorted(glob.glob(os.path.join(split_dir, "*", "metrics.json")))
        if not runs:
            continue
        lines += [f"## {split}", "",
                  "| Model | coverage | " + " | ".join(f"{t} R^2" for t in TARGETS)
                  + " | mean R^2 | " + " | ".join(f"{t} SMAPE" for t in TARGETS) + " |",
                  "|---" * (3 + 2 * len(TARGETS)) + "|"]
        for path in runs:
            with open(path) as f:
                m = json.load(f)
            model = os.path.basename(os.path.dirname(path))
            if model in AUXILIARY:
                model = f"*{model} (auxiliary)*"
            res = m["groups"]["all"]
            cov = m["coverage"]
            r2 = [res[t]["r_squared"] for t in TARGETS]
            finite = [v for v in r2 if not math.isnan(v)]
            mean = sum(finite) / len(finite) if finite else float("nan")
            lines.append(
                f"| {model} | {cov['n_scored']}/{cov['n_truth']} | "
                + " | ".join(_fmt(v, ".3f") for v in r2) + f" | {_fmt(mean, '.3f')} | "
                + " | ".join(_fmt(res[t]["smape"], ".1f") for t in TARGETS) + " |")
        lines.append("")
    return "\n".join(lines)


def build_timings(results_dir: str) -> str | None:
    """All <split>/predictions_<model>.timing.json (written by predict.py) as one table."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "predictions_*.timing.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        return None
    lines = ["# Inference timings", "",
             "Written by `predict.py` for each model and split. *Total* covers the whole command "
             "(loading weights and data, featurization, writing the CSV); *model* is time inside "
             "the model only (synchronized on GPU).", "",
             "| Split | Model | Device | Hardware | Samples | Total [s] | Total [ms/sample] "
             "| Model [s] | Model [ms/sample] | Batch | Slurm job | Host | Finished |",
             "|---" * 13 + "|"]
    for r in rows:
        model = f"*{r['model']} (auxiliary)*" if r["model"] in AUXILIARY else r["model"]
        lines.append(
            f"| {r['split']} | {model} | {r['device']} | {r['hardware']} | {r['n_samples']} "
            f"| {r['seconds_total']:.1f} | {r['ms_per_sample_total']:.3f} | {r['seconds_model']:.1f} "
            f"| {r['ms_per_sample_model']:.4f} | {r.get('batch_size') or ''} | {r.get('slurm_job_id') or ''} "
            f"| {r.get('host', '')} | {r.get('finished', '')} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--timings-out", help="also write the inference-timing table here (if any timings exist)")
    args = p.parse_args(argv)
    md = build(args.results)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    if args.timings_out:
        timings = build_timings(args.results)
        if timings:
            with open(args.timings_out, "w", encoding="utf-8") as f:
                f.write(timings)
            print(f"(timings written to {args.timings_out})")


if __name__ == "__main__":
    main()
