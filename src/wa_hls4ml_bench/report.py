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


def _fmt(x, spec):
    return "N/A" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def build(results_dir: str) -> str:
    lines = ["# wa-hls4ml leaderboard", "",
             "Ground truth: post-logic-synthesis `resource_report` + `latency_report`.",
             "R^2 per target (higher is better) and SMAPE [%] (lower is better); "
             "the full per-group tables are in each `<split>/<model>/METRICS.md`.", ""]
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    md = build(args.results)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
