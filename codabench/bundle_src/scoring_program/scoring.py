"""Scoring program for the AXESS wa-hls4ml Benchmark (results submission).

Reads the participant's two prediction files from the submission:

    predictions_test.csv       one row per test sample
    predictions_exemplar.csv   one row per exemplar sample

each with columns sample_id, BRAM, DSP, FF, LUT, cycles_max, interval_max. They are
scored against the hidden ground truth in reference_data/ (post-synthesis resources,
HLS-estimate latency)
(truth_test.csv, truth_exemplar.csv) and the results are written to scores.json, whose
keys match competition.yaml's leaderboard columns.

A submission must give a finite prediction for every ground-truth sample of both
splits; anything less fails with a message naming the problem. Extra rows (e.g.
test samples that have no post-synthesis report) are ignored.

CODABENCH_ROOT defaults to /app, the layout of a real Codabench container; the
benchmark-builder validator overrides it to dry-run this program locally.
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd

from metrics import r_squared, rmse, smape

ROOT = os.environ.get("CODABENCH_ROOT", "/app")
reference_dir = os.path.join(ROOT, "input", "ref")
prediction_dir = os.path.join(ROOT, "input", "res")
score_dir = os.path.join(ROOT, "output")

SPLITS = ("test", "exemplar")
TARGETS = ["BRAM", "DSP", "FF", "LUT", "cycles_max", "interval_max"]
KEY = {"BRAM": "bram", "DSP": "dsp", "FF": "ff", "LUT": "lut", "cycles_max": "cycles", "interval_max": "ii"}
GROUP = {"2_20": "dense", "2layer": "dense", "3layer": "dense", "latency": "dense",
         "resource": "dense", "conv1d": "conv1d", "conv2d": "conv2d"}


def fail(msg):
    print(f"SUBMISSION ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# The submission must provide exactly these files at the root of its zip.
PREDICTION_FILES = {
    "test": os.path.join(prediction_dir, "predictions_test.csv"),
    "exemplar": os.path.join(prediction_dir, "predictions_exemplar.csv"),
}


def find_file(split):
    """Codabench unzips the submission into input/res. Tolerate a wrapping folder."""
    expected = PREDICTION_FILES[split]
    if os.path.exists(expected):
        return expected
    name = os.path.basename(expected)
    for dirpath, _, files in os.walk(prediction_dir):
        if name in files:
            return os.path.join(dirpath, name)
    fail(f"'{name}' not found in the submission. Zip predictions_test.csv and "
         "predictions_exemplar.csv at the root of the archive.")


def load_predictions(split, truth):
    pred = pd.read_csv(find_file(split), dtype={"sample_id": str})
    missing_cols = [c for c in ["sample_id"] + TARGETS if c not in pred.columns]
    if missing_cols:
        fail(f"predictions_{split}.csv is missing columns {missing_cols}")
    if pred["sample_id"].duplicated().any():
        fail(f"predictions_{split}.csv has duplicate sample_id values")
    pred = pred.set_index("sample_id")
    missing = truth.index.difference(pred.index)
    if len(missing):
        fail(f"predictions_{split}.csv has no prediction for {len(missing)} of {len(truth)} "
             f"scored samples (e.g. {list(missing[:3])}); every scored sample is required")
    pred = pred.loc[truth.index, TARGETS]
    try:
        values = pred.to_numpy(dtype=float)
    except ValueError:
        fail(f"predictions_{split}.csv contains non-numeric predictions")
    if not np.isfinite(values).all():
        fail(f"predictions_{split}.csv contains NaN/inf predictions for scored samples")
    return pd.DataFrame(values, index=truth.index, columns=TARGETS)


def evaluate(truth, pred):
    return {t: {"r2": r_squared(truth[t], pred[t]), "smape": smape(truth[t], pred[t]),
                "rmse": rmse(truth[t], pred[t])} for t in TARGETS}


def finite_mean(values):
    vals = [v for v in values if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def group_of(split, subset):
    return subset.split("/", 1)[1] if split == "exemplar" else GROUP.get(subset, subset)


def report(split, truth, pred):
    """Print the full per-group tables (the leaderboard only shows split-level numbers)."""
    groups = {"all": slice(None)}
    for g in sorted({group_of(split, s) for s in truth["subset"]}):
        groups[g] = (truth["subset"].map(lambda s: group_of(split, s)) == g).to_numpy()
    for g, mask in groups.items():
        res = evaluate(truth.loc[mask, TARGETS] if g != "all" else truth[TARGETS],
                       pred.loc[mask] if g != "all" else pred)
        n = len(truth) if g == "all" else int(mask.sum())
        print(f"\n[{split} / {g}] n={n}")
        print(f"  {'target':14s} {'R^2':>9s} {'SMAPE%':>8s} {'RMSE':>14s}")
        for t in TARGETS:
            r = res[t]
            print(f"  {t:14s} {r['r2']:9.3f} {r['smape']:8.2f} {r['rmse']:14.1f}")


def main():
    scores = {}
    for split in SPLITS:
        truth = pd.read_csv(os.path.join(reference_dir, f"truth_{split}.csv"),
                            dtype={"sample_id": str}).set_index("sample_id")
        pred = load_predictions(split, truth)
        res = evaluate(truth[TARGETS], pred)
        scores[f"{split}_mean_r2"] = finite_mean([res[t]["r2"] for t in TARGETS])
        scores[f"{split}_mean_smape"] = finite_mean([res[t]["smape"] for t in TARGETS])
        for t in TARGETS:
            scores[f"{split}_r2_{KEY[t]}"] = res[t]["r2"]
            scores[f"{split}_smape_{KEY[t]}"] = res[t]["smape"]
        report(split, truth, pred)

    # JSON has no NaN; a zero-variance R^2 (not expected at split level) becomes null.
    scores = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in scores.items()}
    print("\nScores:", json.dumps(scores, indent=1))
    os.makedirs(score_dir, exist_ok=True)
    with open(os.path.join(score_dir, "scores.json"), "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    main()
