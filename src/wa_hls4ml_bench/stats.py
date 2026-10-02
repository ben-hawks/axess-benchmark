"""Rebuild the normalization statistics and prediction cap the reference checkpoints use.

The retrained GNN and Transformer (wa_hls4ml_models@resource-report-retrain) were trained
on the same arrays with the same log transform, so they share one set of statistics:
feature z-score stats, log-space label mean/std, the log shift, and the per-target
prediction cap (the largest training label, applied by the upstream ``transformer/run.py``
and ``GNN/load_pretrained.py``). This reproduces ``FPGAGraphDataset._calculate_normalization_stats``
over the train split, using the labels exactly as the upstream converter built them
(``data.truth_training_labels``).

It is a one-time provenance step, not part of a benchmark run: the output ships as
``weights/normalization_stats.json``. The GNN release also ships the stats it was trained
with (``resource_report_results/gnn/normalization_stats_log.npy``); pass it as
``--check`` to compare.

    python -m wa_hls4ml_bench.stats --train-cache $WA_CACHE/train.npz \
        --out weights/normalization_stats.json --check normalization_stats_log.npy
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import data as D
from .cache import SplitCache
from .features import MODEL_OUTPUT_ORDER, NUMERICAL_FEATURE_KEYS, NUMERICAL_IDX

LOG_EPSILON = 1e-6  # transformer/run.py and GNN/training_scripts/y_03_GAT_vanilla_bigboi.py
MODELS = ("gnn", "transformer")


def compute(train: SplitCache) -> dict:
    if train.truth_train is None:
        raise SystemExit("this train cache predates the truth_train column; rebuild it with "
                         "python -m wa_hls4ml_bench.cache")
    keep = np.isfinite(train.truth_train[:, 0]) & train.feat_ok
    idx = np.nonzero(keep)[0]
    layer_rows = np.concatenate([np.arange(train.offsets[i], train.offsets[i + 1]) for i in idx])
    layers = train.layers[layer_rows]

    feature_means, feature_stds = [], []
    for col in NUMERICAL_IDX:
        vals = layers[:, col]
        vals = vals[vals != -1]  # the original skips -1 entries
        if len(vals):
            feature_means.append(float(np.mean(vals)))
            s = float(np.std(vals))
            feature_stds.append(s if s > 1e-5 else 1.0)
        else:
            feature_means.append(0.0)
            feature_stds.append(1.0)

    col = [D.TARGETS.index(t) for t in MODEL_OUTPUT_ORDER]
    labels = train.truth_train[idx][:, col]  # model output order
    min_val = float(np.min(labels))
    log_shift = LOG_EPSILON + abs(min_val) if min_val <= 0 else LOG_EPSILON
    log_labels = np.log(labels + log_shift)
    stds = np.std(log_labels, axis=0)
    label_stats = {
        "log_epsilon": LOG_EPSILON,
        "log_shift": log_shift,
        "label_means": np.mean(log_labels, axis=0).tolist(),
        "label_stds": np.where(stds > 1e-5, stds, 1.0).tolist(),
        "label_max": labels.max(axis=0).tolist(),  # prediction cap, original units
    }
    return {
        "label_source": "resource_report (post-synthesis) + latency_report (HLS estimate)",
        "n_train_samples": int(len(idx)),
        "feature_keys": NUMERICAL_FEATURE_KEYS,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "label_order": MODEL_OUTPUT_ORDER,
        "models": {m: dict(label_stats) for m in MODELS},
    }


def check(stats: dict, npy_path: str) -> float:
    """Largest relative difference against an upstream normalization_stats_log.npy."""
    ref = np.load(npy_path, allow_pickle=True).item()
    m = stats["models"]["gnn"]
    pairs = [("feature_means", stats["feature_means"]), ("feature_stds", stats["feature_stds"]),
             ("label_means", m["label_means"]), ("label_stds", m["label_stds"])]
    worst = 0.0
    for key, ours in pairs:
        r = np.asarray(ref[key], dtype=float)
        d = np.max(np.abs(np.asarray(ours) - r) / (np.abs(r) + 1e-12))
        print(f"{key:14s} max rel diff {d:.2e}")
        worst = max(worst, d)
    print(f"log_shift      ours {m['log_shift']} upstream {ref['log_shift']}")
    return worst


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--check", help="upstream normalization_stats_log.npy to compare against")
    args = p.parse_args(argv)
    stats = compute(SplitCache(args.train_cache))
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "models"}, indent=2))
    print(json.dumps(stats["models"]["gnn"], indent=2))
    if args.check:
        check(stats, args.check)


if __name__ == "__main__":
    main()
