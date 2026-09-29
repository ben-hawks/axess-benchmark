"""Recompute the normalization statistics the GNN/Transformer checkpoints were trained with.

Neither published checkpoint ships its normalization stats, so this reproduces
``FPGAGraphDataset._calculate_normalization_stats`` (wa_hls4ml_models/transformer/GNN/
Dataset2.py) over the training split, filtered the way the original training arrays
were (``data.has_hls_estimate``). This is a one-time provenance step, not part of a
benchmark run: its output ships next to the weights as ``normalization_stats.json`` and
is verified by reproducing the GNN checkpoint's own stored test metrics
(docs/VALIDATION.md).

    python -m wa_hls4ml_bench.stats --train-cache $WA_CACHE/train.npz --out weights/normalization_stats.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import data as D
from .cache import SplitCache
from .features import MODEL_OUTPUT_ORDER, NUMERICAL_FEATURE_KEYS, NUMERICAL_IDX

# log-transform epsilons used by each training script
LOG_EPSILON = {
    "gnn": 1e-8,          # GNN/training_scripts/y_03_LGAT_simple_bigModel.py
    "transformer": 1e-6,  # transformer/run.py
}


def compute(train: SplitCache) -> dict:
    keep = np.isfinite(train.truth_hls[:, 0]) & train.feat_ok
    idx = np.nonzero(keep)[0]
    layer_rows = np.concatenate([np.arange(train.offsets[i], train.offsets[i + 1]) for i in idx])
    layers = train.layers[layer_rows]

    feature_means, feature_stds = [], []
    for col in NUMERICAL_IDX:
        vals = layers[:, col]
        vals = vals[vals != -1]  # original skips -1 entries (NaN is kept, as in the original)
        if len(vals):
            feature_means.append(float(np.mean(vals)))
            s = float(np.std(vals))
            feature_stds.append(s if s > 1e-5 else 1.0)
        else:
            feature_means.append(0.0)
            feature_stds.append(1.0)

    col = [D.TARGETS.index(t) for t in MODEL_OUTPUT_ORDER]
    labels = train.truth_hls[idx][:, col]  # model output order
    out = {
        "n_train_samples": int(len(idx)),
        "feature_keys": NUMERICAL_FEATURE_KEYS,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "label_order": MODEL_OUTPUT_ORDER,
        "models": {},
    }
    for model, eps in LOG_EPSILON.items():
        min_val = float(np.min(labels))
        log_shift = eps + abs(min_val) if min_val <= 0 else eps
        log_labels = np.log(labels + log_shift)
        stds = np.std(log_labels, axis=0)
        out["models"][model] = {
            "log_epsilon": eps,
            "log_shift": log_shift,
            "label_means": np.mean(log_labels, axis=0).tolist(),
            "label_stds": np.where(stds > 1e-5, stds, 1.0).tolist(),
        }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    stats = compute(SplitCache(args.train_cache))
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
