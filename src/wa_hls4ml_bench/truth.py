"""Write a ground-truth CSV for a split from its featurized cache.

    python -m wa_hls4ml_bench.truth --cache $WA_CACHE/test.npz --out results/test/truth.csv

``--gt post_synthesis`` (default) is the official benchmark ground truth
(``resource_report`` + ``latency_report``). ``--gt hls_estimate`` writes the
C-synthesis labels the GNN/Transformer checkpoints were trained on; it exists only for
the checkpoint-loading check in docs/VALIDATION.md and is not a benchmark result.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import data as D
from .cache import SplitCache


def truth_frame(cache: SplitCache, gt: str) -> pd.DataFrame:
    y = cache.truth_post if gt == D.POST_SYNTHESIS else cache.truth_hls
    keep = np.isfinite(y).all(axis=1)
    df = pd.DataFrame(y[keep], columns=D.TARGETS)
    df.insert(0, "subset", cache.subset[keep])
    df.insert(0, "sample_id", cache.sample_id[keep])
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--gt", choices=[D.POST_SYNTHESIS, D.HLS_ESTIMATE], default=D.POST_SYNTHESIS)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    cache = SplitCache(args.cache)
    df = truth_frame(cache, args.gt)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} of {len(cache)} samples with {args.gt} ground truth to {args.out}")


if __name__ == "__main__":
    main()
