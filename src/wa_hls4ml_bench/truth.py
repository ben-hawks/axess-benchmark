"""The benchmark's single truth function, and a CLI that writes a split's truth CSV.

    python -m wa_hls4ml_bench.truth --cache-dir $WA_CACHE --split test --out results/test/truth.csv

``truth_frame(cache_dir, split)`` and ``scored_ids(cache_dir, split)`` are what scoring
(``score_all.sh``), Codabench packaging (``submission.py``) and the Codabench bundle
(``codabench/build_bundle.py``) all use, so the set of scored samples and their ground
truth can't diverge between them.

``--gt post_synthesis`` (default) is the official benchmark ground truth
(post-synthesis ``resource_report`` + HLS-estimate ``latency_report``; the name refers
to the resource labels, the dataset has no post-synthesis latency). ``--gt hls_estimate``
writes the C-synthesis labels the paper's original (pre-retrain) GNN/Transformer
checkpoints were trained on; it exists only for checking such models
(docs/VALIDATION.md section 5) and is not a benchmark result.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import data as D
from .cache import SplitCache, cache_path


def frame_from_cache(cache: SplitCache, gt: str = D.POST_SYNTHESIS) -> pd.DataFrame:
    """sample_id, subset and the output columns for every sample with this ground truth."""
    y = cache.truth_post if gt == D.POST_SYNTHESIS else cache.truth_hls
    keep = np.isfinite(y).all(axis=1)
    df = pd.DataFrame(y[keep], columns=D.OUTPUT_COLUMNS)
    df.insert(0, "subset", cache.subset[keep])
    df.insert(0, "sample_id", cache.sample_id[keep])
    return df


def truth_frame(cache_dir: str, split: str, gt: str = D.POST_SYNTHESIS) -> pd.DataFrame:
    """Ground truth for the scored samples of a split (the benchmark's single truth function)."""
    return frame_from_cache(SplitCache(cache_path(cache_dir, split)), gt)


def scored_ids(cache_dir: str, split: str) -> pd.Series:
    """The sample_ids that are scored in a split, in cache order."""
    return truth_frame(cache_dir, split)["sample_id"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", help="directory with <split>.npz")
    p.add_argument("--split", choices=D.SPLITS)
    p.add_argument("--cache", help="a single cache file (alternative to --cache-dir + --split)")
    p.add_argument("--gt", choices=[D.POST_SYNTHESIS, D.HLS_ESTIMATE], default=D.POST_SYNTHESIS)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.cache:
        cache = SplitCache(args.cache)
    elif args.cache_dir and args.split:
        cache = SplitCache(cache_path(args.cache_dir, args.split))
    else:
        p.error("give --cache-dir and --split (or --cache FILE)")
    df = frame_from_cache(cache, args.gt)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} of {len(cache)} samples with {args.gt} ground truth to {args.out}")


if __name__ == "__main__":
    main()
