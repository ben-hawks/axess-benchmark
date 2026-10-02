"""Featurize a dataset split once and cache it as a compact .npz.

Parsing the JSON and running the vendored feature extractor is the slow part of
inference (~1 ms/sample), so every model reads this cache rather than re-parsing JSON.

    python -m wa_hls4ml_bench.cache --data-root $WA_DATA --split test --cache-dir $WA_CACHE

Cache contents (N samples, T = total layers across samples):
    sample_id  (N,)   str    meta_data.uuid
    subset     (N,)   str    e.g. "3layer", "conv2d", "exemplar/Jet"
    layers     (T,18) f64    raw per-layer features (features.raw_layer_features)
    offsets    (N+1,) i64    sample i owns layers[offsets[i]:offsets[i+1]]
    truth_post (N,6)  f64    official ground truth (data.TARGETS order), NaN if missing
    truth_hls  (N,6)  f64    HLS-estimate labels (data.TARGETS order), NaN if missing
    truth_train (N,6) f64    labels as the reference models were trained on them
                             (data.truth_training_labels), NaN if the sample was skipped
    feat_ok    (N,)   bool   False if feature extraction failed (sample gets no prediction)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from multiprocessing import Pool

import numpy as np

from . import data as D
from .features import raw_layer_features


def _quiet():
    # The vendored extractor logs a warning per sample whose hls_config and model_config
    # layer counts disagree (the original printed it); keep worker output readable.
    logging.getLogger("wa_hls4ml_bench.features").setLevel(logging.ERROR)


def _process(item):
    subset, sample = item
    sid = D.sample_id(sample)
    try:
        layers = raw_layer_features(sample)
        ok = len(layers) > 0
    except Exception:
        layers, ok = np.zeros((0, 18)), False
    nan6 = [np.nan] * 6
    tp = D.truth_post_synthesis(sample)
    th = D.truth_hls_estimate(sample)
    tt = D.truth_training_labels(sample)
    return (
        sid, subset, layers, ok,
        [tp[t] for t in D.TARGETS] if tp else nan6,
        [th[t] for t in D.TARGETS] if th else nan6,
        [tt[t] for t in D.TARGETS] if tt else nan6,
    )


def cache_path(cache_dir: str, split: str) -> str:
    """Where a split's cache lives inside a cache directory."""
    return os.path.join(cache_dir, f"{split}.npz")


def build_cache(data_root: str, split: str, out_path: str, workers: int = 1) -> dict:
    t0 = time.time()
    ids, subsets, layer_blocks, oks, post, hls, train = [], [], [], [], [], [], []
    items = D.iter_samples(data_root, split)
    if workers > 1:
        with Pool(workers, initializer=_quiet) as pool:
            results = pool.imap(_process, items, chunksize=256)  # imap preserves order
            for r in results:
                _collect(r, ids, subsets, layer_blocks, oks, post, hls, train)
    else:
        for r in map(_process, items):
            _collect(r, ids, subsets, layer_blocks, oks, post, hls, train)

    if len(set(ids)) != len(ids):
        raise ValueError(f"{split}: meta_data.uuid is not unique within the split")

    lengths = np.array([len(b) for b in layer_blocks], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(
        out_path,
        sample_id=np.array(ids),
        subset=np.array(subsets),
        layers=np.concatenate(layer_blocks) if layer_blocks else np.zeros((0, 18)),
        offsets=offsets,
        truth_post=np.array(post, dtype=float),
        truth_hls=np.array(hls, dtype=float),
        truth_train=np.array(train, dtype=float),
        feat_ok=np.array(oks, dtype=bool),
    )
    summary = {
        "split": split,
        "n_samples": len(ids),
        "n_feature_failures": int((~np.array(oks)).sum()),
        "n_post_synthesis_truth": int(np.isfinite(np.array(post)[:, 0]).sum()),
        "n_hls_estimate_truth": int(np.isfinite(np.array(hls)[:, 0]).sum()),
        "n_training_labels": int(np.isfinite(np.array(train)[:, 0]).sum()),
        "max_layers": int(lengths.max()) if len(lengths) else 0,
        "seconds": round(time.time() - t0, 1),
    }
    print(summary, flush=True)
    return summary


def _collect(r, ids, subsets, layer_blocks, oks, post, hls, train):
    sid, subset, layers, ok, tp, th, tt = r
    ids.append(sid)
    subsets.append(subset)
    layer_blocks.append(layers)
    oks.append(ok)
    post.append(tp)
    hls.append(th)
    train.append(tt)


class SplitCache:
    """Read-side view of a cache .npz."""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.sample_id = z["sample_id"]
        self.subset = z["subset"]
        self.layers = z["layers"]
        self.offsets = z["offsets"]
        self.truth_post = z["truth_post"]
        self.truth_hls = z["truth_hls"]
        # absent in caches built before 2026-10-02; only stats.py needs it
        self.truth_train = z["truth_train"] if "truth_train" in z.files else None
        self.feat_ok = z["feat_ok"]

    def __len__(self):
        return len(self.sample_id)

    def raw(self, i: int) -> np.ndarray:
        return self.layers[self.offsets[i]:self.offsets[i + 1]]

    def n_layers(self) -> np.ndarray:
        return np.diff(self.offsets)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="local copy of fastmachinelearning/wa-hls4ml")
    p.add_argument("--split", required=True, choices=D.SPLITS)
    p.add_argument("--cache-dir", help="write <cache-dir>/<split>.npz")
    p.add_argument("--out", help="explicit cache file (alternative to --cache-dir)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = p.parse_args(argv)
    if not (args.cache_dir or args.out):
        p.error("give --cache-dir (or --out FILE)")
    logging.basicConfig(level=logging.ERROR)
    _quiet()
    out = args.out or cache_path(args.cache_dir, args.split)
    build_cache(args.data_root, args.split, out, args.workers)


if __name__ == "__main__":
    main()
