"""Featurize a dataset split once and cache it as a compact .npz.

Parsing the JSON and running the vendored feature extractor is the slow part of
inference (~1 ms/sample), so every model reads this cache rather than re-parsing JSON.

    python -m wa_hls4ml_bench.cache --data-root $WA_DATA --split test --out $WA_CACHE/test.npz

Cache contents (N samples, T = total layers across samples):
    sample_id  (N,)   str    meta_data.uuid
    subset     (N,)   str    e.g. "3layer", "conv2d", "exemplar/Jet"
    layers     (T,18) f64    raw per-layer features (features.raw_layer_features)
    offsets    (N+1,) i64    sample i owns layers[offsets[i]:offsets[i+1]]
    truth_post (N,6)  f64    official ground truth (data.TARGETS order), NaN if missing
    truth_hls  (N,6)  f64    HLS-estimate labels (data.TARGETS order), NaN if missing
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
    return (
        sid, subset, layers, ok,
        [tp[t] for t in D.TARGETS] if tp else nan6,
        [th[t] for t in D.TARGETS] if th else nan6,
    )


def build_cache(data_root: str, split: str, out_path: str, workers: int = 1) -> dict:
    t0 = time.time()
    ids, subsets, layer_blocks, oks, post, hls = [], [], [], [], [], []
    items = D.iter_samples(data_root, split)
    if workers > 1:
        with Pool(workers, initializer=_quiet) as pool:
            results = pool.imap(_process, items, chunksize=256)  # imap preserves order
            for r in results:
                _collect(r, ids, subsets, layer_blocks, oks, post, hls)
    else:
        for r in map(_process, items):
            _collect(r, ids, subsets, layer_blocks, oks, post, hls)

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
        feat_ok=np.array(oks, dtype=bool),
    )
    summary = {
        "split": split,
        "n_samples": len(ids),
        "n_feature_failures": int((~np.array(oks)).sum()),
        "n_post_synthesis_truth": int(np.isfinite(np.array(post)[:, 0]).sum()),
        "n_hls_estimate_truth": int(np.isfinite(np.array(hls)[:, 0]).sum()),
        "max_layers": int(lengths.max()) if len(lengths) else 0,
        "seconds": round(time.time() - t0, 1),
    }
    print(summary, flush=True)
    return summary


def _collect(r, ids, subsets, layer_blocks, oks, post, hls):
    sid, subset, layers, ok, tp, th = r
    ids.append(sid)
    subsets.append(subset)
    layer_blocks.append(layers)
    oks.append(ok)
    post.append(tp)
    hls.append(th)


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
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    _quiet()
    build_cache(args.data_root, args.split, args.out, args.workers)


if __name__ == "__main__":
    main()
