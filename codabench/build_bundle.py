#!/usr/bin/env python3
"""Assemble the Codabench competition bundle from bundle_src/ plus generated data.

    PYTHONPATH=src python codabench/build_bundle.py --cache-dir $WA_CACHE \
        --results-dir $WA_RESULTS --out codabench/build

Inputs (from a normal benchmark run, docs/PERLMUTTER.md):
    <cache-dir>/{test,exemplar}.npz                       featurized splits (ground truth)
    <cache-dir>/train.npz                                 optional, for the weak baseline
    <results-dir>/{test,exemplar}/predictions_<model>.csv reference predictions

Writes:
    <out>/bundle/                    the bundle directory (validate this)
    <out>/competition_bundle.zip     the file to upload to Codabench
    <out>/extra/baseline_mean_submission.zip
        a deliberately weak submission (per-target training mean), for checking that the
        metrics separate it from the reference solution; not part of the bundle
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from wa_hls4ml_bench import data as D  # noqa: E402
from wa_hls4ml_bench.cache import SplitCache  # noqa: E402
from wa_hls4ml_bench.truth import truth_frame  # noqa: E402

SPLITS = ("test", "exemplar")
COLUMNS = ["sample_id"] + D.TARGETS


def zip_flat(zip_path, files):
    """Zip files at the archive root (Codabench expects no wrapping folder)."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, arcname=os.path.basename(path))


def zip_tree(zip_path, root):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, files in os.walk(root):
            for name in sorted(files):
                path = os.path.join(dirpath, name)
                z.write(path, arcname=os.path.relpath(path, root))


def scored_predictions(pred_csv, ids):
    df = pd.read_csv(pred_csv, dtype={"sample_id": str}).set_index("sample_id")
    missing = ids[~ids.isin(df.index)]
    if len(missing):
        sys.exit(f"{pred_csv}: no prediction for {len(missing)} scored samples")
    out = df.loc[ids, D.TARGETS]
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        sys.exit(f"{pred_csv}: non-finite predictions for scored samples")
    return out.reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--reference-model", default="transformer",
                   help="whose predictions become solution/ and sample_submission.zip")
    p.add_argument("--out", default=os.path.join(HERE, "build"))
    args = p.parse_args()

    bundle = os.path.join(args.out, "bundle")
    extra = os.path.join(args.out, "extra")
    shutil.rmtree(args.out, ignore_errors=True)
    shutil.copytree(os.path.join(HERE, "bundle_src"), bundle,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for d in ("reference_data", "solution", extra):
        os.makedirs(os.path.join(bundle, d) if d != extra else extra, exist_ok=True)

    train_mean = None
    train_path = os.path.join(args.cache_dir, "train.npz")
    if os.path.exists(train_path):
        y = SplitCache(train_path).truth_post
        train_mean = np.nanmean(y[np.isfinite(y).all(axis=1)], axis=0)

    solution_files, baseline_files = [], []
    for split in SPLITS:
        truth = truth_frame(SplitCache(os.path.join(args.cache_dir, f"{split}.npz")), D.POST_SYNTHESIS)
        truth.to_csv(os.path.join(bundle, "reference_data", f"truth_{split}.csv"), index=False)
        truth[["sample_id"]].to_csv(os.path.join(bundle, "starting_kit", f"{split}_sample_ids.csv"), index=False)

        pred = scored_predictions(
            os.path.join(args.results_dir, split, f"predictions_{args.reference_model}.csv"), truth["sample_id"])
        path = os.path.join(bundle, "solution", f"predictions_{split}.csv")
        pred[COLUMNS].to_csv(path, index=False)
        solution_files.append(path)

        if train_mean is not None:
            base = pd.DataFrame(np.tile(train_mean, (len(truth), 1)), columns=D.TARGETS)
            base.insert(0, "sample_id", truth["sample_id"].to_numpy())
            path = os.path.join(extra, f"predictions_{split}.csv")
            base.to_csv(path, index=False)
            baseline_files.append(path)
        print(f"{split}: {len(truth)} scored samples")

    zip_flat(os.path.join(bundle, "starting_kit", "sample_submission.zip"), solution_files)
    if baseline_files:
        zip_flat(os.path.join(extra, "baseline_mean_submission.zip"), baseline_files)
        for path in baseline_files:
            os.remove(path)

    if not os.path.exists(os.path.join(bundle, "logo.png")):
        print("WARNING: no logo.png in bundle_src/. competition.yaml requires `image`, so "
              "add codabench/bundle_src/logo.png and rebuild before uploading.")
    zip_tree(os.path.join(args.out, "competition_bundle.zip"), bundle)
    size = os.path.getsize(os.path.join(args.out, "competition_bundle.zip")) / 1e6
    print(f"wrote {args.out}/competition_bundle.zip ({size:.1f} MB)")


if __name__ == "__main__":
    main()
