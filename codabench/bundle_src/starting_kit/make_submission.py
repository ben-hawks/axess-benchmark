#!/usr/bin/env python3
"""Check two prediction CSVs and zip them into an uploadable submission.

    python make_submission.py --test predictions_test.csv \
        --exemplar predictions_exemplar.csv --out submission.zip

Checks the same things the scoring program enforces: the required columns, no
duplicate ids, a finite prediction for every scored sample (test_sample_ids.csv /
exemplar_sample_ids.csv next to this script). Extra columns and rows are dropped, so
predictions written by the benchmark package's `wa_hls4ml_bench.predict` (which include
a `subset` column and unscored samples) can be passed in directly.
"""

import argparse
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ["sample_id", "BRAM", "DSP", "FF", "LUT", "cycles_max", "interval_max"]


def check(path, split):
    ids = pd.read_csv(os.path.join(HERE, f"{split}_sample_ids.csv"), dtype={"sample_id": str})["sample_id"]
    df = pd.read_csv(path, dtype={"sample_id": str})
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        sys.exit(f"{path}: missing columns {missing_cols}")
    if df["sample_id"].duplicated().any():
        sys.exit(f"{path}: duplicate sample_id values")
    df = df.set_index("sample_id")
    missing = ids[~ids.isin(df.index)]
    if len(missing):
        sys.exit(f"{path}: no prediction for {len(missing)} of {len(ids)} scored {split} samples, "
                 f"e.g. {list(missing[:3])}")
    out = df.loc[ids, COLUMNS[1:]]
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        sys.exit(f"{path}: NaN/inf predictions for scored {split} samples")
    print(f"{split}: {len(out)} scored samples ok ({len(df) - len(out)} extra rows dropped)")
    return out.reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True)
    p.add_argument("--exemplar", required=True)
    p.add_argument("--out", default="submission.zip")
    args = p.parse_args()
    frames = {"test": check(args.test, "test"), "exemplar": check(args.exemplar, "exemplar")}
    with tempfile.TemporaryDirectory() as tmp, \
            zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for split, df in frames.items():
            name = f"predictions_{split}.csv"
            path = os.path.join(tmp, name)
            df.to_csv(path, index=False)
            z.write(path, arcname=name)  # files at the zip root, no wrapping folder
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
