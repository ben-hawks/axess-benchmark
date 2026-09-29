#!/usr/bin/env python3
"""Download the wa-hls4ml results dataset from HuggingFace.

    python scripts/fetch_data.py --out $WA_DATA                 # test + exemplar (~0.7 GB)
    python scripts/fetch_data.py --out $WA_DATA --splits train  # only needed to regenerate
                                                                # normalization_stats.json

Run it on a login node (or a data-transfer node). Pin --revision to a commit hash for a
reproducible run; the revision actually used is written to <out>/REVISION.
"""

import argparse
import os

from huggingface_hub import snapshot_download

REPO_ID = "fastmachinelearning/wa-hls4ml"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", nargs="+", default=["test", "exemplar"],
                   choices=["train", "val", "test", "exemplar"])
    p.add_argument("--revision", default="main")
    args = p.parse_args()

    patterns = [f"{s}/*.json" for s in args.splits] + ["README.md"]
    path = snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=args.revision,
                             allow_patterns=patterns, local_dir=args.out)
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(REPO_ID, revision=args.revision).sha
    with open(os.path.join(args.out, "REVISION"), "w") as f:
        f.write(sha + "\n")
    print(f"{REPO_ID}@{sha} ({', '.join(args.splits)}) -> {path}")


if __name__ == "__main__":
    main()
