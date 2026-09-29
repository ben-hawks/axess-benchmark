#!/usr/bin/env python3
"""Fetch the pretrained reference-model weights and verify their checksums.

    python scripts/fetch_weights.py --out $WA_WEIGHTS
    python scripts/fetch_weights.py --out $WA_WEIGHTS --from-local /path/to/checkpoints

Files and sha256 sums come from weights/MANIFEST.json. normalization_stats.json is small
and versioned in this repo, so it is copied from weights/ rather than downloaded.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, "..", "weights")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--from-local", help="copy checkpoints from this directory instead of HuggingFace")
    args = p.parse_args()

    with open(os.path.join(WEIGHTS_DIR, "MANIFEST.json")) as f:
        manifest = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    shutil.copy(os.path.join(WEIGHTS_DIR, "normalization_stats.json"), args.out)

    ok = True
    for name, info in manifest["files"].items():
        dest = os.path.join(args.out, name)
        if not (os.path.exists(dest) and sha256(dest) == info["sha256"]):
            if args.from_local:
                shutil.copy(os.path.join(args.from_local, name), dest)
            else:
                from huggingface_hub import hf_hub_download

                hf_hub_download(repo_id=manifest["hf_repo"], filename=name,
                                revision=manifest.get("hf_revision", "main"), local_dir=args.out)
        got = sha256(dest)
        status = "ok" if got == info["sha256"] else f"CHECKSUM MISMATCH (got {got})"
        ok &= got == info["sha256"]
        print(f"{name}: {status}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
