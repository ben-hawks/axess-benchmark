"""Loading the wa-hls4ml results dataset (HuggingFace JSON) and extracting ground truth.

The dataset is published at huggingface.co/datasets/fastmachinelearning/wa-hls4ml as
one JSON array per (split, subset) file, e.g. ``test/test_3layer_merged.json``. The
exemplar set is a single file, ``exemplar/exemplar_models.json``. See data/SCHEMA.md.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Iterator

import ijson

SPLITS = ("train", "val", "test", "exemplar")

# Benchmark target columns, in the order every CSV in this package uses.
TARGETS = ["BRAM", "DSP", "FF", "LUT", "cycles_max", "interval_max"]

# Paper Table 4 groups the synthetic test subsets into dense / conv1d / conv2d.
SUBSET_GROUP = {
    "2_20": "dense",
    "2layer": "dense",
    "3layer": "dense",
    "latency": "dense",
    "resource": "dense",
    "conv1d": "conv1d",
    "conv2d": "conv2d",
}

# Ground-truth definitions. POST_SYNTHESIS is the official benchmark target.
# HLS_ESTIMATE is what the GNN/Transformer reference checkpoints were trained on; it is
# used only to verify that those checkpoints are loaded correctly (see docs/VALIDATION.md).
POST_SYNTHESIS = "post_synthesis"
HLS_ESTIMATE = "hls_estimate"


def split_files(data_root: str, split: str) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    files = sorted(glob.glob(os.path.join(data_root, split, "*.json")))
    if not files:
        raise FileNotFoundError(
            f"no JSON files under {os.path.join(data_root, split)} -- "
            "run scripts/fetch_data.sh first"
        )
    return files


def subset_name(path: str, split: str) -> str:
    """``test/test_3layer_merged.json`` -> ``3layer``; the exemplar file -> ``exemplar``."""
    if split == "exemplar":
        return "exemplar"
    m = re.match(rf"{split}_(.+)_merged\.json$", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def exemplar_architecture(sample: dict) -> str:
    """``Bipc_2b_1rf_R`` -> ``Bipc`` (paper Table 5 groups the exemplar set by architecture)."""
    return sample["meta_data"]["model_name"].split("_")[0]


def _records(path: str) -> Iterator[dict]:
    """Stream records from a JSON-array file (the 3layer train file alone is 1.9 GB).

    ijson with use_float=True yields the same Python types as json.load.
    """
    with open(path, "rb") as f:
        first = f.read(64).lstrip()[:1]
    if first == b"[":
        with open(path, "rb") as f:
            yield from ijson.items(f, "item", use_float=True)
    else:  # object-rooted file: {key: sample, ...}
        with open(path, encoding="utf-8") as f:
            yield from json.load(f).values()


def iter_samples(data_root: str, split: str) -> Iterator[tuple[str, dict]]:
    """Yield ``(subset, sample)`` for every record in a split, file by file."""
    for path in split_files(data_root, split):
        subset = subset_name(path, split)
        for sample in _records(path):
            if sample is None:
                continue
            if split == "exemplar":
                yield f"exemplar/{exemplar_architecture(sample)}", sample
            else:
                yield subset, sample


def sample_id(sample: dict) -> str:
    """Unique per-sample id: ``meta_data.uuid``, or ``meta_data.model_id`` for the legacy
    ``2_20`` subset (which has no uuid). Both also name the sample's project tarball."""
    md = sample["meta_data"]
    return md.get("uuid") or md["model_id"]


def _num(val) -> float:
    return float(val)


def truth_post_synthesis(sample: dict) -> dict | None:
    """Official ground truth: post-logic-synthesis ``resource_report`` + ``latency_report``.

    Returns None when the sample has no post-synthesis report (~9.3% of the test set);
    such samples are excluded from scoring, never imputed.
    """
    rr = sample.get("resource_report")
    lr = sample.get("latency_report")
    if not rr or not lr:
        return None
    try:
        return {
            "BRAM": _num(rr["bram"]),
            "DSP": _num(rr["dsp"]),
            "FF": _num(rr["ff"]),
            "LUT": _num(rr["lut"]),
            "cycles_max": _num(lr["cycles_max"]),
            "interval_max": _num(lr["interval_max"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def has_hls_estimate(sample: dict) -> bool:
    """Sample filter used when the GNN/Transformer training arrays were built.

    Mirrors ``ModelProcessor.has_valid_resource_report`` in
    wa_hls4ml_models/dataset/Dataset_to_csvs6_with_ii.py: keep a sample iff its
    ``hls_resource_report`` has a non-empty ff/lut/bram/dsp entry. On the test split this
    keeps exactly 94,430 samples -- the test_size recorded in the GNN checkpoint.
    """
    hls = sample.get("hls_resource_report") or {}
    return any(hls.get(k) for k in ("ff", "lut", "bram", "dsp"))


def truth_hls_estimate(sample: dict) -> dict | None:
    """C-synthesis ``hls_resource_report`` labels, exactly as the GNN/Transformer were trained.

    Mirrors ``ModelProcessor.get_resource_report`` (int(float(x)), 0 on parse failure).
    """
    if not has_hls_estimate(sample):
        return None
    lat = sample.get("latency_report") or {}
    hls = sample.get("hls_resource_report") or {}

    def to_int(v):
        try:
            return int(float(v))
        except Exception:
            return 0

    return {
        "BRAM": float(to_int(hls.get("bram", 0))),
        "DSP": float(to_int(hls.get("dsp", 0))),
        "FF": float(to_int(hls.get("ff", 0))),
        "LUT": float(to_int(hls.get("lut", 0))),
        "cycles_max": float(to_int(lat.get("cycles_max", 0))),
        "interval_max": float(to_int(lat.get("interval_max", 0))),
    }


TRUTH_FNS = {POST_SYNTHESIS: truth_post_synthesis, HLS_ESTIMATE: truth_hls_estimate}
