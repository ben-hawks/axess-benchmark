"""Run a pretrained reference model over a dataset split and write a predictions CSV.

    # GNN / Transformer read the featurized cache (see cache.py)
    python -m wa_hls4ml_bench.predict --model gnn --cache-dir $WA_CACHE --split test \
        --weights-dir $WA_WEIGHTS --out results/test/predictions_gnn.csv

    # the rule4ml models (mlp, rule4ml_gnn) read the raw JSON directly
    python -m wa_hls4ml_bench.predict --model mlp --data-root $WA_DATA --split test \
        --out results/test/predictions_mlp.csv

Output columns: sample_id, subset, BRAM, DSP, FF, LUT, cycles_max, interval_max, with a
row for **every** sample of the split, whether or not it has ground truth. A sample the
model can't handle (features that couldn't be built, a network longer than the
Transformer's 51 positions, a configuration rule4ml can't represent) gets a row with
empty outputs, and the count is printed. score.py reports such rows as missing
predictions; submission.py refuses to package a model that has them for scored samples.

Next to the CSV, ``predictions_<model>.timing.json`` records how long the run took:
wall time for the whole command, time spent in the model itself (synchronized on GPU),
the device and hardware, and the Slurm job ID if any. scripts/score_all.sh collects these
into ``<results>/TIMINGS.md``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import socket
import time

import numpy as np
import pandas as pd

from . import data as D
from .cache import cache_path
from .features import MODEL_OUTPUT_ORDER, encode_nodes

# Retrained on post-synthesis labels: github.com/ben-hawks/wa_hls4ml_models, release
# resource-report-retrain (see weights/MANIFEST.json).
GNN_CHECKPOINT = "gnn_resource_report_final_model.pth"
TRANSFORMER_CHECKPOINT = "transformer_resource_report_final_model.pt"
STATS_FILE = "normalization_stats.json"


def _device(name):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class _ModelClock:
    """Accumulates time spent inside the model, synchronizing the GPU so it's honest."""

    def __init__(self, device=None):
        self.seconds = 0.0
        self._cuda = device is not None and getattr(device, "type", str(device)) == "cuda"

    def _sync(self):
        if self._cuda:
            import torch

            torch.cuda.synchronize()

    def __enter__(self):
        self._sync()
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.seconds += time.perf_counter() - self._t


def _load_stats(weights_dir, model):
    with open(os.path.join(weights_dir, STATS_FILE)) as f:
        s = json.load(f)
    m = s["models"][model]
    return (np.asarray(s["feature_means"], np.float32), np.asarray(s["feature_stds"], np.float32),
            np.asarray(m["label_means"], np.float32), np.asarray(m["label_stds"], np.float32),
            float(m["log_shift"]), np.asarray(m["label_max"], np.float32))


def _denormalize(out, label_means, label_stds, log_shift, label_max):
    """Dataset2.denormalize_labels (undo z-score, then log: exp - shift, clamp at 0), then
    cap at the largest training label, as upstream transformer/run.py and
    GNN/load_pretrained.py do: for a few inputs the log-space output extrapolates far past
    anything physically possible (e.g. millions of DSPs)."""
    import torch

    y = out * torch.as_tensor(label_stds, device=out.device) + torch.as_tensor(label_means, device=out.device)
    y = torch.clamp(torch.exp(y) - log_shift, min=0.0)
    return torch.minimum(y, torch.as_tensor(label_max, device=out.device))


def _frame(ids, subsets, preds_model_order, handled):
    """One row per sample; rows the model couldn't handle get empty (NaN) outputs."""
    out = np.full((len(ids), len(MODEL_OUTPUT_ORDER)), np.nan, dtype=np.float32)
    out[handled] = preds_model_order
    df = pd.DataFrame({"sample_id": ids, "subset": subsets})
    for j, col in enumerate(MODEL_OUTPUT_ORDER):
        df[col] = out[:, j]
    return df[["sample_id", "subset"] + D.OUTPUT_COLUMNS]


def predict_gnn(cache, weights_dir, device, batch_size=1024):
    import torch
    from types import SimpleNamespace

    from .models import gnn

    model, _ = gnn.load(os.path.join(weights_dir, GNN_CHECKPOINT), device)
    fm, fs, lm, ls, shift, cap = _load_stats(weights_dir, "gnn")
    handled = cache.feat_ok.copy()
    idx = np.nonzero(handled)[0]
    preds = np.zeros((len(idx), 6), dtype=np.float32)
    clock = _ModelClock(device)
    with torch.no_grad():
        for a in range(0, len(idx), batch_size):
            xs, eis, bs, globs, n0 = [], [], [], [], 0
            for k, i in enumerate(idx[a:a + batch_size]):
                x, g = encode_nodes(cache.raw(i), fm, fs)
                n = len(x)
                src = np.arange(n - 1) + n0  # sequential dataflow edges, as in Dataset2.get
                eis.append(np.stack([src, src + 1]))
                xs.append(x)
                bs.append(np.full(n, k))
                globs.append(g)
                n0 += n
            glob = torch.as_tensor(np.stack(globs), device=device)
            batch = SimpleNamespace(
                x=torch.as_tensor(np.concatenate(xs), device=device),
                edge_index=torch.as_tensor(np.concatenate(eis, axis=1), dtype=torch.long, device=device),
                batch=torch.as_tensor(np.concatenate(bs), dtype=torch.long, device=device),
                strategy=glob[:, :2], io_type=glob[:, 2:],
            )
            with clock:
                out = _denormalize(model(batch), lm, ls, shift, cap)
            preds[a:a + len(out)] = out.cpu().numpy()
    df = _frame(cache.sample_id, cache.subset, preds, handled)
    df.attrs["model_seconds"] = clock.seconds
    return df


def predict_transformer(cache, weights_dir, device, batch_size=1024):
    import torch

    from .models import transformer

    model = transformer.load(os.path.join(weights_dir, TRANSFORMER_CHECKPOINT), device)
    fm, fs, lm, ls, shift, cap = _load_stats(weights_dir, "transformer")
    handled = cache.feat_ok & (cache.n_layers() <= model.max_layers)
    idx = np.nonzero(handled)[0]
    preds = np.zeros((len(idx), 6), dtype=np.float32)
    clock = _ModelClock(device)
    with torch.no_grad():
        for a in range(0, len(idx), batch_size):
            chunk = idx[a:a + batch_size]
            L = int(cache.n_layers()[chunk].max())
            x = np.zeros((len(chunk), L, 33), dtype=np.float32)  # padded rows are zeros, as in get_transformer
            mask = np.ones((len(chunk), L), dtype=bool)
            for k, i in enumerate(chunk):
                nodes, _ = encode_nodes(cache.raw(i), fm, fs)
                x[k, :len(nodes)] = nodes
                mask[k, :len(nodes)] = False
            with clock:
                out = _denormalize(model(torch.as_tensor(x, device=device), torch.as_tensor(mask, device=device)),
                                   lm, ls, shift, cap)
            preds[a:a + len(out)] = out.cpu().numpy()
    df = _frame(cache.sample_id, cache.subset, preds, handled)
    df.attrs["model_seconds"] = clock.seconds
    return df


def predict_mlp(data_root, split, kind="mlp"):
    from .models.mlp import Rule4mlMLP

    mlp = Rule4mlMLP(kind)
    ids, subsets, gl, sl, handled = [], [], [], [], []
    for subset, sample in D.iter_samples(data_root, split):
        built = mlp.inputs(sample)
        ids.append(D.sample_id(sample))
        subsets.append(subset)
        handled.append(built is not None)
        if built is not None:
            gl.append(built[0])
            sl.append(built[1])
    clock = _ModelClock()
    with clock:
        preds = mlp.predict(gl, sl)
    built = pd.DataFrame(preds)[D.OUTPUT_COLUMNS]
    df = pd.DataFrame({"sample_id": ids, "subset": subsets})
    for col in D.OUTPUT_COLUMNS:
        df[col] = np.nan
    df.loc[np.asarray(handled), D.OUTPUT_COLUMNS] = built.to_numpy()
    df.attrs["model_seconds"] = clock.seconds
    return df


def _hardware(device) -> str:
    dev = str(device)
    if dev.startswith("cuda"):
        import torch

        return torch.cuda.get_device_name(torch.device(dev))
    return f"{platform.processor() or platform.machine()} ({os.cpu_count()} logical CPUs visible)"


def timing_path(out_csv: str) -> str:
    base = out_csv[:-4] if out_csv.endswith(".csv") else out_csv
    return base + ".timing.json"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["gnn", "transformer", "mlp", "rule4ml_gnn"],
                   help="rule4ml_gnn: rule4ml's bundled GIN model, auxiliary, not a reference solution")
    p.add_argument("--split", choices=D.SPLITS)
    p.add_argument("--cache-dir", help="directory with <split>.npz (gnn, transformer)")
    p.add_argument("--cache", help="a single cache file (alternative to --cache-dir + --split)")
    p.add_argument("--data-root", help="local dataset copy (mlp, rule4ml_gnn)")
    p.add_argument("--weights-dir", default=os.environ.get("WA_WEIGHTS", "weights"))
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    if args.model in ("mlp", "rule4ml_gnn"):
        if not (args.data_root and args.split):
            p.error(f"--model {args.model} needs --data-root and --split")
        df = predict_mlp(args.data_root, args.split, "mlp" if args.model == "mlp" else "gnn")
        device = "cpu"
    else:
        if args.cache:
            path = args.cache
        elif args.cache_dir and args.split:
            path = cache_path(args.cache_dir, args.split)
        else:
            p.error(f"--model {args.model} needs --cache-dir and --split (or --cache FILE)")
        from .cache import SplitCache

        cache = SplitCache(path)
        device = _device(args.device)
        fn = predict_gnn if args.model == "gnn" else predict_transformer
        df = fn(cache, args.weights_dir, device, args.batch_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    dt = time.perf_counter() - t0
    n = len(df)
    n_missing = int(df[D.OUTPUT_COLUMNS].isna().all(axis=1).sum())
    if n_missing:
        print(f"{args.model}: {n_missing} of {n} samples couldn't be handled; their rows have empty outputs")
    model_s = float(df.attrs.get("model_seconds", float("nan")))
    timing = {
        "model": args.model,
        "split": args.split or os.path.basename(args.cache or "").replace(".npz", ""),
        "device": str(device),
        "hardware": _hardware(device),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "n_samples": n,
        "n_missing": n_missing,
        "seconds_total": round(dt, 3),
        "seconds_model": round(model_s, 3),
        "ms_per_sample_total": round(1e3 * dt / max(n, 1), 4),
        "ms_per_sample_model": round(1e3 * model_s / max(n - n_missing, 1), 4),
        "batch_size": args.batch_size if args.model in ("gnn", "transformer") else None,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(timing_path(args.out), "w") as f:
        json.dump(timing, f, indent=2)
    print(f"{args.model}: wrote {n} predictions to {args.out} ({dt:.1f}s total, "
          f"{timing['ms_per_sample_total']:.3f} ms/sample incl. loading and featurization; "
          f"model only {model_s:.1f}s, {timing['ms_per_sample_model']:.4f} ms/sample; "
          f"device={device}, {timing['hardware']})")


if __name__ == "__main__":
    main()
