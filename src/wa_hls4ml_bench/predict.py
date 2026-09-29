"""Run a pretrained reference model over a dataset split and write a predictions CSV.

    # GNN / Transformer read the featurized cache (see cache.py)
    python -m wa_hls4ml_bench.predict --model gnn --cache $WA_CACHE/test.npz \
        --weights-dir $WA_WEIGHTS --out results/test/predictions_gnn.csv

    # the baseline MLP (rule4ml) reads the raw JSON directly
    python -m wa_hls4ml_bench.predict --model mlp --data-root $WA_DATA --split test \
        --out results/test/predictions_mlp.csv

Output columns: sample_id, subset, BRAM, DSP, FF, LUT, cycles_max, interval_max.
Every sample whose features could be built gets a prediction, whether or not it has
ground truth. Scoring (score.py) joins on sample_id against a truth CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import data as D
from .features import MODEL_OUTPUT_ORDER, encode_nodes

GNN_CHECKPOINT = "gnn_final_model.pth"
TRANSFORMER_CHECKPOINT = "transformer_best_model.pt"
STATS_FILE = "normalization_stats.json"


def _device(name):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_stats(weights_dir, model):
    with open(os.path.join(weights_dir, STATS_FILE)) as f:
        s = json.load(f)
    m = s["models"][model]
    return (np.asarray(s["feature_means"], np.float32), np.asarray(s["feature_stds"], np.float32),
            np.asarray(m["label_means"], np.float32), np.asarray(m["label_stds"], np.float32),
            float(m["log_shift"]))


def _denormalize(out, label_means, label_stds, log_shift):
    """Dataset2.denormalize_labels: undo z-score, then log (exp - shift), clamp at 0."""
    import torch

    y = out * torch.as_tensor(label_stds, device=out.device) + torch.as_tensor(label_means, device=out.device)
    return torch.clamp(torch.exp(y) - log_shift, min=0.0)


def _to_frame(ids, subsets, preds_model_order):
    df = pd.DataFrame({"sample_id": ids, "subset": subsets})
    for j, col in enumerate(MODEL_OUTPUT_ORDER):
        df[col] = preds_model_order[:, j]
    return df[["sample_id", "subset"] + D.TARGETS]


def predict_gnn(cache, weights_dir, device, batch_size=1024):
    import torch
    from types import SimpleNamespace

    from .models import gnn

    model, _ = gnn.load(os.path.join(weights_dir, GNN_CHECKPOINT), device)
    fm, fs, lm, ls, shift = _load_stats(weights_dir, "gnn")
    idx = np.nonzero(cache.feat_ok)[0]
    preds = np.zeros((len(idx), 6), dtype=np.float32)
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
            out = _denormalize(model(batch), lm, ls, shift)
            preds[a:a + len(out)] = out.cpu().numpy()
    return _to_frame(cache.sample_id[idx], cache.subset[idx], preds)


def predict_transformer(cache, weights_dir, device, batch_size=1024):
    import torch

    from .models import transformer

    model = transformer.load(os.path.join(weights_dir, TRANSFORMER_CHECKPOINT), device)
    fm, fs, lm, ls, shift = _load_stats(weights_dir, "transformer")
    ok = cache.feat_ok & (cache.n_layers() <= model.max_layers)
    idx = np.nonzero(ok)[0]
    preds = np.zeros((len(idx), 6), dtype=np.float32)
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
            out = model(torch.as_tensor(x, device=device), torch.as_tensor(mask, device=device))
            preds[a:a + len(out)] = _denormalize(out, lm, ls, shift).cpu().numpy()
    return _to_frame(cache.sample_id[idx], cache.subset[idx], preds)


def predict_mlp(data_root, split):
    from .models.mlp import Rule4mlMLP

    mlp = Rule4mlMLP()
    ids, subsets, gl, sl, skipped = [], [], [], [], 0
    for subset, sample in D.iter_samples(data_root, split):
        built = mlp.inputs(sample)
        if built is None:
            skipped += 1
            continue
        ids.append(D.sample_id(sample))
        subsets.append(subset)
        gl.append(built[0])
        sl.append(built[1])
    print(f"mlp: {len(ids)} samples built, {skipped} not representable by rule4ml", flush=True)
    preds = mlp.predict(gl, sl)
    df = pd.DataFrame({"sample_id": ids, "subset": subsets, **preds})
    return df[["sample_id", "subset"] + D.TARGETS]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["gnn", "transformer", "mlp"])
    p.add_argument("--cache", help="featurized split .npz (gnn, transformer)")
    p.add_argument("--data-root", help="local dataset copy (mlp)")
    p.add_argument("--split", choices=D.SPLITS, help="split name (mlp)")
    p.add_argument("--weights-dir", default=os.environ.get("WA_WEIGHTS", "weights"))
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    t0 = time.time()
    if args.model == "mlp":
        if not (args.data_root and args.split):
            p.error("--model mlp needs --data-root and --split")
        df = predict_mlp(args.data_root, args.split)
        device = "cpu"
    else:
        if not args.cache:
            p.error(f"--model {args.model} needs --cache")
        from .cache import SplitCache

        cache = SplitCache(args.cache)
        device = _device(args.device)
        fn = predict_gnn if args.model == "gnn" else predict_transformer
        df = fn(cache, args.weights_dir, device, args.batch_size)
        if len(df) < len(cache):
            print(f"{args.model}: no prediction for {len(cache) - len(df)} of {len(cache)} samples", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    dt = time.time() - t0
    print(f"{args.model}: wrote {len(df)} predictions to {args.out} "
          f"({dt:.1f}s total, {1e3 * dt / max(len(df), 1):.3f} ms/sample incl. featurization, device={device})")


if __name__ == "__main__":
    main()
