"""Baseline MLP reference model (rule4ml v0.2.0, bundled pretrained v2 weights).

rule4ml ships 6 independent single-target Keras MLPs. Its public
``MultiModelWrapper.predict()`` wants a live Keras model and takes a Cartesian product
over configs, so this calls the lower-level functions it uses internally, feeding each
sample's own ``model_config`` (already in the per-layer schema rule4ml derives from a
Keras model) and ``hls_config``. See docs/VALIDATION.md for the history of this path.
"""

from __future__ import annotations

import json
import os

import numpy as np

TARGET_MAP = {"BRAM": "BRAM", "DSP": "DSP", "FF": "FF", "LUT": "LUT",
              "CYCLES": "cycles_max", "INTERVAL": "interval_max"}

VALID_HLS4ML = {"0.8.1", "1.1.0"}
VALID_VIVADO = {"2019.1", "2019.2", "2020.1", "2020.2", "2021.1", "2021.2",
                "2022.1", "2022.2", "2023.1", "2023.2", "2024.1", "2024.2"}
# Only used if a sample's version is outside what rule4ml's categorical maps know about.
# The MLP's inputs do not include vivado_version, so this has no effect on its outputs.
FALLBACK_VIVADO = "2024.2"


class Rule4mlMLP:
    def __init__(self):
        import rule4ml
        from rule4ml.models.wrappers import KerasModelWrapper

        root = os.path.dirname(rule4ml.__file__)
        boards = json.load(open(os.path.join(root, "parsers", "supported_boards.json")))
        self.part_to_board = {v["part"]: k for k, v in boards.items()}
        base = os.path.join(root, "models", "weights", "v2", "mlp")
        self.wrappers = {}
        for t in TARGET_MAP:
            w = KerasModelWrapper()
            w.load(f"{base}/{t}.config.json", f"{base}/{t}.weights.h5")
            self.wrappers[t] = w

    def inputs(self, sample):
        """(global_inputs, sequential_inputs), or None if rule4ml cannot represent the sample."""
        from rule4ml.models.wrappers import get_global_inputs, get_layers_data

        raw_hls = sample["hls_config"]
        precision = raw_hls["Model"]["Precision"]
        if isinstance(precision, dict):
            precision = precision.get("default", precision)
        board = self.part_to_board.get(sample.get("target_part"))
        if board is None:
            return None
        hls_config = {
            "board": board,
            "model": {
                "strategy": raw_hls["Model"]["Strategy"],
                "precision": precision,
                "reuse_factor": raw_hls["Model"]["ReuseFactor"],
            },
        }
        hls4ml_version = sample.get("hls4ml_version")
        if hls4ml_version not in VALID_HLS4ML:
            hls4ml_version = None
        vivado_version = sample.get("vivado_version", sample.get("backend_version"))
        if vivado_version not in VALID_VIVADO:
            vivado_version = FALLBACK_VIVADO
        try:
            g = get_global_inputs(sample["model_config"], hls_config,
                                  clock_period=raw_hls.get("clock_period"),
                                  hls4ml_version=hls4ml_version, vivado_version=vivado_version)
            s = get_layers_data(sample["model_config"])
        except Exception:
            return None
        return g, s

    def predict(self, globals_list, seqs_list, batch_size=4000) -> dict:
        from rule4ml.models.wrappers import to_dataframe

        n = len(globals_list)
        out = {}
        for t, wrapper in self.wrappers.items():
            vals = np.zeros(n)
            for a in range(0, n, batch_size):
                b = min(a + batch_size, n)
                df = to_dataframe(
                    meta_data=[{} for _ in range(a, b)],
                    global_inputs=[dict(g) for g in globals_list[a:b]],
                    sequential_inputs=[list(s) for s in seqs_list[a:b]],
                    global_categorical_maps=wrapper.global_categorical_maps,
                    sequential_categorical_maps=wrapper.sequential_categorical_maps,
                    targets=[{} for _ in range(a, b)],
                )
                vals[a:b] = np.asarray(wrapper.predict_from_df(df, verbose=0)).reshape(-1)
            out[TARGET_MAP[t]] = vals
        return out
