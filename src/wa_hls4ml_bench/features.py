"""Per-layer feature extraction and encoding for the GNN and Transformer reference models.

Two stages, both reproducing the preprocessing the published checkpoints were trained with:

1. ``raw_layer_features(sample)`` -> (n_layers, 18) raw per-layer features. Vendored from
   ``ModelProcessor`` in wa_hls4ml_models/dataset/Dataset_to_csvs6_with_ii.py
   (commit 4aff94b; feature code unchanged through resource-report-retrain, ac394e9). The logic is kept as close to the original as possible, including its
   quirks, because the checkpoints only work on features built exactly this way. The only
   change is that it takes an in-memory sample instead of a one-sample JSON file.
   tests/test_features_equivalence.py checks it against the original.

2. ``encode_nodes(...)`` -> (n_layers, 33) normalized node features plus the 4-dim global
   (strategy, io_type) one-hot. Vectorized port of ``FPGAGraphDataset.get`` /
   ``get_transformer`` in wa_hls4ml_models/transformer/GNN/Dataset2.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Stage 1: raw features (vendored; do not "clean up" -- see module docstring)
# --------------------------------------------------------------------------------------

INDEPENDENT_LAYER_NAMES = [
    'Dense', 'QConv2D', 'QDense', 'QConv1D', 'MaxPooling2D', 'MaxPooling1D',
    'AveragePooling2D', 'AveragePooling1D', 'Flatten', 'QDepthwiseConv2D',
    'QDepthwiseConv1D', 'QSeparableConv2D', 'QSeparableConv1D',
    'Activation', 'QActivation',
    'BatchNormalization',
]

LAYER_TYPE_MAPPING = {
    'activation': 0,
    'QConv2D': 3,
    'Dense': 1,
    'QDense': 1,
    'QConv1D': 2,
    'MaxPooling2D': 9,
    'MaxPooling1D': 9,
    'AveragePooling2D': 10,
    'AveragePooling1D': 10,
    'Flatten': 8,
    'QDepthwiseConv2D': 7,
    'QDepthwiseConv1D': 6,
    'QSeparableConv2D': 5,
    'QSeparableConv1D': 4,
    'Activation': 0,
    'QActivation': 0,
    'BatchNormalization': 11,
}

STRATEGY_MAPPING = {'latency': 0, 'resource': 1}

ACTIVATION_MAPPING = {
    'NA': 0,
    'linear': 1,
    'relu': 2,
    'quantized_relu': 2,
    'tanh': 3,
    'quantized_tanh': 3,
    'sigmoid': 4,
    'quantized_sigmoid': 4,
    'quantized_softmax': 5,
}

FEATURES = {
    "d_in1": 0, "d_in2": 1, "d_in3": 2,
    "d_out1": 3, "d_out2": 4, "d_out3": 5,
    "prec": 6, "rf": 7, "strategy": 8,
    "layer_type": 9, "activation_type": 10,
    "filters": 11, "kernel_size": 12, "stride": 13, "padding": 14, "pooling": 15,
    "batchnorm": 16, "io_type": 17,
}
FEATURE_COLUMNS = list(FEATURES.keys())
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_COLUMNS)}

PADDING_MAPPING = {'NA': 0, 'valid': 1, 'same': 2}
IO_TYPE_MAPPING = {'io_parallel': 0, 'io_stream': 1}


def _get_layers(data: Dict) -> Tuple[List[str], List[int]]:
    layers, layers_indices = [], []
    model_config = data.get('model_config', [])
    for i, layer in enumerate(model_config):
        class_name = layer.get('class_name', '')
        if class_name in INDEPENDENT_LAYER_NAMES:
            layers.append(class_name)
            layers_indices.append(i)
        if i == len(model_config) - 1:
            layers_indices.append(i)
    return layers, layers_indices


def _parse_weight_string(weight_str: str) -> Optional[str]:
    try:
        start_index = weight_str.find("<") + 1
        end_index = weight_str.find(">")
        if start_index > 0 and end_index > start_index:
            return weight_str[start_index:end_index].split(',')[0]
        return None
    except (ValueError, AttributeError, IndexError):
        return None


def _get_layer_info(data: Dict, layer_index: int) -> List[Any]:
    layer_info = [None] * len(FEATURE_COLUMNS)
    model_info = data['model_config'][layer_index]

    input_shape = model_info.get('input_shape', [0])[1:]
    output_shape = model_info.get('output_shape', [0])[1:]
    for i in range(3):
        layer_info[i] = input_shape[i] if i < len(input_shape) else 0
    for i in range(3):
        layer_info[i + 3] = output_shape[i] if i < len(output_shape) else 0

    class_name = model_info.get('class_name', '')
    layer_info[FEATURE_INDEX['batchnorm']] = 1 if class_name == "BatchNormalization" else 0

    if class_name in ('Activation', 'QActivation'):
        layer_info[FEATURE_INDEX['layer_type']] = LAYER_TYPE_MAPPING[class_name]
        layer_info[FEATURE_INDEX['pooling']] = 0
        activation_value = model_info.get('activation', 'linear')
        activation_type = 0
        for activation_name, activation_code in ACTIVATION_MAPPING.items():
            if activation_name in activation_value:
                activation_type = activation_code
                break
        layer_info[FEATURE_INDEX['activation_type']] = activation_type
    elif class_name in ('MaxPooling2D', 'MaxPooling1D'):
        layer_info[FEATURE_INDEX['pooling']] = 2
        layer_info[FEATURE_INDEX['layer_type']] = LAYER_TYPE_MAPPING['MaxPooling1D']
        layer_info[FEATURE_INDEX['activation_type']] = 0
    elif class_name in ('AveragePooling2D', 'AveragePooling1D'):
        layer_info[FEATURE_INDEX['pooling']] = 2
        layer_info[FEATURE_INDEX['layer_type']] = LAYER_TYPE_MAPPING['AveragePooling1D']
        layer_info[FEATURE_INDEX['activation_type']] = 0
    else:
        layer_info[FEATURE_INDEX['pooling']] = 0
        layer_info[FEATURE_INDEX['activation_type']] = 0

    if class_name in LAYER_TYPE_MAPPING and class_name not in ('Activation', 'QActivation'):
        layer_info[FEATURE_INDEX['layer_type']] = LAYER_TYPE_MAPPING[class_name]

    for param in ['filters', 'kernel_size', 'stride', 'padding']:
        layer_info[FEATURE_INDEX[param]] = 0
    if class_name in ('QConv2D', 'QConv1D'):
        layer_info[FEATURE_INDEX['filters']] = model_info['filters']
        layer_info[FEATURE_INDEX['kernel_size']] = model_info['kernel_size'][0]
        layer_info[FEATURE_INDEX['stride']] = model_info['strides'][0]
        layer_info[FEATURE_INDEX['padding']] = PADDING_MAPPING.get(model_info['padding'], 1)

    return layer_info


def _process_hls_config(data: Dict, input_features: pd.DataFrame) -> None:
    hls_config = data.get('hls_config', {})
    model_config = hls_config.get('Model', {})
    layer_configs = hls_config.get('LayerName', {})

    global_io_type = hls_config.get('io_type', None)
    if global_io_type is not None:
        io_type_val = IO_TYPE_MAPPING.get(global_io_type, -1)
    else:
        model_layers = data.get('model_config', [])
        has_conv = any(
            l.get('class_name', '').lower() in [
                'qconv1d', 'qconv2d', 'conv1d', 'conv2d',
                'qdepthwiseconv1d', 'qdepthwiseconv2d',
                'qseparableconv1d', 'qseparableconv2d',
            ] for l in model_layers
        )
        io_type_val = IO_TYPE_MAPPING['io_stream' if has_conv else 'io_parallel']
    input_features.loc[:, 'io_type'] = io_type_val

    if not layer_configs or len(layer_configs) == 0:
        global_prec = model_config.get('Precision', None)
        global_rf = model_config.get('ReuseFactor', None)
        global_strategy = model_config.get('Strategy', 'latency')
        if isinstance(global_prec, dict):
            global_prec = global_prec.get('weight', None)
        if global_prec is not None:
            input_features['prec'] = _parse_weight_string(global_prec)
        if global_rf is not None:
            input_features['rf'] = global_rf
        input_features['strategy'] = STRATEGY_MAPPING.get(str(global_strategy).lower(), 0)
        return

    layer_configs_new = {
        k: v for k, v in layer_configs.items()
        if not any(x in k for x in ['alpha', 'input'])
    }

    input_features_layer_type = input_features['layer_type'].tolist()
    layer_configs_new_list = list(layer_configs_new.keys())
    layers_to_remove = []

    input_features_count = 0
    for i, layer_name in enumerate(layer_configs_new_list):
        if "linear" in layer_name:
            if i + 1 < len(layer_configs_new_list):
                next_layer_name = layer_configs_new_list[i + 1]
                if ("activation" in next_layer_name or "flatten" in next_layer_name
                        or input_features_layer_type[input_features_count] != 0):
                    layers_to_remove.append(layer_name)
                    input_features_count -= 1
            elif i == len(layer_configs_new_list) - 1:
                if input_features_layer_type[-1] != 0:
                    layers_to_remove.append(layer_name)
        input_features_count += 1

    for layer_name in layers_to_remove:
        del layer_configs_new[layer_name]

    if len(layer_configs_new) != len(input_features):
        logger.warning(
            "Mismatch in number of layers: %d in HLS config, %d in model",
            len(layer_configs_new), len(input_features),
        )
        return

    for i, layer_name in enumerate(layer_configs_new.keys()):
        layer_config = layer_configs_new[layer_name]
        if ("conv" in layer_name or "dense" in layer_name) and "linear" not in layer_name \
                and "activation" not in layer_name:
            try:
                weight = _parse_weight_string(layer_config.get('Precision', {}).get('weight', ''))
                input_features.at[i, 'rf'] = layer_config.get('ReuseFactor', 0)
            except Exception as e:
                logger.warning("Error processing layer %s: %s", layer_name, e)
                continue
        else:
            if i > 0:
                input_features.at[i, 'rf'] = 1
                weight = input_features.iloc[i - 1]['prec']
            else:
                continue
        input_features.at[i, 'prec'] = weight

    strategy_value = model_config.get('Strategy', 'latency')
    input_features['strategy'] = STRATEGY_MAPPING.get(strategy_value.lower(), 0)


def raw_layer_features(sample: Dict, fill_from_global: bool = True) -> np.ndarray:
    """(n_layers, 18) float64 raw features for one sample.

    With ``fill_from_global=False`` this is exactly the original extractor (unset entries
    are NaN). The default additionally applies ``_fill_from_global_model_config``.
    """
    names, indices = _get_layers(sample)
    rows = [_get_layer_info(sample, indices[i]) for i in range(len(names))]
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=object)
    try:
        _process_hls_config(sample, df)
    except Exception as e:  # the original also logs and keeps the partial features
        logger.error("Error processing HLS configuration: %s", e)
    out = np.empty((len(df), len(FEATURE_COLUMNS)), dtype=float)
    for j, col in enumerate(FEATURE_COLUMNS):
        out[:, j] = df[col].values  # same object->float cast as the original np.full fill
    if fill_from_global:
        _fill_from_global_model_config(sample, out)
    return out


def _fill_from_global_model_config(sample: Dict, out: np.ndarray) -> None:
    """Benchmark addition, not in the original extractor.

    When ``hls_config.LayerName`` cannot be matched to the model's layers, the original
    returns early and leaves prec/rf/strategy unset (NaN), so the reference models output
    NaN. On the published data this hits exactly the 119 Bipc samples of the exemplar set
    and no test-set sample. Those NaN entries are filled from the global
    ``hls_config.Model`` Precision/ReuseFactor/Strategy, the same values the original uses
    when a sample has no per-layer config at all. Entries that are already set are never
    changed, so every sample the original handles is bit-identical.
    """
    cols = [FEATURE_INDEX["prec"], FEATURE_INDEX["rf"], FEATURE_INDEX["strategy"]]
    if not np.isnan(out[:, cols]).any():
        return
    model_config = sample.get('hls_config', {}).get('Model', {})
    prec = model_config.get('Precision', None)
    if isinstance(prec, dict):
        prec = prec.get('weight', None)
    prec = _parse_weight_string(prec) if prec is not None else None
    values = {
        "prec": float(prec) if prec is not None else np.nan,
        "rf": float(model_config.get('ReuseFactor', np.nan)),
        "strategy": float(STRATEGY_MAPPING.get(str(model_config.get('Strategy', 'latency')).lower(), 0)),
    }
    for name, val in values.items():
        col = out[:, FEATURE_INDEX[name]]
        col[np.isnan(col)] = val


# --------------------------------------------------------------------------------------
# Stage 2: encoding (vectorized port of Dataset2.FPGAGraphDataset)
# --------------------------------------------------------------------------------------

NUMERICAL_FEATURE_KEYS = [
    "d_in1", "d_in2", "d_in3", "d_out1", "d_out2", "d_out3",
    "prec", "rf", "filters", "kernel_size", "stride", "pooling",
]
NUMERICAL_IDX = np.array([FEATURES[k] for k in NUMERICAL_FEATURE_KEYS])
NUM_LAYER_TYPES = 12      # len(set(layer_type_mapping_provided.values()))
NUM_ACTIVATION_TYPES = 6
NUM_PADDING_TYPES = 3
NUM_IO_TYPES = 2
NODE_FEATURE_DIM = len(NUMERICAL_FEATURE_KEYS) + NUM_LAYER_TYPES + NUM_ACTIVATION_TYPES + NUM_PADDING_TYPES  # 33

# Order of the 6 outputs of both checkpoints -> benchmark column names.
MODEL_OUTPUT_ORDER = ["cycles_max", "FF", "LUT", "BRAM", "DSP", "interval_max"]  # CYCLES,FF,LUT,BRAM,DSP,II


def _one_hot(values: np.ndarray, num_classes: int) -> np.ndarray:
    """Matches Dataset2._one_hot_encode: out-of-range or NaN -> all-zero row."""
    out = np.zeros((len(values), num_classes), dtype=np.float32)
    ok = np.isfinite(values)
    ints = np.zeros(len(values), dtype=np.int64)
    ints[ok] = values[ok].astype(np.int64)  # int() truncation, as in the original
    ok &= (ints >= 0) & (ints < num_classes)
    out[np.nonzero(ok)[0], ints[ok]] = 1.0
    return out


def encode_nodes(raw: np.ndarray, feature_means: np.ndarray, feature_stds: np.ndarray):
    """raw: (n_layers, 18) for one sample (no padding rows).

    Returns (x, globals): x is (n_layers, 33) float32, globals is (4,) float32
    = [strategy one-hot (2), io_type one-hot (2)] taken from the first layer.
    """
    num = raw[:, NUMERICAL_IDX].astype(np.float32)
    num = np.where(num == -1.0, np.float32(0.0), num)
    num = (num - feature_means) / feature_stds
    x = np.concatenate([
        num,
        _one_hot(raw[:, FEATURES["layer_type"]], NUM_LAYER_TYPES),
        _one_hot(raw[:, FEATURES["activation_type"]], NUM_ACTIVATION_TYPES),
        _one_hot(raw[:, FEATURES["padding"]], NUM_PADDING_TYPES),
    ], axis=1).astype(np.float32)
    first = raw[:1]
    glob = np.concatenate([
        _one_hot(first[:, FEATURES["strategy"]], 2)[0],
        _one_hot(first[:, FEATURES["io_type"]], NUM_IO_TYPES)[0],
    ])
    return x, glob
