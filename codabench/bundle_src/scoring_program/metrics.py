"""Metrics for the wa-hls4ml scoring program.

A self-contained copy of the formulas in the benchmark's src/wa_hls4ml_bench/score.py
(paper Section 3.2, Eq. 1-3). Codabench uploads scoring_program/ on its own, so it can't
import from the benchmark package; keep the two in sync.
"""

import numpy as np

SMAPE_EPSILON = 1.0  # the smallest strictly positive value the integer counts can take


def r_squared(y_true, y_pred):
    """Eq. 1. NaN when the ground truth has zero variance."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)


def smape(y_true, y_pred, epsilon=SMAPE_EPSILON):
    """Eq. 2, in percent (0-200)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(200.0 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)))


def rmse(y_true, y_pred):
    """Eq. 3, native units."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
