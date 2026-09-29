import math

import numpy as np
import pandas as pd

from wa_hls4ml_bench import score as S


def test_smape_epsilon_handles_all_zero():
    # both zero -> 0% (epsilon=1 avoids 0/0), not NaN
    assert S.smape([0, 0], [0, 0]) == 0.0
    assert math.isclose(S.smape([0], [1]), 200 * 1 / 2)


def test_r_squared_zero_variance_is_nan():
    assert math.isnan(S.r_squared([3, 3, 3], [1, 2, 3]))
    assert S.r_squared([1, 2, 3], [1, 2, 3]) == 1.0


def test_rpe_sign_positive_is_underprediction():
    assert S.relative_percent_error([10], [5])[0] > 0


def _frame(ids, subsets, vals):
    df = pd.DataFrame({"sample_id": ids, "subset": subsets})
    for t in S.TARGETS:
        df[t] = vals
    return df


def test_score_groups_and_coverage():
    truth = _frame(["a", "b", "c", "d"], ["3layer", "3layer", "conv1d", "conv2d"], [1.0, 2.0, 3.0, 4.0])
    pred = _frame(["a", "b", "c", "x"], ["3layer", "3layer", "conv1d", "conv2d"], [1.0, 2.0, 3.0, 9.0])
    res = S.score(truth, pred)
    assert res["coverage"] == {"n_truth": 4, "n_scored": 3, "n_truth_without_prediction": 1,
                               "n_predictions_without_truth": 1}
    assert set(res["groups"]) == {"all", "dense", "conv1d"}
    assert res["groups"]["all"]["LUT"]["r_squared"] == 1.0


def test_score_rejects_nan_predictions():
    truth = _frame(["a"], ["3layer"], [1.0])
    pred = _frame(["a"], ["3layer"], [np.nan])
    try:
        S.score(truth, pred)
    except ValueError:
        return
    raise AssertionError("NaN predictions must be rejected")
