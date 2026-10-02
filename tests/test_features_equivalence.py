"""The vendored feature extractor must match the original bit for bit.

Needs a checkout of github.com/ben-hawks/wa_hls4ml_models (tag resource-report-retrain,
or the wa-hls4ml-paper submodule); point WA_MODELS_REPO at it.
Skipped otherwise (the golden-prediction test in test_pipeline.py covers the same path
end to end without it).
"""

import json
import logging
import os
import sys
import tempfile

import numpy as np
import pytest

from wa_hls4ml_bench import data as D
from wa_hls4ml_bench.features import FEATURE_COLUMNS, raw_layer_features

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "data")
REPO = os.environ.get("WA_MODELS_REPO")


@pytest.mark.skipif(not REPO, reason="set WA_MODELS_REPO to a wa_hls4ml_models checkout")
def test_matches_original_model_processor(monkeypatch, capsys):
    sys.path.insert(0, os.path.join(REPO, "dataset"))
    import Dataset_to_csvs6_with_ii as orig

    logging.disable(logging.CRITICAL)
    proc = orig.ModelProcessor()
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for split in ("test", "exemplar"):
            for _, sample in D.iter_samples(FIXTURES, split):
                path = os.path.join(tmp, "s.json")
                with open(path, "w") as f:
                    json.dump(sample, f)
                df, _ = proc.process_json_to_csv(path)
                expected = np.full((len(df), 18), -1.0)
                for j, c in enumerate(FEATURE_COLUMNS):
                    expected[:, j] = df[c].values
                np.testing.assert_array_equal(raw_layer_features(sample, fill_from_global=False), expected)
                # the global-config fallback only ever fills cells the original left NaN
                filled = raw_layer_features(sample)
                known = ~np.isnan(expected)
                np.testing.assert_array_equal(filled[known], expected[known])
                assert np.isfinite(filled).all()
                n += 1
    assert n > 30
