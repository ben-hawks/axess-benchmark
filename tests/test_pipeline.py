"""End-to-end smoke test on the fixture samples: cache -> predict -> score.

The GNN/Transformer part needs the checkpoints (WA_WEIGHTS, default ./weights) and
compares against golden predictions recorded from the validated reference run
(docs/VALIDATION.md). Run it after setting up a new machine before submitting jobs:

    WA_WEIGHTS=$WA_WEIGHTS python -m pytest tests -q
    WA_WEIGHTS=$WA_WEIGHTS WA_TEST_DEVICE=cuda python -m pytest tests -q   # on a GPU node
"""

import os

import numpy as np
import pandas as pd
import pytest

from wa_hls4ml_bench import cache as C
from wa_hls4ml_bench import data as D
from wa_hls4ml_bench import score as S
from wa_hls4ml_bench.truth import truth_frame

HERE = os.path.dirname(__file__)
FIXTURES = os.path.join(HERE, "fixtures", "data")
WEIGHTS = os.environ.get("WA_WEIGHTS", os.path.join(HERE, "..", "weights"))
HAVE_WEIGHTS = all(os.path.exists(os.path.join(WEIGHTS, f)) for f in
                   ("gnn_final_model.pth", "transformer_best_model.pt", "normalization_stats.json"))


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    d = tmp_path_factory.mktemp("cache")
    out = {}
    for split in ("test", "exemplar"):
        path = str(d / f"{split}.npz")
        C.build_cache(FIXTURES, split, path, workers=1)
        out[split] = C.SplitCache(path)
    return out


def test_cache_contents(caches):
    test = caches["test"]
    assert len(test) == 32
    assert test.feat_ok.all()
    assert set(test.subset) == {"2_20", "2layer", "3layer", "conv1d", "conv2d", "latency", "resource"}
    # fixture includes samples without a post-synthesis report; they must be NaN, not 0
    assert (~np.isfinite(test.truth_post[:, 0])).sum() > 0
    ex = caches["exemplar"]
    assert {s.split("/")[0] for s in ex.subset} == {"exemplar"}
    assert len({s.split("/")[1] for s in ex.subset}) == 7


def test_truth_excludes_missing(caches):
    tf = truth_frame(caches["test"], D.POST_SYNTHESIS)
    assert np.isfinite(tf[D.TARGETS].to_numpy()).all()
    assert len(tf) == int(np.isfinite(caches["test"].truth_post[:, 0]).sum())


@pytest.mark.skipif(not HAVE_WEIGHTS, reason=f"no checkpoints in {WEIGHTS}")
@pytest.mark.parametrize("model", ["gnn", "transformer"])
@pytest.mark.parametrize("split", ["test", "exemplar"])
def test_matches_golden_predictions(caches, model, split):
    import torch

    from wa_hls4ml_bench import predict as P

    device = torch.device(os.environ.get("WA_TEST_DEVICE", "cpu"))
    fn = P.predict_gnn if model == "gnn" else P.predict_transformer
    got = fn(caches[split], WEIGHTS, device).set_index("sample_id")
    want = pd.read_csv(os.path.join(HERE, "fixtures", f"golden_{split}_{model}.csv"),
                       dtype={"sample_id": str}).set_index("sample_id")
    assert list(got.index) == list(want.index)
    np.testing.assert_allclose(got[D.TARGETS].to_numpy(), want[D.TARGETS].to_numpy(), rtol=2e-3, atol=1e-2)

    tf = truth_frame(caches[split], D.POST_SYNTHESIS)
    res = S.score(tf, got.reset_index())
    assert res["coverage"]["n_truth_without_prediction"] == 0
