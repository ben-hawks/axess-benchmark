"""Codabench submission packaging (wa_hls4ml_bench.submission) on the fixture samples."""

import os
import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest

from wa_hls4ml_bench import cache as C
from wa_hls4ml_bench import data as D
from wa_hls4ml_bench import submission as SUB

HERE = os.path.dirname(__file__)
FIXTURES = os.path.join(HERE, "fixtures", "data")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("sub")
    cache_dir, results = root / "cache", root / "results"
    cache_dir.mkdir()
    for split in SUB.SPLITS:
        C.build_cache(FIXTURES, split, str(cache_dir / f"{split}.npz"), workers=1)
        (results / split).mkdir(parents=True)
        # golden GNN predictions cover every fixture sample, scored or not
        shutil.copy(os.path.join(HERE, "fixtures", f"golden_{split}_gnn.csv"),
                    results / split / "predictions_gnn.csv")
    return str(cache_dir), str(results)


def test_packages_scored_samples_only(setup):
    cache_dir, results = setup
    with pytest.raises(SystemExit) as e:
        SUB.main(["--results", results, "--cache-dir", cache_dir])
    assert e.value.code == 0
    zip_path = os.path.join(results, "codabench", "gnn_submission.zip")
    with zipfile.ZipFile(zip_path) as z:
        assert sorted(z.namelist()) == ["predictions_exemplar.csv", "predictions_test.csv"]
        for split in SUB.SPLITS:
            got = pd.read_csv(z.open(f"predictions_{split}.csv"), dtype={"sample_id": str})
            want = SUB.scored_ids(cache_dir, split)
            assert list(got.columns) == SUB.COLUMNS
            assert list(got["sample_id"]) == list(want)  # exactly the scored ids, in order
            assert np.isfinite(got[D.TARGETS].to_numpy()).all()
    # the fixture test split includes unscored samples; they must have been dropped
    golden = pd.read_csv(os.path.join(HERE, "fixtures", "golden_test_gnn.csv"))
    assert len(golden) > len(SUB.scored_ids(cache_dir, "test"))


def test_incomplete_predictions_are_not_packaged(setup, tmp_path):
    cache_dir, results = setup
    bad = tmp_path / "results"
    shutil.copytree(results, bad, ignore=shutil.ignore_patterns("codabench"))
    test_csv = bad / "test" / "predictions_gnn.csv"
    pred = pd.read_csv(test_csv, dtype={"sample_id": str})
    scored = set(SUB.scored_ids(cache_dir, "test"))
    drop = pred.index[pred["sample_id"].isin(scored)][0]
    pred.drop(index=drop).to_csv(test_csv, index=False)
    with pytest.raises(SystemExit) as e:
        SUB.main(["--results", str(bad), "--cache-dir", cache_dir])
    assert e.value.code == 1
    assert not os.path.exists(bad / "codabench" / "gnn_submission.zip")
