#!/bin/bash
# Write truth CSVs from the caches, score every predictions_*.csv under $WA_RESULTS/<split>/,
# then build $WA_RESULTS/LEADERBOARD.md and package every model with predictions for
# both splits as an upload-ready Codabench submission ($WA_RESULTS/codabench/).
# Works on any machine (Perlmutter or local); needs WA_CACHE, WA_RESULTS, WA_SPLITS and the package on PYTHONPATH (perlmutter/env.sh).
set -euo pipefail
: "${WA_SPLITS:=test exemplar}"

for split in $WA_SPLITS; do
    dir="$WA_RESULTS/$split"
    mkdir -p "$dir"
    python -m wa_hls4ml_bench.truth --cache "$WA_CACHE/$split.npz" --out "$dir/truth.csv"
    shopt -s nullglob
    for pred in "$dir"/predictions_*.csv; do
        name=$(basename "$pred" .csv); name=${name#predictions_}
        python -m wa_hls4ml_bench.score --pred "$pred" --truth "$dir/truth.csv" \
            --out "$dir/$name" --title "$name on $split (ground truth: post-synthesis resources, HLS latency)" > /dev/null
        echo "scored $split/$name"
    done
done
python -m wa_hls4ml_bench.report --results "$WA_RESULTS" --out "$WA_RESULTS/LEADERBOARD.md"
# Codabench submission zips: a submission needs both the test and exemplar splits.
if [ -f "$WA_CACHE/test.npz" ] && [ -f "$WA_CACHE/exemplar.npz" ]; then
    python -m wa_hls4ml_bench.submission --results "$WA_RESULTS" --cache-dir "$WA_CACHE"
else
    echo "skipping Codabench submissions: need both $WA_CACHE/test.npz and exemplar.npz"
fi
