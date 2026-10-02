# Starting kit: AXESS wa-hls4ml Benchmark

## Contents

| File | What |
|---|---|
| `make_submission.py` | checks your two prediction CSVs and zips them for upload |
| `test_sample_ids.csv` | the 92,933 test `sample_id`s that are scored |
| `exemplar_sample_ids.csv` | the 886 exemplar `sample_id`s that are scored |
| `sample_submission.zip` | a ready-to-upload example: the reference Transformer's predictions |

## Workflow

1. Download the dataset (see the **Data** page).
2. Train your model on `train/` (and `val/`), using only `model_config`, `hls_config`
   and the constraint fields as inputs.
3. Predict `BRAM, DSP, FF, LUT, cycles_max, interval_max` for every sample in `test/`
   and `exemplar/`, and write one CSV per split with a `sample_id` column plus those six.
4. Package and check:

   ```bash
   python make_submission.py --test predictions_test.csv \
       --exemplar predictions_exemplar.csv --out submission.zip
   ```

5. Upload `submission.zip` on the **My Submissions** tab.

`make_submission.py` needs `numpy` and `pandas`. It rejects anything the scoring
program would reject: missing columns, duplicate ids, a missing or non-finite prediction
for a scored sample.

## Running the reference models yourself

The benchmark repository
[github.com/ben-hawks/axess-benchmark](https://github.com/ben-hawks/axess-benchmark) has
the three reference solutions with pretrained weights, the exact preprocessing they
need, the scoring code, and Slurm workflows for NERSC Perlmutter. Its
`python -m wa_hls4ml_bench.predict` writes CSVs that `make_submission.py` accepts
directly. To produce `sample_submission.zip`:

```bash
python -m wa_hls4ml_bench.predict --model transformer --cache test.npz --out predictions_test.csv
python -m wa_hls4ml_bench.predict --model transformer --cache exemplar.npz --out predictions_exemplar.csv
python make_submission.py --test predictions_test.csv --exemplar predictions_exemplar.csv
```

See that repository's `docs/PERLMUTTER.md` for building the caches and fetching the
weights.
