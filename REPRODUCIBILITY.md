# Reproducibility Statement

This package is engineered for end-to-end reproducibility. Anyone with the
software prerequisites in `requirements.txt` can re-run `bash run_all.sh`
and produce numerically equivalent results.

## What is guaranteed

- **Data identity.** Yahoo Finance daily closes for the seven tickers
  over 2020-01-01 to 2024-12-31 are fetched once and cached under
  `results/raw/data/`. Yahoo serves historical data deterministically.
  Cached CSVs ship with the released artefact for offline reruns.
- **Model identity.** The fifteen random seeds are fixed in `seeds.txt`.
  All deterministic operations (Adam, MinMaxScaler, MSE loss, Wilcoxon
  test, bootstrap) produce the same output for the same seed.
- **Hyper-parameter identity.** Learning rate, batch size, optimiser
  choice, early-stopping patience, sequence length, train/val/test split
  proportions are all hard-coded constants. Hyperparameters are never
  tuned on the test split.

## What is variable

- **BLAS/CUDA reductions.** GPU non-determinism in CUDA reductions can
  shift the third decimal place of an RMSE figure. Setting
  `torch.use_deterministic_algorithms(True)` and the appropriate env
  variables eliminates this at a small speed cost; see PyTorch docs.
- **Yahoo Finance back-corrections.** Yahoo occasionally revises split
  or dividend adjustments. The CSV cache pins data as of first fetch.

## How to verify

After a clean run on a fresh machine:

```
python -m src.aggregate
diff <(md5sum results/tables/*.csv | sort) reference_hashes.txt
```

A perfect match is expected on identical software stacks. Hash drift
within the third decimal place is reported in `KNOWN_VARIANCE.md`.
