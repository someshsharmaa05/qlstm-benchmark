# Synthetic Data — GARCH(1,1) Generated Series

**All files in this directory are synthetic.** They are generated reproducibly
by the GARCH(1,1) process in `../src/authentic_smoke_test.py` using the seeds
listed in `../seeds.txt`. No real market data is contained here.

## Files

There are 20 CSV files, one per (asset class × seed) combination:

- 4 asset classes: `EQUITY_MID`, `EQUITY_HIGH`, `FX`, `CRYPTO`
- 5 seeds: 42, 137, 271, 389, 521

Naming: `{ASSET}_seed{SEED}.csv`, e.g. `CRYPTO_seed42.csv`.

## Columns

| Column          | Meaning                                                       |
| --------------- | ------------------------------------------------------------- |
| `seq_idx`       | Sequence index (each row is one 30-step input window; 3069 rows per file) |
| `target_return` | Next-step log return to be predicted (the regression target)  |
| `feat_logret`   | Log return at the final time step of the window (scaled)      |
| `feat_sign`     | Sign of the return at the final step (scaled)                 |
| `feat_absret`   | Absolute return at the final step (scaled)                    |
| `feat_rollmean` | 5-step rolling mean of returns at the final step (scaled)     |

The full 30-step feature window for each sequence is reconstructed by the
data pipeline in `src/authentic_smoke_test.py` (`make_dataset`); these CSVs
expose the target and the final-step feature vector for transparency and
quick inspection.

## GARCH parameters per asset class

| Asset class  | ω        | α    | β    | Annualised σ (realised) |
| ------------ | -------- | ---- | ---- | ----------------------- |
| FX           | 2.0e-7   | 0.05 | 0.93 | 0.056                   |
| EQUITY_MID   | 2.0e-6   | 0.08 | 0.90 | 0.183                   |
| EQUITY_HIGH  | 1.5e-5   | 0.10 | 0.85 | 0.318                   |
| CRYPTO       | 8.0e-5   | 0.14 | 0.82 | 0.844                   |

Innovations are Student-t with 5 degrees of freedom, scaled to unit variance.

## Regenerating

```python
from src.authentic_smoke_test import make_dataset
X, y, _ = make_dataset("CRYPTO", n=3100, seed=42)
# X: (3069, 30, 4) feature windows ; y: (3069,) target returns
```

This produces bit-identical output to the CSVs in this directory.
