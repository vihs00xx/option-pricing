# DeepONet Greek-loss weight study

Selected weight: 1; price gate passed: True. Price RMSE limit: 0.215683. Each continuation received 300 CPU seconds.

| Weight | Price RMSE | Greek score | Passes price gate |
|---|---:|---:|---|
| 0.01 | 0.182668 | 0.0146749 | True |
| 0.1 | 0.209366 | 0.0164375 | True |
| 1 | 0.211373 | 0.0127851 | True |

## Replication

Three seeds: the selection seed and two new full-pipeline initializations. The dataset is fixed. This measures training variability, not uncertainty over datasets. The weight was fixed before new-seed runs. The final test was not accessed.

| Quantity | RMSE mean | Sample standard deviation |
|---|---:|---:|
| price | 0.257525 | 0.042927 |
| delta | 0.0180554 | 0.00178593 |
| gamma | 0.00352394 | 0.000183062 |
| vega | 1.42304 | 0.0954477 |
| theta | 1.76816 | 0.1719 |
| rho | 1.48588 | 0.0387063 |

The original price-only and weight-one checkpoints were retained and their hashes rechecked. The weight-one arm was reused from the previous matched configuration; it was not rerun. There are no paired price-only continuations for the two new seeds, so superiority over price-only training across seeds is not established.

## Price stability and checkpoint verification

All selected checkpoints reproduce their saved RMSEs after reloading. All retained checkpoint hashes match.

| Seed | Role | Selected epoch | Selected price RMSE | Passes fixed price limit | Last price RMSE |
|---|---|---:|---:|---|---:|
| 22092026 | selection_seed | 340 | 0.211373 | True | 0.246717 |
| 22092027 | fresh_seed | 370 | 0.264941 | False | 0.321567 |
| 22092028 | fresh_seed | 448 | 0.296261 | False | 0.296261 |

Both fresh seeds exceed the fixed price limit. Greek accuracy is more stable than price accuracy in this small study. Weight 0.01 remains a price-oriented candidate, but has only one seed and cannot yet be called more robust. The final test remains untouched.
