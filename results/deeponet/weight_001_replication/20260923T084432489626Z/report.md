# Paired DeepONet Greek weight replication

Selected model family: **weight_0.01** by the prespecified validation rule. This is a three-seed exploratory result; one seed participated in weight selection. The final test set was not accessed.

| Quantity | Weight 0.01 mean ± SD | Weight 1 mean ± SD |
|---|---:|---:|
| price | 0.161397 ± 0.065979 | 0.257525 ± 0.042927 |
| delta | 0.0147471 ± 0.00552837 | 0.0180554 ± 0.00178593 |
| gamma | 0.0034472 ± 0.00113016 | 0.00352394 ± 0.000183062 |
| vega | 1.19767 ± 0.289395 | 1.42304 ± 0.0954477 |
| theta | 1.79554 ± 0.734303 | 1.76816 ± 0.1719 |
| rho | 1.43305 ± 0.421314 | 1.48588 ± 0.0387063 |

| Seed | Weight 0.01 price RMSE | Weight 1 price RMSE |
|---|---:|---:|
| 22092026 | 0.182668 | 0.211373 |
| 22092027 | 0.0874065 | 0.264941 |
| 22092028 | 0.214117 | 0.296261 |

Both weights used the same pilot checkpoint, seed, learning rate, batch size and 300-second continuation budget within each pair. The original seed reused its previously completed runs; the two fresh seeds were newly continued at weight 0.01. Training data and analytical Greek scales are identical. The selected model family is a validation decision, not a final test result.
