# DeepONet optimizer refinement across seeds

The two-arm refinement (small-step Adam vs full-batch L-BFGS, 180 seconds per arm) was repeated on the other saved weight-0.01 seeds with the procedure from `experiments/deeponet/refine_optimizer.py`. Selection used validation data only. The audit from the first refinement is reused here as an observed benchmark for cross-seed comparability, not as an untouched test. The final benchmark was not accessed.

| Seed | Parent price RMSE | Adam price RMSE | L-BFGS price RMSE | Selected |
|---|---:|---:|---:|---|
| 22092026 | 0.182668 | 0.117586 | 0.128508 | adam_small_steps |
| 22092027 | 0.0874065 | 0.0628626 | 0.061804 | lbfgs |
| 22092028 | 0.214117 | 0.147621 | 0.15289 | adam_small_steps |

| Seed | Quantity | Parent | Selected |
|---|---|---:|---:|
| 22092026 | price | 0.182668 | 0.117586 |
| 22092026 | delta | 0.0163895 | 0.0133167 |
| 22092026 | gamma | 0.00366284 | 0.00324752 |
| 22092026 | vega | 1.38483 | 1.28619 |
| 22092026 | theta | 1.94521 | 1.70686 |
| 22092026 | rho | 1.70121 | 1.46684 |
| 22092027 | price | 0.0874065 | 0.061804 |
| 22092027 | delta | 0.00858369 | 0.00727515 |
| 22092027 | gamma | 0.00222476 | 0.0021386 |
| 22092027 | vega | 0.864347 | 0.857339 |
| 22092027 | theta | 0.997932 | 0.984474 |
| 22092027 | rho | 0.947437 | 0.889788 |
| 22092028 | price | 0.214117 | 0.147621 |
| 22092028 | delta | 0.0192682 | 0.0173131 |
| 22092028 | gamma | 0.00445402 | 0.00422942 |
| 22092028 | vega | 1.34384 | 1.25459 |
| 22092028 | theta | 2.44348 | 2.27476 |
| 22092028 | rho | 1.65049 | 1.50373 |

Validation shape violations (proportion above 1e-8 / maximum):

| Seed | Constraint | Parent | Selected |
|---|---|---|---|
| 22092026 | price_bounds | 0.1353 / 0.5956 | 0.0879 / 0.5400 |
| 22092026 | delta_bounds | 0.0601 / 0.1028 | 0.0718 / 0.1163 |
| 22092026 | negative_gamma | 0.0342 / 0.0128 | 0.0396 / 0.0124 |
| 22092027 | price_bounds | 0.0586 / 0.4542 | 0.0684 / 0.4139 |
| 22092027 | delta_bounds | 0.0708 / 0.0532 | 0.0625 / 0.0536 |
| 22092027 | negative_gamma | 0.0464 / 0.0216 | 0.0479 / 0.0220 |
| 22092028 | price_bounds | 0.0879 / 0.4741 | 0.1001 / 0.5325 |
| 22092028 | delta_bounds | 0.0796 / 0.0593 | 0.0767 / 0.0607 |
| 22092028 | negative_gamma | 0.0562 / 0.0100 | 0.0542 / 0.0102 |

Observed audit price RMSE (observed benchmark, evaluated after validation selection):

| Seed | Model | In training range | Broader stress range |
|---|---|---:|---:|
| 22092026 | baseline | 0.186997 | 0.351646 |
| 22092026 | selected | 0.122372 | 0.310783 |
| 22092027 | baseline | 0.0968328 | 0.188256 |
| 22092027 | selected | 0.0629381 | 0.171803 |
| 22092028 | baseline | 0.193049 | 0.412321 |
| 22092028 | selected | 0.144659 | 0.420856 |

## Verdict

- Refinement improves price accuracy on every seed: **True**.
- L-BFGS is a dependable improvement: **False**.
- Relative price RMSE change of the selected arm per seed: -35.6%, -29.3%, -31.1%.

Verdict rule: refinement_improves_price: Every seed selects a refined arm with lower price RMSE than its parent.; lbfgs_dependable: L-BFGS is the selected arm on at least two of three seeds and lowers price RMSE versus the parent on all three seeds.
