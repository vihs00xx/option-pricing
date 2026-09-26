# Fresh shared comparison of 1D option-pricing methods

Protocol frozen: 2026-09-26T11:24:00.043762+00:00. The cases, numerical budgets, checkpoint hashes, source snapshot, and selection rule were saved before predictions. No model was retrained or selected using these results. These cases are now observed and must not be reused as untouched tests for subsequent tuning.

The primary parity checkpoint is **parity_22092027**, chosen by the lowest existing validation price RMSE among the three previously refined seeds. This does not promote or replace the older registered primary. The other two parity checkpoints are evaluated as secondary results, not as a test-set selection contest.

## Fixed-parameter comparison

All five methods use 256 fresh calls at K=100, T=1, r=0.05, q=0, and sigma=0.2. The four reported outputs are price, Delta, Gamma, and calendar Theta. No Vega or Rho claim is made for the fixed PINN.

### Shared training-domain region (128 queries)

Moneyness is 0.7–1.3 and remaining maturity is 0.05–1 year. These points lie within both neural models' supported domains. RMSEs:

| Method | price | delta | gamma | theta |
|---|---:|---:|---:|---:|
| Analytical BS | 0 | 0 | 0 | 0 |
| Conditional RQMC | 2.7534e-05 | 8.1355e-07 | 6.66e-08 | 5.3038e-05 |
| CN + Richardson | 9.0897e-10 | 7.3675e-09 | 1.6758e-07 | 2.7774e-05 |
| Anchored PINN | 0.011074 | 0.0028653 | 0.0011249 | 0.094227 |
| Refined parity DeepONet | 0.042651 | 0.0060233 | 0.0012447 | 0.3178 |

### All fixed-parameter cases (256 queries)

The pooled panel additionally contains 64 moneyness-tail cases (0.5–0.7 or 1.3–1.5, outside the DeepONet moneyness training range) and 64 short-maturity cases (0.01–0.05 year). Region-level metrics are retained separately; this mixture is a designed stress assessment rather than a market-frequency average.

| Method | price | delta | gamma | theta |
|---|---:|---:|---:|---:|
| Analytical BS | 0 | 0 | 0 | 0 |
| Conditional RQMC | 2.0448e-05 | 7.4229e-07 | 1.1372e-07 | 9.3413e-05 |
| CN + Richardson | 1.4672e-09 | 1.0446e-08 | 3.1237e-07 | 5.9396e-05 |
| Anchored PINN | 0.021526 | 0.0082174 | 0.0055567 | 1.354 |
| Refined parity DeepONet | 0.17086 | 0.024619 | 0.0064562 | 2.345 |

## Cross-contract comparison

The PINN is excluded. There are 128 market states, each queried as a call and a put: 128 interpolation queries, 32 short-maturity queries, and 96 extrapolation queries. Extrapolation moves one of moneyness, volatility, rate, dividend yield, or strike; the maturity category moves both remaining time and contractual maturity together. All quantities use decimal rate/volatility inputs and time in years.

### Interpolation (128 queries)

| Method | price | delta | gamma | vega | theta | rho |
|---|---:|---:|---:|---:|---:|---:|
| Analytical BS | 0 | 0 | 0 | 0 | 0 | 0 |
| Conditional RQMC | 4.5548e-05 | 7.7089e-07 | 5.2528e-08 | 0.00014621 | 0.0001041 | 3.581e-05 |
| CN + Richardson | 1.198e-09 | 6.5867e-09 | 1.1387e-07 | 8.0491e-05 | 5.0254e-05 | 1.9377e-06 |
| Refined parity DeepONet | 0.074199 | 0.0069881 | 0.0019999 | 1.0029 | 0.53799 | 1.7836 |

### Full cross-contract set (256 queries)

| Method | price | delta | gamma | vega | theta | rho |
|---|---:|---:|---:|---:|---:|---:|
| Analytical BS | 0 | 0 | 0 | 0 | 0 | 0 |
| Conditional RQMC | 4.5463e-05 | 7.4244e-07 | 7.3139e-08 | 0.0001235 | 0.00014708 | 3.7353e-05 |
| CN + Richardson | 1.5092e-09 | 5.5075e-09 | 1.4089e-07 | 8.2137e-05 | 4.6181e-05 | 4.7986e-06 |
| Refined parity DeepONet | 0.12723 | 0.016373 | 0.004087 | 1.2515 | 2.8089 | 1.9556 |

## Financial consistency

All models use the same absolute tolerance of 1e-5 for the price bounds, option-specific Delta bounds, and nonnegative Gamma. Counts and worst violation magnitudes are reported separately. These finite sampled checks do not establish global no-arbitrage guarantees. Exact parity is a narrower property than admissible prices or Greeks.

| Panel | Model | Constraint | Violating fraction | Maximum magnitude |
|---|---|---|---:|---:|
| fixed | Anchored PINN | price_bounds | 9.77% | 0.063911 |
| fixed | Anchored PINN | delta_bounds | 9.38% | 0.024198 |
| fixed | Anchored PINN | negative_gamma | 11.33% | 0.0057125 |
| fixed | Analytical BS | price_bounds | 0.00% | 0 |
| fixed | Analytical BS | delta_bounds | 0.00% | 0 |
| fixed | Analytical BS | negative_gamma | 0.00% | 0 |
| fixed | Conditional RQMC | price_bounds | 0.00% | 0 |
| fixed | Conditional RQMC | delta_bounds | 0.00% | 0 |
| fixed | Conditional RQMC | negative_gamma | 0.00% | 0 |
| fixed | CN + Richardson | price_bounds | 0.00% | 1.8897e-25 |
| fixed | CN + Richardson | delta_bounds | 0.00% | 4.6063e-25 |
| fixed | CN + Richardson | negative_gamma | 0.00% | 1.2309e-24 |
| fixed | Refined parity DeepONet | price_bounds | 26.17% | 1.0042 |
| fixed | Refined parity DeepONet | delta_bounds | 7.03% | 0.036317 |
| fixed | Refined parity DeepONet | negative_gamma | 19.53% | 0.0097582 |
| varying | Analytical BS | price_bounds | 0.00% | 0 |
| varying | Analytical BS | delta_bounds | 0.00% | 0 |
| varying | Analytical BS | negative_gamma | 0.00% | 0 |
| varying | Conditional RQMC | price_bounds | 0.00% | 0 |
| varying | Conditional RQMC | delta_bounds | 0.00% | 0 |
| varying | Conditional RQMC | negative_gamma | 0.00% | 0 |
| varying | CN + Richardson | price_bounds | 0.00% | 0 |
| varying | CN + Richardson | delta_bounds | 0.00% | 0 |
| varying | CN + Richardson | negative_gamma | 0.00% | 0 |
| varying | Refined parity DeepONet | price_bounds | 5.47% | 0.24708 |
| varying | Refined parity DeepONet | delta_bounds | 0.78% | 0.017584 |
| varying | Refined parity DeepONet | negative_gamma | 7.81% | 0.00625 |

## Matched timing workloads

CPU float64, two configured numerical-library threads, one warm-up and five measured repetitions per batch size. Each panel/method uses a fresh subprocess. Loading is excluded; query conversion, computation, and output conversion are included. Four outputs are requested in the fixed panel, six in the varying panel. The fixed CN adapter avoids unused volatility/rate bump solves and is tested against the existing full solver. Native conditional RQMC still calculates its six outputs internally. Timings describe these implementations, not an optimally cached grid or a theoretical lower bound.

| Panel | Method | Batch | Median ms/query | Batch IQR ms | Queries/s | Peak RSS MB | Peak increment MB |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed | Anchored PINN | 1 | 2.9108 | 0.4308 | 343.55 | 273.4 | 0.1 |
| fixed | Anchored PINN | 8 | 0.31701 | 1.022 | 3154.4 | 273.9 | 0.1 |
| fixed | Anchored PINN | 32 | 0.083287 | 0.7299 | 12007 | 274.2 | 0.0 |
| fixed | Analytical BS | 1 | 0.9521 | 0.4071 | 1050.3 | 236.9 | 0.0 |
| fixed | Analytical BS | 8 | 0.16172 | 1.4117 | 6183.3 | 236.9 | 0.0 |
| fixed | Analytical BS | 32 | 0.045106 | 3.402 | 22170 | 236.9 | 0.0 |
| fixed | Conditional RQMC | 1 | 660.27 | 620.96 | 1.5145 | 240.9 | 0.2 |
| fixed | Conditional RQMC | 8 | 636.35 | 1772.2 | 1.5715 | 241.1 | 0.0 |
| fixed | Conditional RQMC | 32 | 384.82 | 7362.2 | 2.5986 | 241.1 | 0.0 |
| fixed | CN + Richardson | 1 | 339 | 18.98 | 2.9499 | 241.2 | 1.1 |
| fixed | CN + Richardson | 8 | 167.1 | 1526.2 | 5.9845 | 241.4 | 0.1 |
| fixed | CN + Richardson | 32 | 158.71 | 4091.1 | 6.3009 | 241.6 | 0.3 |
| fixed | Refined parity DeepONet | 1 | 4.4643 | 0.4682 | 224 | 241.9 | 0.0 |
| fixed | Refined parity DeepONet | 8 | 0.50719 | 0.7208 | 1971.7 | 242.3 | 0.0 |
| fixed | Refined parity DeepONet | 32 | 0.1585 | 0.3214 | 6309.3 | 242.7 | 0.0 |
| varying | Analytical BS | 1 | 0.2887 | 0.0375 | 3463.8 | 237.8 | 0.0 |
| varying | Analytical BS | 8 | 0.066537 | 0.0363 | 15029 | 237.8 | 0.0 |
| varying | Analytical BS | 32 | 0.018841 | 0.1262 | 53077 | 237.9 | 0.0 |
| varying | Conditional RQMC | 1 | 988.87 | 633.43 | 1.0113 | 241.3 | 0.1 |
| varying | Conditional RQMC | 8 | 444.98 | 219.88 | 2.2473 | 241.4 | 0.0 |
| varying | Conditional RQMC | 32 | 352.25 | 2716 | 2.8389 | 241.4 | 0.0 |
| varying | CN + Richardson | 1 | 624.47 | 66.102 | 1.6013 | 241.2 | 0.5 |
| varying | CN + Richardson | 8 | 610.34 | 748.97 | 1.6384 | 241.6 | 0.4 |
| varying | CN + Richardson | 32 | 729.53 | 4203.7 | 1.3707 | 241.7 | 0.2 |
| varying | Refined parity DeepONet | 1 | 2.0494 | 0.247 | 487.95 | 241.6 | 0.0 |
| varying | Refined parity DeepONet | 8 | 0.30538 | 0.1598 | 3274.7 | 242.0 | 0.1 |
| varying | Refined parity DeepONet | 32 | 0.17466 | 0.5943 | 5725.5 | 242.5 | 0.0 |

## Previously recorded training costs

| Model | Stage | Recorded seconds |
|---|---|---:|
| anchored_pinn | initial | 3002.77 |
| anchored_pinn | refinement | 1008.35 |
| anchored_pinn | anchored refinement | 1378.63 |
| parity_22092027 | pilot | 246.54 |
| parity_22092027 | Greek-supervised continuation | 1377.96 |
| parity_22092027 | adam_small_steps | 180.04 |
| parity_22092027 | lbfgs | 180.35 |

The parity table includes both refinement arms because validation selection required both. Historical stage timings are not matched to the current inference hardware/thread protocol. They exclude unrecorded label-generation costs and the wider architecture/weight search. All three seeds' recorded stages are retained in historical_training_costs.csv; these are not complete end-to-end project costs.

RSS is process-wide and sampled at 1 ms; it includes the common imported runtime and is not isolated model storage or an exact allocator peak. The timing cases are predefined subsets of the shared-domain/interpolation cases. They are independent-query workloads; CN surface reuse and large neural batches beyond 32 are not assessed here.

## Interpretation and limits

- Analytical Black–Scholes supplies the labels, so its reported error is zero by definition. It remains an available direct solution for this problem.
- Conditional RQMC uses analytical continuation inside the estimator. Its strong performance here should not be attributed to an entirely label-free simulation method.
- Compare errors down a quantity's column on the same cases. Different Greeks have different units; their raw RMSE sizes are not interchangeable.
- Faster neural queries alone do not establish a practical advantage: accuracy requirements, financial violations, training, label generation, and architecture search all matter. No accuracy-matched or end-to-end break-even claim is made.
- The three parity seeds share training data and architecture. Their spread is descriptive, not evidence from three independently designed studies; the smallest fresh-test error must not be used to change checkpoint selection.
- RQMC standard errors are retained with each prediction. There is one randomized estimate per case, so this experiment does not measure repeated-trial interval coverage.
- New cases were checked for rounded financial-query overlaps against available DeepONet NPZ splits, including earlier evaluations. Excluding contractual T from this overlap check makes it stricter for the time-homogeneous analytical problem. This audit does not recover missing historical training samples.

## Reproduction

Use the project-local `.venv-benchmark` Python with NumPy 2.2.6. Run `python -m experiments.comparison.shared_evaluation evaluate`, then `python -m experiments.comparison.run_pinned_timing`, then `python -m experiments.comparison.summarize_shared`. The timing wrapper checks the frozen Python, NumPy, SciPy and PyTorch versions in each worker. Existing completed prediction files are reused under the frozen protocol; neither `freeze` nor model training should be repeated for this result. `protocol.json` and its digest pin the cases, checkpoints, settings, and implementation. Predictions, timing repetitions, worst cases, regional errors, and shape metrics remain in this result folder. The original partial timing series was archived after a global NumPy change; final timing measurements were restarted consistently under the restored versions, without changing cases, models, budgets or output workloads.
