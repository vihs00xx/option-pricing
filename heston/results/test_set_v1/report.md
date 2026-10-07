# Frozen one-stock Heston test set (heston_test_v1)

Frozen at 2026-10-06T18:56:35+00:00. Frozen one-stock Heston test set: European calls and puts, price and 11 sensitivities. Built, referenced and sealed before any network trains.

| Split | Cases | Feller violated | Definition |
|---|---:|---:|---|
| interpolation | 128 | 42 | inside the training domain |
| short_maturity | 32 | 6 | remaining life below the training domain (short-maturity panel) |
| extrapolation | 72 | 29 | one input outside the training domain, the rest a domain draw |
| anchor | 20 | 14 | hand-picked regimes and literature parameter sets |

Latin hypercube seed 20261007 + split offset; calls and puts alternate. Extrapolation values (2 cases each): S 0.6, 0.65, 1.35, 1.4; tau 2.25, 2.5, 3.0, 3.5; v0 0.005, 0.0075, 0.42, 0.49; theta 0.005, 0.0075, 0.42, 0.49; kappa 0.25, 0.375, 6.0, 7.0; xi 0.05, 0.075, 1.15, 1.3; rho -0.95, -0.925, 0.4, 0.5; r -0.04, -0.03, 0.12, 0.14; q -0.02, -0.01, 0.07, 0.08.

## References

Lewis single integral with the little-trap characteristic function (src/heston/reference.py), all 12 quantities from closed-form integrands, adaptive Gauss-Legendre with a per-case convergence ladder (levels 0-3).

Acceptance: two consecutive levels agree to 1e-9 on every quantity.

Replacement rule (declared before the build): a generated case whose ladder does not reach 1e-9 agreement by level 3, or whose torch twin disagrees with the NumPy reference by more than 1e-8 on any quantity, is replaced by the next draw of its split's Latin hypercube stream (same slot, draw + 1); an anchor that fails stops the build.

Replacements: none.

Levels used: level 1: 252. Largest last change 7.1e-15; largest torch-twin difference 7.1e-15; torch nodes up to 1024.

## Cross-check (recorded only)

QE-RQMC, 2^12 Sobol points x 16 scrambles, Brownian bridge, dt = 1/32, all 12 quantities: RMS z 10524608.70, share |z| <= 1.96 0.885, max |z| 578757792.5. The z-values include time-step and bump bias, so they are a sanity check, not a coverage test.

Largest |z|:

| Case | Quantity | max abs z |
|---|---|---:|
| short_maturity_001 | gamma | 578757792.5 |
| short_maturity_022 | vanna | 452.0 |
| short_maturity_002 | vanna | 164.3 |
| anchor_11_high_v0_short_tau | d_kappa | 20.2 |
| short_maturity_017 | d_kappa | 19.0 |
| short_maturity_025 | gamma | 17.6 |
| short_maturity_029 | d_kappa | 17.1 |
| interpolation_003 | d_kappa | 15.5 |
| short_maturity_015 | d_kappa | 12.9 |
| short_maturity_008 | d_kappa | 10.9 |

## Separation from development contracts

max over the 9 inputs of |difference| / domain width, against every Phase 0 development contract; the nearest distance is recorded.

Minimum 0.098; by split interpolation 0.107, short_maturity 0.098, extrapolation 0.153, anchor 0.187.

No pricing method has been evaluated on these cases.

## Rules

- No training, tuning or selection on these cases.
- Every split is reported.
- Any later model change gets a new test set.

`protocol.sha256` identifies `protocol.json`, which pins `cases.json` and `references.csv`; `metadata.json` pins every file and the generating sources.
