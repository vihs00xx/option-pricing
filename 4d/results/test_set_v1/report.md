# Frozen four-stock test set

Frozen at 2026-10-04T09:02:26.777865+00:00. Final assessment of four-stock pricing methods, neural and classical. No training, tuning or model selection may use these cases or their references.

## Cases

| Split | Cases | Definition |
|---|---:|---|
| interpolation | 128 | inside the training domain |
| short_maturity | 32 | remaining life below the training domain; tau = 0.005 + 0.045 * (the tau column's uniform) |
| extrapolation | 52 | one parameter outside the training domain |
| anchor | 19 | hand-chosen regimes |

Strike 100; calls and puts alternate within each split. Latin hypercube seed 20261004 plus split offset; correlations from a separate generator (seed + 100 + offset). scipy.stats.qmc.LatinHypercube(d=17, seed=seed + split offset): columns 1-14 scale to the box parameters, columns 15-17 are three uniforms whose sorted spacings (0, u(1), u(2), u(3), 1) are a uniform Dirichlet(1, 1, 1, 1) draw for the weights.

Training domain:

| Parameter | Range |
|---|---|
| moneyness_1 | 0.7 to 1.3 |
| moneyness_2 | 0.7 to 1.3 |
| moneyness_3 | 0.7 to 1.3 |
| moneyness_4 | 0.7 to 1.3 |
| tau | 0.05 to 2 |
| volatility_1 | 0.1 to 0.6 |
| volatility_2 | 0.1 to 0.6 |
| volatility_3 | 0.1 to 0.6 |
| volatility_4 | 0.1 to 0.6 |
| rate | -0.02 to 0.1 |
| dividend_1 | 0 to 0.06 |
| dividend_2 | 0 to 0.06 |
| dividend_3 | 0 to 0.06 |
| dividend_4 | 0 to 0.06 |
| weight | 0.1 to 0.7 |
| correlation | -0.5 to 0.9 |

Weights: On the simplex with every weight >= 0.1, uniform: w = 0.1 + 0.6 * D with D ~ Dirichlet(1, 1, 1, 1). Correlation: The 6 off-diagonal entries each in [-0.5, 0.9] and the smallest eigenvalue of the matrix >= 0.05; uniform on that set (box draws with rejection).

Extrapolation values (4 cases each, stock or pair rotated): moneyness 0.55, 1.45; tau 2.5, 3.0; volatility 0.05, 0.75; rate -0.04, 0.14; dividend 0.09; weight 0.03, 0.8; correlation -0.7, 0.97.

## References

Conditional estimator of src/pricing (one Gaussian factor in closed form, direction "volatility") integrated over the three residual factors by tensor Gauss-Hermite; analytic conditional Deltas and spot Hessian; central common-random-number bumps for vega, theta, rho and correlation sensitivities. Includes the Phase 0 root-solver fix.

Accept at the first ladder level whose largest absolute change over all 27 quantities against the previous level is below the tolerance and keep that level's values; the next level must confirm (change below the tolerance). A failed confirmation resumes the climb.

| Accepted nodes | Cases | interpolation | short_maturity | extrapolation | anchor | median s | max s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 188 | 102 | 32 | 37 | 17 | 7.6 | 42.9 |
| 40 | 29 | 17 | 0 | 11 | 1 | 15.3 | 107.2 |
| 48 | 7 | 4 | 0 | 3 | 0 | 120.0 | 156.6 |
| 64 | 3 | 3 | 0 | 0 | 0 | 93.3 | 188.0 |
| 80 | 4 | 2 | 0 | 1 | 1 | 217.2 | 405.1 |

Replacements (pre-declared rule: A generated case (interpolation, short maturity, extrapolation) not accepted by 128 nodes is replaced by the next spare draw of its own split stream and logged with its parameters. An anchor that is not accepted stops the build.):

None.

## Cross-checks (recorded, never used to change a reference)

| Check | Worst value | Case | Quantity |
|---|---:|---|---|
| Equal-direction relative difference | 3.34e-01 | extrapolation_volatility_017 | correlation_14 |
| Bump halving relative change | 5.73e-06 | interpolation_063 | vega_4 |
| RQMC max abs z | 1.33e+01 | short_maturity_007 | rho |

RQMC (65536 points x 16 replicates, geometric control): RMS z over all cases and quantities 1.103; share of reported 95% intervals containing the reference 0.943; 1 of 6237 quantity z-values exceed 5 in magnitude.

Relative differences divide by max(|reference|, floor), with floors 1e-7 for Gamma and 1e-4 otherwise. Equal loadings shrink the smoothing factor when volatilities differ strongly, so that cross-check is expected to be under-resolved at the accepted level in such cases (Phase 0: it converges to the volatility-direction value with more nodes).

Largest equal-direction differences:

| Case | Quantity | Relative | Accepted nodes |
|---|---|---:|---:|
| extrapolation_volatility_017 | correlation_14 | 3.3e-01 | 32 |
| interpolation_037 | correlation_34 | 1.3e-01 | 32 |
| extrapolation_volatility_018 | correlation_24 | 1.3e-02 | 32 |
| extrapolation_volatility_019 | vega_3 | 1.2e-02 | 32 |
| interpolation_117 | vega_4 | 1.0e-03 | 32 |
| extrapolation_tau_012 | gamma_33 | 9.9e-04 | 32 |
| interpolation_043 | correlation_24 | 9.2e-04 | 32 |
| interpolation_051 | vega_2 | 6.7e-04 | 32 |
| extrapolation_correlation_052 | correlation_13 | 6.2e-04 | 32 |
| interpolation_001 | correlation_24 | 3.4e-04 | 32 |

Largest RQMC |z|:

| Case | Quantity | max abs z | RMS z | inside 95% |
|---|---|---:|---:|---:|
| short_maturity_007 | rho | 13.31 | 2.62 | 0.96 |
| anchor_17_negative_rate_put | delta_3 | 4.88 | 2.05 | 0.70 |
| short_maturity_029 | correlation_13 | 4.78 | 2.41 | 0.56 |
| interpolation_122 | gamma_11 | 4.34 | 1.39 | 0.93 |
| anchor_10_high_equicorrelation_put | gamma_44 | 4.20 | 1.77 | 0.78 |
| interpolation_025 | theta | 4.10 | 1.40 | 0.85 |
| anchor_07_short_atm_call | correlation_12 | 4.04 | 1.30 | 0.93 |
| extrapolation_tau_011 | delta_3 | 4.02 | 1.74 | 0.78 |
| extrapolation_moneyness_002 | vega_3 | 3.84 | 1.06 | 0.93 |
| interpolation_126 | vega_3 | 3.76 | 1.28 | 0.93 |

Largest bump-halving changes:

| Case | Quantity | Relative |
|---|---|---:|
| interpolation_063 | vega_4 | 5.7e-06 |
| short_maturity_018 | theta | 3.2e-06 |
| interpolation_006 | vega_4 | 2.9e-06 |
| short_maturity_007 | correlation_24 | 2.4e-06 |
| short_maturity_013 | correlation_23 | 1.9e-06 |

## Separation from development contracts

For a test case and a development contract, the largest over parameters of |difference| / domain width (moneyness, tau, volatility, rate, dividend, weight and the 6 correlations; widths from the training domain). The reported separation is the minimum over all pairs.

Minimum 0.333 (anchor_13_negative_equicorrelation_call against e4_stress); by split: interpolation 0.371, short_maturity 0.442, extrapolation 0.394, anchor 0.333. No case duplicates any of the 98 development contracts.

No pricing method has been evaluated on these cases.

## Rules

- Train, tune and select models only on data generated with other seeds; never on these cases or references.
- Development and training data must use seeds other than 20261004 and its offsets and must keep a positive minimum distance from every test case under the separation metric; report that distance.
- Before evaluating any model on these cases, freeze its checkpoint hash, settings and the evaluation protocol.
- Report every method on every split; do not drop cases after seeing results.
- Once any method is evaluated here, the cases are observed; later model changes need a new test set.

`protocol.sha256` identifies `protocol.json`, which pins `cases.json` and `references.csv`; `metadata.json` pins every output and the generating sources. Verify with `python -m experiments.verify results/test_set_v1`.
