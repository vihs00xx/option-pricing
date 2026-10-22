# Frozen 2-stock Heston basket test set

Frozen at 2026-10-19T08:45:37.386381+00:00. Final assessment of two-stock Heston basket pricing methods, neural and classical. No training, tuning or model selection may use these cases or their references.

Each stock has its own Heston variance (v0, kappa, theta, xi) and leverage rho_i = corr(dW_i, dZ_i). The variance drivers Z have correlation C (vol_correlation); W_i = rho_i Z_i + sqrt(1 - rho_i^2) Y_i with Y independent of Z and correlation P, so the stock correlation is R = (rho rho') o C + (rbar rbar') o P (src/heston_basket/model.py). The Feller condition is not imposed.

## Cases

| Split | Cases | Definition |
|---|---:|---|
| interpolation | 128 | inside the training domain |
| short_maturity | 32 | remaining life below the training domain; tau = 0.005 + 0.045 * (the tau column's uniform) |
| extrapolation | 46 | one parameter outside the training domain |
| anchor | 14 | hand-chosen regimes |

Strike 100; calls and puts alternate within each split. scipy.stats.qmc.LatinHypercube(d=17, seed=seed + split offset): columns 1-16 scale to the box parameters, column 17 is a uniform whose spacings (0, u, 1) are a uniform Dirichlet(1, 1) draw for the weights.

Training domain:

| Parameter | Range |
|---|---|
| moneyness_1 | 0.7 to 1.3 |
| moneyness_2 | 0.7 to 1.3 |
| tau | 0.05 to 2 |
| rate | -0.02 to 0.1 |
| dividend_1 | 0 to 0.06 |
| dividend_2 | 0 to 0.06 |
| v0_1 | 0.01 to 0.36 |
| v0_2 | 0.01 to 0.36 |
| kappa_1 | 0.5 to 5 |
| kappa_2 | 0.5 to 5 |
| theta_1 | 0.01 to 0.36 |
| theta_2 | 0.01 to 0.36 |
| xi_1 | 0.1 to 1 |
| xi_2 | 0.1 to 1 |
| leverage_1 | -0.9 to 0.3 |
| leverage_2 | -0.9 to 0.3 |
| weight | 0.1 to 0.9 |
| vol_correlation | -0.2 to 0.9 |
| residual_correlation | -0.5 to 0.9 |
| correlation | -0.7 to 0.95 |

Weights: On the simplex with every weight >= 0.1, uniform: w = 0.1 + 0.8 * D with D ~ Dirichlet(1, 1). Correlation: The variance correlation C and the residual correlation P are drawn independently, each with entries uniform on its range and the smallest eigenvalue >= 0.05 (rejection); the stock correlation R follows from the leverages, so R is always valid. The R range above is the attainable range, used to scale network inputs.

Extrapolation values: moneyness 0.55, 1.45; tau 2.5, 3.0; v0 0.005, 0.49; kappa 0.25, 7.0; theta 0.005, 0.49; xi 0.05, 1.3; leverage -0.95, 0.45; rate -0.04, 0.14; dividend 0.09; weight 0.03, 0.97; vol_correlation -0.5, 0.97; residual_correlation -0.8, 0.97.

## References

Conditional QE simulation (src/heston_basket/engine.py): Andersen's QE scheme with martingale correction for each stock's variance, the variance normals correlated by C; given the variance paths the log returns are Gaussian and the basket option is the conditional estimator of src/pricing (one factor in closed form, the residual factor sampled). Replicated scrambled-Sobol RQMC with a Brownian bridge; analytic conditional Deltas and Gammas; central common-random-number bumps for theta, vega and the parameter and correlation sensitivities; rho = -tau V + tau sum_i S_i dV/dS_i.

Climb the points per replicate until the price standard error is at most the target (currency units, strike 100); keep the first level that meets it. A case still above the target at the last level keeps that level and is flagged; no case is replaced.

| Points per replicate | Cases | interpolation | short_maturity | extrapolation | anchor | median s | max s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 106 | 58 | 32 | 13 | 3 | 10.7 | 71.4 |
| 16384 | 61 | 36 | 0 | 21 | 4 | 207.7 | 461.5 |

Above the price standard-error target of 0.0005 at 16384 points: interpolation_003, interpolation_005, interpolation_009, interpolation_014, interpolation_015, interpolation_021, interpolation_023, interpolation_029, interpolation_032, interpolation_044, interpolation_047, interpolation_048, interpolation_049, interpolation_051, interpolation_060, interpolation_068, interpolation_070, interpolation_075, interpolation_079, interpolation_080, interpolation_083, interpolation_085, interpolation_089, interpolation_093, interpolation_095, interpolation_098, interpolation_099, interpolation_109, interpolation_112, interpolation_116, interpolation_119, interpolation_121, interpolation_123, interpolation_125, extrapolation_moneyness_001, extrapolation_tau_005, extrapolation_tau_006, extrapolation_tau_007, extrapolation_v0_009, extrapolation_theta_017, extrapolation_theta_018, extrapolation_xi_023, extrapolation_rate_029, extrapolation_rate_030, extrapolation_vol_correlation_039, extrapolation_residual_correlation_043, anchor_03_itm_call, anchor_06_feller_violation_call, anchor_07_strong_leverage_put, anchor_10_dispersed_variances_call, anchor_12_negative_residual_correlation_call, anchor_13_dominant_weight_call, anchor_14_long_slow_reversion_put.

Largest relative standard errors (standard error / max(|reference|, floor)):

| Case | Quantity | Relative SE | Price SE |
|---|---|---:|---:|
| interpolation_058 | d_leverage_2 | 2.6e+00 | 1.0e-04 |
| anchor_12_negative_residual_correlation_call | d_kappa_1 | 2.6e+00 | 6.4e-04 |
| extrapolation_weight_038 | vol_correlation_12 | 2.5e+00 | 1.1e-04 |
| extrapolation_theta_018 | vol_correlation_12 | 1.8e+00 | 3.5e-03 |
| interpolation_088 | d_leverage_1 | 1.3e+00 | 3.9e-04 |
| interpolation_031 | d_xi_2 | 1.2e+00 | 5.0e-04 |
| short_maturity_007 | vol_correlation_12 | 1.0e+00 | 2.5e-05 |
| anchor_07_strong_leverage_put | d_leverage_1 | 9.7e-01 | 1.7e-03 |
| short_maturity_023 | d_leverage_2 | 6.5e-01 | 3.7e-05 |
| short_maturity_031 | vol_correlation_12 | 6.2e-01 | 2.0e-04 |

## Cross-checks (recorded, never used to change a reference)

Independent run (own seed, check points): RMS z over all cases and quantities 1.044; 1 of 4400 z-values exceed 5 in magnitude.

| Check | Worst value | Case | Quantity |
|---|---:|---|---|
| Independent run max abs z | 5.04e+00 | interpolation_008 | delta_2 |
| Doubled time step, max abs z | 5.09e+02 | short_maturity_004 | d_kappa_2 |
| Bump halving relative change | 1.74e+01 | anchor_12_negative_residual_correlation_call | d_kappa_1 |

Doubled time step, price difference: median -2.17e-05, largest magnitude 1.03e-02 (currency, strike 100).

z = difference / sqrt(se_1^2 + se_2^2). Relative differences divide by max(|reference|, floor), with floors 1e-7 for Gamma and 1e-4 otherwise.

## Separation from development contracts

For a test case and a development contract, the largest over parameters of |difference| / domain width (moneyness, tau, rate, dividends, the Heston parameters, weights and the entries of C and R; widths from the training domain). The reported separation is the minimum over all pairs.

Minimum 0.333 (extrapolation_xi_021 against random08); by split: interpolation 0.403, short_maturity 0.430, extrapolation 0.333, anchor 0.433. No case duplicates any of the 51 development contracts.

No pricing method has been evaluated on these cases.

## Rules

- Train, tune and select models only on data generated with other seeds; never on these cases or references.
- Development and training data must use seeds other than 20261018 and its offsets and must keep a positive minimum distance from every test case under the separation metric; report that distance.
- Before evaluating any model on these cases, freeze its checkpoint hash, settings and the evaluation protocol.
- Report every method on every split; do not drop cases after seeing results.
- Every reference carries a standard error and a time-step bias check; report method errors next to them, and do not read differences below about three reference standard errors as differences.
- Once any method is evaluated here, the cases are observed; later model changes need a new test set.

`protocol.sha256` identifies `protocol.json`, which pins `cases.json`, `references.csv` and `reference_standard_errors.csv`; `metadata.json` pins every output and the generating sources. Verify with `python -m experiments.verify results/test_set_v1`.
