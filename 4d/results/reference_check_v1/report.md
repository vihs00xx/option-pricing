# Phase 0: four-stock reference check

Development contracts only (seeds 404000-404999). Reference = tensor Gauss-Hermite over the three residual factors of the conditional estimator, direction "volatility", 27 quantities (price, 4 Deltas, 10 Gammas, 4 Vegas, theta, rho, 6 correlation sensitivities).

## Findings

- **Root fix.** Both new tests fail on the unpatched solver with "Conditional strike root did not converge" and pass on the patched one. On 22 cases where the old solver worked, the new one returns bitwise-identical values (max difference 0.0).
- **Node schedule.** No contract can be accepted at 24, because 24 is the first level and has no previous level to compare with. 68 of the 75 random contracts (91%) are accepted at 32. The cloud estimate was about 80%. The rest of the random draws are accepted at 40 (4), 48 (2) and 80 (1, random02: smallest eigenvalue 0.060, q = 11.0). No confirmation failed. The cloud sanity values reproduce exactly: C* at tau 2 changes by 3.94e-4, 7.93e-6, 1.38e-7 and 6.59e-9 between 16, 24, 32, 40 and 48 nodes, and is accepted at 48. N1 at tau 1 is accepted at 96. E4 never converges: its change at 128 is still 6.6e-3.
- **Quantity that sets the level.** The last change is almost always largest in theta or rho, or in vega or correlation for a few contracts. These are bumped quantities, so their roundoff is quadrature error divided by a step of 1e-5 or 1e-4. Price, Deltas and Gammas converge earlier. At high node counts the bumped quantities reach a floor of about 1e-11 to 5e-11.
- **Equal-loadings cross-check.** At the accepted level it agrees to 1.1e-9 for random02 and 1.7e-10 for n1_tau1. For cstar_tau3 at 64 nodes it differs by 1.6e-3 (in rho). This comes from the equal direction converging slowly, not from an error in the reference: the equal direction approaches the volatility value as 1.6e-3, 1.4e-4 and 5.6e-6 at 64, 96 and 128 nodes. This supports the default direction="volatility".
- **Cost.** Greeks cost 30 to 40 times as much as the price alone, from 2 x (number of bumped parameters) extra evaluations. Four stocks with Greeks take 1.2 s at 32 nodes and 4.7 s at 48. Five stocks with Greeks take 79 s at 32 and 412 s at 48 (3.9 GB peak), and the hard five-stock contract needs 48 nodes for 1e-8. Six stocks are price-only in practice: Greeks at 16 nodes already take 122 s, and price at 24 nodes takes 20 s and 3.9 GB. The moderate family converges to the bump floor by 16 to 24 nodes in every dimension.
- **RQMC.** RQMC is unbiased within its error bars. Pooled 95% coverage is 94.4% at both 4096 and 65536 points (RMS z 1.10 and 1.09). The z values within a run are correlated across the 27 quantities, so the effective sample is closer to the 260 and 60 runs than to the number of z values. Coverage is lowest for price (92.3% and 91.7%) and theta. This confirms mild under-coverage in direction, but it is smaller than the 93% overall the cloud expected.
- **Heavy tails.** Replicate means of C* at tau 2 are strongly right-skewed and heavy-tailed: excess kurtosis 14.1 with the control (95% CI 9.7 to 17.5, skew 2.9, price coverage 91/100) and 8.5 without it (CI 5.3 to 11.1, coverage 43/50). An easy contract (random29) shows 0.16 (CI -0.18 to 0.60) and 96% coverage. Each controlled run fits its own coefficient (about 22% spread for C*), so the no-control figure is the clean one. Both point the same way: on hard contracts, Student-t intervals from 16 replicates under-cover.
- **Quadrature against RQMC at d = 4.** A 65536-point RQMC run takes about 55 s and leaves price errors of 1e-5 to 1e-3. Gauss-Hermite reaches 1e-8 on all 27 quantities within 13 s for 94 of the 97 accepted contracts, counting every level climbed. The exceptions are cstar_tau3 (29 s), random02 (54 s) and n1_tau1 (90 s). Quadrature is the reference for d = 4. RQMC is still needed from d = 6 upwards.

## 1. Root-solver fix

`_root` in `src/pricing/conditional.py` now detects the two-point cycle that Newton falls into when the step is rounding noise of size ulp(log K)/f', and accepts it once every step is below an 8-eps noise floor. Anything the old tolerance test accepted is returned unchanged.

| check | unpatched | patched |
|---|---|---|
| `test_short_maturity_converges_to_rounding_level` | RuntimeError: Conditional strike root did not converge | pass |
| `test_near_singular_correlation_long_maturity` | RuntimeError: Conditional strike root did not converge | pass |
| full 4d test suite | - | 20 passed |

C* at tau 0.001: 16 vs 32 nodes differ by at most 3.75e-16 (relative 1.1e-14) on price, Deltas and Gammas; the root residual on the 32-node grid is 1.12e-15 = 1.26 ulp(log K) (np.longdouble is float64 on Windows, so the residual was recomputed in mpmath (40 digits) instead).

Bitwise identity: 22 cases (C* at tau {0.02,0.05,1,2} x nodes {16,32,48} with greeks, plus 10 random 4-stock domain contracts (seed 404001) at 32 nodes with greeks); the old code failed on 0, maximum absolute difference 0.0.

## 2. Node schedule

Accepted at the first level whose largest change over all 27 quantities is below 1e-08; one more level must confirm. Seconds are single-threaded and include all levels up to acceptance.

| nodes | contracts | random_share | median_seconds_to_accept | max_seconds_to_accept | named |
|---|---|---|---|---|---|
| 32 | 84 | 91% | 2.5 | 3.3 | cstar_tau0.005, cstar_tau0.02, cstar_tau0.05, vol_dispersion, n1_tau0.02, two_sector, positive_min_eig_0.05, one_weight_0.7, deep_itm_call, deep_otm_call, deep_itm_put, deep_otm_put, random00_tau0.005, random00_tau0.01, random01_tau0.005, random01_tau0.01 |
| 40 | 6 | 5% | 5.7 | 6.3 | random02_tau0.005, random02_tau0.01 |
| 48 | 4 | 3% | 11.7 | 13.0 | cstar_tau1, cstar_tau2 |
| 64 | 1 | 0% | 28.8 | 28.8 | cstar_tau3 |
| 80 | 1 | 1% | 53.9 | 53.9 |  |
| 96 | 1 | 0% | 89.5 | 89.5 | n1_tau1 |
| not by 128 | 1 | 0% |  |  | e4_stress |

Named corners:

| contract | tau | q | min_eigenvalue | max_variance | accepted_nodes | confirm_change | last_change | largest_last_change_quantity | flag |
|---|---|---|---|---|---|---|---|---|---|
| cstar_tau0.005 | 0.005 | 5.96 | 0.349 | 0.0018 | 32 | 1.9e-11 | 1.9e-11 | theta |  |
| cstar_tau0.02 | 0.02 | 5.96 | 0.349 | 0.0072 | 32 | 8.1e-11 | 8.1e-11 | theta |  |
| cstar_tau0.05 | 0.05 | 5.96 | 0.349 | 0.018 | 32 | 1.3e-11 | 1.3e-11 | theta |  |
| cstar_tau1 | 1 | 5.96 | 0.349 | 0.36 | 48 | 5.3e-11 | 5.3e-11 | theta |  |
| cstar_tau2 | 2 | 5.96 | 0.349 | 0.72 | 48 | 3.4e-10 | 3.4e-10 | rho |  |
| cstar_tau3 | 3 | 5.96 | 0.349 | 1.08 | 64 | 8.5e-12 | 8.5e-12 | rho |  |
| vol_dispersion | 1 | 2.11 | 0.7 | 0.562 | 32 | 3.2e-11 | 3.2e-11 | theta |  |
| n1_tau0.02 | 0.02 | 40 | 0.1 | 0.00405 | 32 | 4.7e-11 | 4.7e-11 | theta |  |
| n1_tau1 | 1 | 40 | 0.1 | 0.203 | 96 | 5.9e-11 | 5.9e-11 | rho |  |
| two_sector | 1 | 2 | 0.2 | 0.203 | 32 | 3.4e-11 | 3.4e-11 | theta |  |
| positive_min_eig_0.05 | 1 | 1.04 | 0.05 | 0.203 | 32 | 2.1e-11 | 2.1e-11 | theta |  |
| one_weight_0.7 | 1 | 2.11 | 0.7 | 0.203 | 32 | 4.9e-11 | 4.9e-11 | theta |  |
| deep_itm_call | 0.5 | 2.11 | 0.7 | 0.101 | 32 | 1.2e-10 | 1.2e-10 | theta |  |
| deep_otm_call | 0.5 | 2.11 | 0.7 | 0.101 | 32 | 7.3e-13 | 7.3e-13 | theta |  |
| deep_itm_put | 0.5 | 2.11 | 0.7 | 0.101 | 32 | 1.3e-11 | 1.3e-11 | correlation_12 |  |
| deep_otm_put | 0.5 | 2.11 | 0.7 | 0.101 | 32 | 7.1e-12 | 7.1e-12 | theta |  |
| random00_tau0.005 | 0.005 | 4.55 | 0.537 | 0.00119 | 32 | 1.5e-11 | 1.5e-11 | theta |  |
| random00_tau0.01 | 0.01 | 4.55 | 0.537 | 0.00237 | 32 | 2.6e-11 | 2.6e-11 | theta |  |
| random01_tau0.005 | 0.005 | 2.43 | 0.165 | 0.00101 | 32 | 2.1e-10 | 2.1e-10 | theta |  |
| random01_tau0.01 | 0.01 | 2.43 | 0.165 | 0.00202 | 32 | 3.5e-11 | 3.5e-11 | theta |  |
| random02_tau0.005 | 0.005 | 11 | 0.0603 | 0.00149 | 40 | 2.8e-11 | 2.8e-11 | correlation_34 |  |
| random02_tau0.01 | 0.01 | 11 | 0.0603 | 0.00298 | 40 | 8.6e-10 | 8.6e-10 | theta |  |
| e4_stress | 1 | 400 | 0.01 | 0.09 |  |  | 6.6e-03 | correlation_14 | not accepted by 128 |

Equal-loadings cross-check at the accepted level (contracts accepted at 64 or more):

| contract | accepted_nodes | equal_direction_max_diff |
|---|---|---|
| random02 | 80 | 1.12e-09 |
| cstar_tau3 | 64 | 1.65e-03 |
| n1_tau1 | 96 | 1.70e-10 |

## 3. Cost by number of stocks

Tensor grid of n^(d-1) points, one process, BLAS on one thread; peak = peak working set.

| family | d | greeks | nodes | grid_points | quantities | seconds | peak_gb | max_change_vs_previous | skipped |
|---|---|---|---|---|---|---|---|---|---|
| hard | 3 | False | 16 | 256 | 1 | 0.00 | 0.09 |  |  |
| hard | 3 | True | 16 | 256 | 18 | 0.03 | 0.09 |  |  |
| hard | 3 | False | 24 | 576 | 1 | 0.00 | 0.10 | 2.0e-05 |  |
| hard | 3 | True | 24 | 576 | 18 | 0.03 | 0.10 | 5.2e-04 |  |
| hard | 3 | False | 32 | 1024 | 1 | 0.00 | 0.10 | 1.9e-07 |  |
| hard | 3 | True | 32 | 1024 | 18 | 0.03 | 0.09 | 9.0e-06 |  |
| hard | 3 | False | 40 | 1600 | 1 | 0.00 | 0.09 | 1.1e-10 |  |
| hard | 3 | True | 40 | 1600 | 18 | 0.05 | 0.09 | 2.1e-07 |  |
| hard | 3 | False | 48 | 2304 | 1 | 0.01 | 0.09 | 1.1e-10 |  |
| hard | 3 | True | 48 | 2304 | 18 | 0.06 | 0.09 | 4.4e-09 |  |
| hard | 4 | False | 16 | 4096 | 1 | 0.01 | 0.10 |  |  |
| hard | 4 | True | 16 | 4096 | 27 | 0.17 | 0.10 |  |  |
| hard | 4 | False | 24 | 13824 | 1 | 0.02 | 0.10 | 1.8e-05 |  |
| hard | 4 | True | 24 | 13824 | 27 | 0.58 | 0.10 | 3.9e-04 |  |
| hard | 4 | False | 32 | 32768 | 1 | 0.05 | 0.10 | 3.2e-07 |  |
| hard | 4 | True | 32 | 32768 | 27 | 1.21 | 0.11 | 7.9e-06 |  |
| hard | 4 | False | 40 | 64000 | 1 | 0.12 | 0.12 | 9.4e-09 |  |
| hard | 4 | True | 40 | 64000 | 27 | 2.45 | 0.13 | 1.4e-07 |  |
| hard | 4 | False | 48 | 110592 | 1 | 0.19 | 0.13 | 2.7e-10 |  |
| hard | 4 | True | 48 | 110592 | 27 | 4.69 | 0.16 | 6.6e-09 |  |
| hard | 5 | False | 16 | 65536 | 1 | 0.13 | 0.12 |  |  |
| hard | 5 | True | 16 | 65536 | 38 | 4.13 | 0.14 |  |  |
| hard | 5 | False | 24 | 331776 | 1 | 0.73 | 0.23 | 7.4e-06 |  |
| hard | 5 | True | 24 | 331776 | 38 | 23.41 | 0.33 | 4.7e-04 |  |
| hard | 5 | False | 32 | 1048576 | 1 | 2.37 | 0.51 | 2.2e-07 |  |
| hard | 5 | True | 32 | 1048576 | 38 | 79.37 | 0.84 | 9.2e-06 |  |
| hard | 5 | False | 40 | 2560000 | 1 | 5.95 | 1.12 | 4.6e-09 |  |
| hard | 5 | True | 40 | 2560000 | 38 | 202.66 | 1.92 | 2.6e-07 |  |
| hard | 5 | False | 48 | 5308416 | 1 | 13.00 | 2.22 | 4.3e-11 |  |
| hard | 5 | True | 48 | 5308416 | 38 | 411.90 | 3.87 | 8.4e-09 |  |
| hard | 6 | False | 16 | 1048576 | 1 | 2.62 | 0.59 |  |  |
| hard | 6 | True | 16 | 1048576 | 51 | 122.31 | 1.06 |  |  |
| hard | 6 | False | 24 | 7962624 | 1 | 20.52 | 3.85 | 2.5e-05 |  |
| moderate | 3 | False | 16 | 256 | 1 | 0.00 | 0.10 |  |  |
| moderate | 3 | True | 16 | 256 | 18 | 0.01 | 0.09 |  |  |
| moderate | 3 | False | 24 | 576 | 1 | 0.00 | 0.09 | 1.4e-14 |  |
| moderate | 3 | True | 24 | 576 | 18 | 0.03 | 0.10 | 6.2e-11 |  |
| moderate | 3 | False | 32 | 1024 | 1 | 0.00 | 0.09 | 1.8e-15 |  |
| moderate | 3 | True | 32 | 1024 | 18 | 0.03 | 0.09 | 1.6e-10 |  |
| moderate | 3 | False | 40 | 1600 | 1 | 0.00 | 0.10 | 1.8e-15 |  |
| moderate | 3 | True | 40 | 1600 | 18 | 0.03 | 0.09 | 7.8e-11 |  |
| moderate | 3 | False | 48 | 2304 | 1 | 0.01 | 0.10 | 3.6e-15 |  |
| moderate | 3 | True | 48 | 2304 | 18 | 0.05 | 0.10 | 1.2e-10 |  |
| moderate | 4 | False | 16 | 4096 | 1 | 0.01 | 0.10 |  |  |
| moderate | 4 | True | 16 | 4096 | 27 | 0.12 | 0.10 |  |  |
| moderate | 4 | False | 24 | 13824 | 1 | 0.02 | 0.10 | 3.6e-15 |  |
| moderate | 4 | True | 24 | 13824 | 27 | 0.46 | 0.10 | 1.2e-11 |  |
| moderate | 4 | False | 32 | 32768 | 1 | 0.05 | 0.10 | 5.3e-15 |  |
| moderate | 4 | True | 32 | 32768 | 27 | 1.16 | 0.11 | 3.4e-11 |  |
| moderate | 4 | False | 40 | 64000 | 1 | 0.10 | 0.12 | 5.3e-15 |  |
| moderate | 4 | True | 40 | 64000 | 27 | 2.25 | 0.13 | 3.6e-11 |  |
| moderate | 4 | False | 48 | 110592 | 1 | 0.19 | 0.13 | 0.0e+00 |  |
| moderate | 4 | True | 48 | 110592 | 27 | 4.13 | 0.16 | 4.0e-11 |  |
| moderate | 5 | False | 16 | 65536 | 1 | 0.12 | 0.12 |  |  |
| moderate | 5 | True | 16 | 65536 | 38 | 3.46 | 0.14 |  |  |
| moderate | 5 | False | 24 | 331776 | 1 | 0.64 | 0.23 | 7.8e-14 |  |
| moderate | 5 | True | 24 | 331776 | 38 | 21.60 | 0.33 | 5.9e-11 |  |
| moderate | 5 | False | 32 | 1048576 | 1 | 1.96 | 0.51 | 5.0e-14 |  |
| moderate | 5 | True | 32 | 1048576 | 38 | 72.98 | 0.84 | 1.6e-11 |  |
| moderate | 5 | False | 40 | 2560000 | 1 | 5.29 | 1.12 | 1.9e-13 |  |
| moderate | 5 | True | 40 | 2560000 | 38 | 178.67 | 1.92 | 5.7e-12 |  |
| moderate | 5 | False | 48 | 5308416 | 1 | 10.91 | 2.22 | 1.2e-13 |  |
| moderate | 5 | True | 48 | 5308416 | 38 | 372.19 | 3.87 | 1.4e-11 |  |
| moderate | 6 | False | 16 | 1048576 | 1 | 2.27 | 0.59 |  |  |
| moderate | 6 | True | 16 | 1048576 | 51 | 105.47 | 1.06 |  |  |
| moderate | 6 | False | 24 | 7962624 | 1 | 18.15 | 3.85 | 1.4e-12 |  |

## 4. RQMC against the reference

rqmc_estimate: 16 replicates, direction volatility, greeks, geometric control. z = (estimate - reference) / reported standard error, pooled over contracts, seeds and quantities.

| points | runs | price_rmse | rms_z | coverage_95 | max_abs_z | mean_seconds |
|---|---|---|---|---|---|---|
| 4096 | 260 | 1.51e-03 | 1.096 | 0.944 | 5.53 | 3.67 |
| 65536 | 60 | 2.78e-04 | 1.092 | 0.944 | 4.10 | 55.19 |

| points | group | n | rms_z | coverage_95 | max_abs_z |
|---|---|---|---|---|---|
| 4096 | price | 260 | 1.182 | 0.923 | 4.62 |
| 4096 | delta | 1040 | 1.131 | 0.932 | 4.74 |
| 4096 | gamma | 2600 | 1.062 | 0.953 | 4.81 |
| 4096 | vega | 1040 | 1.137 | 0.939 | 5.53 |
| 4096 | theta | 260 | 1.140 | 0.927 | 4.18 |
| 4096 | rho | 260 | 1.081 | 0.938 | 3.30 |
| 4096 | correlation | 1560 | 1.078 | 0.947 | 4.06 |
| 65536 | price | 60 | 1.219 | 0.917 | 3.78 |
| 65536 | delta | 240 | 1.117 | 0.942 | 3.99 |
| 65536 | gamma | 600 | 1.093 | 0.948 | 3.78 |
| 65536 | vega | 240 | 1.156 | 0.925 | 3.86 |
| 65536 | theta | 60 | 1.373 | 0.867 | 4.10 |
| 65536 | rho | 60 | 1.076 | 0.950 | 3.18 |
| 65536 | correlation | 360 | 0.949 | 0.967 | 2.75 |

Replicate-mean tails (price-only runs x 16 replicates at 4096 points: 100 runs with the geometric control, 50 without):

| contract | control | replicate_means | excess_kurtosis | excess_kurtosis_ci95 | excess_kurtosis_standardized | skewness | price_coverage_95 | mean_minus_reference_over_se |
|---|---|---|---|---|---|---|---|---|
| cstar_tau2 | True | 1600 | 14.11 | 9.68 to 17.53 | 1.51 | 2.89 | 0.91 | 0.11 |
| random29 | True | 1600 | 0.16 | -0.18 to 0.60 | -0.28 | -0.15 | 0.96 | 0.51 |
| cstar_tau2 | False | 800 | 8.52 | 5.30 to 11.10 | 1.07 | 2.32 | 0.86 | -2.30 |
