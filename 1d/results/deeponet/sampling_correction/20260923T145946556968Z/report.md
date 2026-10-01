# Corrected sampling comparison (uniform vs mixed)

Equal-sized datasets, identical contracts; only the coordinate distribution differs. Both arms use one fixed set of Greek normalization scales computed from the uniform arm's training labels and frozen in the protocol before training, so sampling is the only changed factor. Budgets are matched update counts (500 price-only pilot epochs, 800 supervised continuation epochs, batch 512); equal update counts do not imply equal computation or wall time. Checkpoint selection used validation prices only. The evaluation package from the previous study is reused here for direct comparison and is explicitly an observed benchmark, not an untouched test. The frozen final benchmark was not accessed.

## Shared normalization (the correction)

| Target | Scale | Effective Greek-loss coefficient |
|---|---:|---:|
| delta | 0.5789 | 0.0298396 |
| gamma | 0.0193611 | 26.6771 |
| vega | 22.6929 | 1.94188e-05 |
| theta | 12.1187 | 6.80909e-05 |
| rho | 36.3526 | 7.56708e-06 |

Source dataset: `results\deeponet\training_ablation\20260923T111505263380Z\data\train_uniform.npz` (file SHA-256 9ab71c8cee9250f9...). Method: per-column root mean square of the training label rows (columns 1..5 of the six target columns: price, delta, gamma, vega, theta, rho).

The previous sampling results changed sampling and normalization jointly; this study isolates sampling with shared normalization scales.

## Budget accounting (matched update counts)

| Run | Optimizer updates | Expected | Runtime (s) | Stop reason | Premature |
|---|---:|---:|---:|---|---|
| mixed_22092026 | 12800 | 12800 | 1284.3 | epochs_completed | False |
| mixed_22092027 | 12800 | 12800 | 1287.7 | epochs_completed | False |
| mixed_22092028 | 12800 | 12800 | 1289.4 | epochs_completed | False |
| uniform_22092026 | 12800 | 12800 | 845.0 | epochs_completed | False |
| uniform_22092027 | 12800 | 12800 | 846.9 | epochs_completed | False |
| uniform_22092028 | 12800 | 12800 | 848.2 | epochs_completed | False |

## Validation RMSE per seed (mean ± sample SD across seeds)

| Quantity | Uniform per seed | Mixed per seed | Uniform mean ± SD | Mixed mean ± SD |
|---|---|---|---:|---:|
| price | 0.142688, 0.112543, 0.162382 | 0.170198, 0.172442, 0.224957 | 0.139204 ± 0.0251 | 0.189199 ± 0.03099 |
| delta | 0.0131659, 0.0108802, 0.0138399 | 0.0136656, 0.0155224, 0.0171876 | 0.0126287 ± 0.001551 | 0.0154585 ± 0.001762 |
| gamma | 0.00318405, 0.00271605, 0.00341393 | 0.00228584, 0.00308106, 0.00341504 | 0.00310468 ± 0.0003556 | 0.00292731 ± 0.0005801 |
| vega | 1.04631, 0.973547, 1.2029 | 1.13102, 1.21643, 1.29438 | 1.07425 ± 0.1172 | 1.21394 ± 0.08171 |
| theta | 1.50926, 1.22935, 1.72226 | 1.01358, 1.25398, 1.54612 | 1.48696 ± 0.2472 | 1.27122 ± 0.2667 |
| rho | 1.37237, 1.02804, 1.28008 | 1.56318, 1.55503, 1.94923 | 1.22683 ± 0.1782 | 1.68915 ± 0.2253 |

## Broad and near-expiry/near-strike errors (focus set, observed benchmark)

| Quantity | Region | Uniform mean ± SD | Mixed mean ± SD | Change |
|---|---|---:|---:|---:|
| price | all | 0.151818 ± 0.0273 | 0.161827 ± 0.03541 | mixed vs uniform price RMSE on focus all +6.6% |
| delta | all | 0.0177883 ± 0.002813 | 0.0151405 ± 0.002047 | mixed vs uniform delta RMSE on focus all -14.9% |
| gamma | all | 0.00786114 ± 0.0006555 | 0.00404219 ± 0.0009438 | mixed vs uniform gamma RMSE on focus all -48.6% |
| vega | all | 1.04065 ± 0.1201 | 1.06763 ± 0.1357 | mixed vs uniform vega RMSE on focus all +2.6% |
| theta | all | 4.36883 ± 0.9064 | 1.8801 ± 0.4054 | mixed vs uniform theta RMSE on focus all -57.0% |
| rho | all | 1.21335 ± 0.144 | 1.52889 ± 0.1868 | mixed vs uniform rho RMSE on focus all +26.0% |
| price | near_strike | 0.159281 ± 0.02572 | 0.121963 ± 0.05128 | mixed vs uniform price RMSE on focus near_strike -23.4% |
| delta | near_strike | 0.0208614 ± 0.004076 | 0.0154352 ± 0.002112 | mixed vs uniform delta RMSE on focus near_strike -26.0% |
| gamma | near_strike | 0.0110552 ± 0.0009101 | 0.00519426 ± 0.001315 | mixed vs uniform gamma RMSE on focus near_strike -53.0% |
| vega | near_strike | 0.93387 ± 0.1086 | 0.765332 ± 0.1279 | mixed vs uniform vega RMSE on focus near_strike -18.0% |
| theta | near_strike | 5.89214 ± 1.25 | 2.04047 ± 0.4306 | mixed vs uniform theta RMSE on focus near_strike -65.4% |
| rho | near_strike | 1.05844 ± 0.145 | 1.18164 ± 0.1581 | mixed vs uniform rho RMSE on focus near_strike +11.6% |
| price | near_expiry | 0.17904 ± 0.02342 | 0.117122 ± 0.006799 | mixed vs uniform price RMSE on focus near_expiry -34.6% |
| delta | near_expiry | 0.0267961 ± 0.004679 | 0.0153222 ± 0.001988 | mixed vs uniform delta RMSE on focus near_expiry -42.8% |
| gamma | near_expiry | 0.0136201 ± 0.001075 | 0.00585504 ± 0.001498 | mixed vs uniform gamma RMSE on focus near_expiry -57.0% |
| vega | near_expiry | 1.12259 ± 0.1263 | 0.822004 ± 0.1079 | mixed vs uniform vega RMSE on focus near_expiry -26.8% |
| theta | near_expiry | 7.70472 ± 1.612 | 3.09709 ± 0.6858 | mixed vs uniform theta RMSE on focus near_expiry -59.8% |
| rho | near_expiry | 1.16078 ± 0.1758 | 0.981824 ± 0.1091 | mixed vs uniform rho RMSE on focus near_expiry -15.4% |
| price | near_strike_and_expiry | 0.196586 ± 0.02557 | 0.0872243 ± 0.01765 | mixed vs uniform price RMSE on focus near_strike_and_expiry -55.6% |
| delta | near_strike_and_expiry | 0.0310815 ± 0.006448 | 0.0175827 ± 0.002958 | mixed vs uniform delta RMSE on focus near_strike_and_expiry -43.4% |
| gamma | near_strike_and_expiry | 0.0180277 ± 0.001411 | 0.00757077 ± 0.002001 | mixed vs uniform gamma RMSE on focus near_strike_and_expiry -58.0% |
| vega | near_strike_and_expiry | 1.0053 ± 0.09477 | 0.479597 ± 0.02583 | mixed vs uniform vega RMSE on focus near_strike_and_expiry -52.3% |
| theta | near_strike_and_expiry | 9.73211 ± 2.072 | 3.24219 ± 0.7036 | mixed vs uniform theta RMSE on focus near_strike_and_expiry -66.7% |
| rho | near_strike_and_expiry | 0.954865 ± 0.1978 | 0.779635 ± 0.2115 | mixed vs uniform rho RMSE on focus near_strike_and_expiry -18.4% |

## Constraint violations (frequency and maximum severity reported separately)

| Dataset | Constraint | Uniform frequency | Mixed frequency | Uniform max severity | Mixed max severity |
|---|---|---:|---:|---:|---:|
| validation | price_bounds | 0.0931 | 0.0952 | 0.4168 | 0.6207 |
| validation | delta_bounds | 0.0775 | 0.0721 | 0.08314 | 0.04599 |
| validation | negative_gamma | 0.0506 | 0.0617 | 0.01366 | 0.01402 |
| stress | price_bounds | 0.1592 | 0.1587 | 1.639 | 2.772 |
| stress | delta_bounds | 0.0864 | 0.0662 | 0.07439 | 0.07402 |
| stress | negative_gamma | 0.1478 | 0.1815 | 0.01328 | 0.02199 |

## Comparison with the previous sampling results

The previous sampling results changed sampling and normalization jointly; this study isolates sampling with shared normalization scales.

- theta on focus near_strike_and_expiry: previous mixed-vs-uniform -56.1%, corrected (shared scales) -66.7%.
- gamma on focus near_strike_and_expiry: previous mixed-vs-uniform -43.2%, corrected (shared scales) -58.0%.
- delta on focus near_strike_and_expiry: previous mixed-vs-uniform -41.0%, corrected (shared scales) -43.4%.
- theta on focus near_expiry: previous mixed-vs-uniform -50.4%, corrected (shared scales) -59.8%.
- gamma on focus near_expiry: previous mixed-vs-uniform -42.4%, corrected (shared scales) -57.0%.
- delta on focus near_expiry: previous mixed-vs-uniform -39.0%, corrected (shared scales) -42.8%.
- theta on focus near_strike: previous mixed-vs-uniform -54.2%, corrected (shared scales) -65.4%.
- gamma on focus near_strike: previous mixed-vs-uniform -37.4%, corrected (shared scales) -53.0%.
- delta on focus near_strike: previous mixed-vs-uniform -22.0%, corrected (shared scales) -26.0%.

## Exceptions (measured regressions under shared scales)

- mixed vs uniform price RMSE on focus all: regresses (uniform 0.151818 -> mixed 0.161827, +6.6%).
- mixed vs uniform vega RMSE on focus all: regresses (uniform 1.04065 -> mixed 1.06763, +2.6%).
- mixed vs uniform rho RMSE on focus all: regresses (uniform 1.21335 -> mixed 1.52889, +26.0%).
- mixed vs uniform rho RMSE on focus near_strike: regresses (uniform 1.05844 -> mixed 1.18164, +11.6%).
- mixed vs uniform price RMSE on validation: regresses (uniform 0.139204 -> mixed 0.189199, +35.9%).
- mixed vs uniform delta RMSE on validation: regresses (uniform 0.0126287 -> mixed 0.0154585, +22.4%).
- mixed vs uniform vega RMSE on validation: regresses (uniform 1.07425 -> mixed 1.21394, +13.0%).
- mixed vs uniform rho RMSE on validation: regresses (uniform 1.22683 -> mixed 1.68915, +37.7%).
- mixed vs uniform price_bounds violation frequency on validation: regresses (uniform 0.093099 -> mixed 0.0952148, +2.3%).
- mixed vs uniform price_bounds maximum severity on validation: regresses (uniform 0.416839 -> mixed 0.620744, +48.9%).
- mixed vs uniform negative_gamma violation frequency on validation: regresses (uniform 0.0506185 -> mixed 0.0616862, +21.9%).
- mixed vs uniform negative_gamma maximum severity on validation: regresses (uniform 0.0136558 -> mixed 0.0140169, +2.6%).
- mixed vs uniform price_bounds maximum severity on stress: regresses (uniform 1.63915 -> mixed 2.77241, +69.1%).
- mixed vs uniform negative_gamma violation frequency on stress: regresses (uniform 0.147786 -> mixed 0.181478, +22.8%).
- mixed vs uniform negative_gamma maximum severity on stress: regresses (uniform 0.0132811 -> mixed 0.0219867, +65.5%).
