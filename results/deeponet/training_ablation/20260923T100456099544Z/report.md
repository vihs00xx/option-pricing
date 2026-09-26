# DeepONet training ablations

Three single-factor comparisons under one two-stage training procedure (180 s price-only pilot, then 300 s supervised continuation at Greek-loss weight 0.01), three seeds each, identical wall budgets and concurrency. Model selection used training and validation data only. The fresh evaluation package (in-domain, stress, near-expiry/near-strike focus, paired call/put rows) was opened once after all selection decisions and is now observed; a later independent assessment needs fresh cases. The frozen final benchmark was not accessed.

Penalty weight selected on validation: 0.1 (gate passed). Acceptability rule: Mean price RMSE at most 1.1 times control mean and every mean Greek RMSE at most 1.15 times control mean across the three seeds.

## 1. Call-put parity by construction

The parity arm predicts the shared midpoint (C+P)/2 and derives both prices from C-P = S exp(-q tau) - K exp(-r tau); the control arm predicts prices directly. Parity residuals are absolute call-put inconsistencies of each prediction relation on paired rows (mean over rows, mean over seeds):

| Quantity | Control | Parity |
|---|---:|---:|
| price | 0.169294 | 2.14293e-16 |
| delta | 0.0120314 | 2.62738e-17 |
| gamma | 0.00112325 | 0 |
| vega | 0.628019 | 0 |
| theta | 0.443322 | 7.13294e-16 |
| rho | 1.87838 | 6.10623e-16 |

Validation RMSE (mean ± SD over seeds):

| Quantity | Control | Parity |
|---|---:|---:|
| price | 0.25061 ± 0.01473 | 0.16558 ± 0.01606 |
| delta | 0.0232398 ± 0.002834 | 0.0163877 ± 0.001892 |
| gamma | 0.00532727 ± 0.0004169 | 0.00416145 ± 0.0004847 |
| vega | 1.83743 ± 0.1631 | 1.29971 ± 0.152 |
| theta | 2.80921 ± 0.2975 | 1.9489 ± 0.2213 |
| rho | 2.28204 ± 0.1784 | 1.38564 ± 0.1594 |

Fresh in-domain RMSE (observed benchmark):

| Quantity | Control | Parity |
|---|---:|---:|
| price | 0.266742 ± 0.01393 | 0.167287 ± 0.01459 |
| delta | 0.0227729 ± 0.002631 | 0.0162066 ± 0.002098 |
| gamma | 0.00456019 ± 0.000337 | 0.00366676 ± 0.0004559 |
| vega | 1.84347 ± 0.1476 | 1.2625 ± 0.1426 |
| theta | 3.19983 ± 0.3177 | 2.22621 ± 0.3011 |
| rho | 2.31223 ± 0.1246 | 1.29533 ± 0.1315 |

Fresh stress RMSE (observed benchmark):

| Quantity | Control | Parity |
|---|---:|---:|
| price | 0.561469 ± 0.07648 | 0.324283 ± 0.04483 |
| delta | 0.0395789 ± 0.005375 | 0.0267977 ± 0.002919 |
| gamma | 0.00635405 ± 0.0006958 | 0.00480285 ± 0.0007614 |
| vega | 3.03837 ± 0.2552 | 2.21009 ± 0.2962 |
| theta | 4.1986 ± 0.3213 | 3.30581 ± 0.2089 |
| rho | 5.41426 ± 0.4796 | 2.9747 ± 0.3887 |

Validation shape violations (frequency / severity):

| Constraint | Control | Parity |
|---|---|---|
| price_bounds | 0.0908 / 0.7206 | 0.0913 / 0.558 |
| delta_bounds | 0.0679 / 0.06415 | 0.0618 / 0.07183 |
| negative_gamma | 0.0503 / 0.009911 | 0.0500 / 0.008076 |

Parity arm acceptable under the rule: **True**.

## 2. Bound-violation penalties

Each penalty arm adds one squared normalized violation term (all three for penalty_all) with weight 0.1 to the supervised loss during the continuation stage; stage 1 is unchanged for every arm. Validation violations (frequency / severity):

| Arm | Price | Delta | Gamma | Price RMSE | Acceptable |
|---|---|---|---|---:|---|
| control | 0.0908 / 0.7206 | 0.0679 / 0.06415 | 0.0503 / 0.009911 | 0.25061 | True |
| penalty_price | 0.0881 / 0.8075 | 0.0658 / 0.06625 | 0.0490 / 0.01015 | 0.235059 | True |
| penalty_delta | 0.0952 / 0.6976 | 0.0474 / 0.0584 | 0.0562 / 0.01095 | 0.233539 | True |
| penalty_gamma | 0.0625 / 0.6405 | 0.0983 / 0.06887 | 0.0116 / 0.003554 | 0.245106 | True |
| penalty_all | 0.0760 / 0.5956 | 0.0744 / 0.04755 | 0.0132 / 0.004848 | 0.217866 | True |

Fresh stress-set violations (observed benchmark):

| Arm | Price | Delta | Gamma |
|---|---|---|
| control | 0.1419 / 3.74 | 0.0620 / 0.1039 | 0.1737 / 0.02039 |
| penalty_price | 0.1400 / 3.545 | 0.0614 / 0.1056 | 0.1572 / 0.01923 |
| penalty_delta | 0.1562 / 3.943 | 0.0475 / 0.0988 | 0.1636 / 0.02772 |
| penalty_gamma | 0.0993 / 3.644 | 0.0999 / 0.1319 | 0.1074 / 0.02239 |
| penalty_all | 0.1248 / 3.199 | 0.0737 / 0.08571 | 0.1058 / 0.01822 |

Validation RMSE (mean ± SD):

| Quantity | Control | penalty_price | penalty_delta | penalty_gamma | penalty_all |
|---|---:|---:|---:|---:|---:|
| price | 0.25061 ± 0.01473 | 0.235059 ± 0.03537 | 0.233539 ± 0.03017 | 0.245106 ± 0.01661 | 0.217866 ± 0.02876 |
| delta | 0.0232398 ± 0.002834 | 0.0224094 ± 0.003641 | 0.0219043 ± 0.002377 | 0.0233958 ± 0.0007659 | 0.0209557 ± 0.003832 |
| gamma | 0.00532727 ± 0.0004169 | 0.00515131 ± 0.0005977 | 0.00495986 ± 0.0002709 | 0.00512599 ± 0.000271 | 0.00478525 ± 0.001012 |
| vega | 1.83743 ± 0.1631 | 1.77143 ± 0.2562 | 1.66397 ± 0.1433 | 1.67838 ± 0.112 | 1.61361 ± 0.2773 |
| theta | 2.80921 ± 0.2975 | 2.70595 ± 0.4353 | 2.62121 ± 0.3025 | 2.71534 ± 0.2374 | 2.47455 ± 0.5009 |
| rho | 2.28204 ± 0.1784 | 2.25511 ± 0.2239 | 2.07674 ± 0.1645 | 2.02995 ± 0.05631 | 1.8331 ± 0.379 |

## 3. Near-expiry and near-strike sampling

The mixed arm replaces half of each contract's uniform coordinates with concentrations near the strike, near expiry and near both, at identical dataset size and identical contracts. Focus-set RMSE by region (fresh cases, mean ± SD):

| Quantity | Region | Uniform | Mixed |
|---|---|---:|---:|
| price | all | 0.130509 ± 0.01575 | 0.134586 ± 0.02497 |
| delta | all | 0.0158006 ± 0.001297 | 0.0132329 ± 0.002736 |
| gamma | all | 0.00744581 ± 0.00033 | 0.00436384 ± 0.0009871 |
| vega | all | 0.987949 ± 0.03438 | 0.868827 ± 0.07558 |
| theta | all | 3.79059 ± 0.5848 | 1.96316 ± 0.6544 |
| rho | all | 1.25781 ± 0.05688 | 1.23434 ± 0.2439 |
| price | near_strike | 0.134654 ± 0.01422 | 0.11823 ± 0.03384 |
| delta | near_strike | 0.0182518 ± 0.001747 | 0.0137084 ± 0.002658 |
| gamma | near_strike | 0.0104859 ± 0.0004617 | 0.00572635 ± 0.001335 |
| vega | near_strike | 0.907394 ± 0.0231 | 0.639292 ± 0.02206 |
| theta | near_strike | 5.09847 ± 0.7816 | 2.25043 ± 0.9681 |
| rho | near_strike | 1.12507 ± 0.06703 | 1.00551 ± 0.2299 |
| price | near_expiry | 0.152433 ± 0.02207 | 0.130709 ± 0.02197 |
| delta | near_expiry | 0.0237721 ± 0.00162 | 0.0145903 ± 0.002258 |
| gamma | near_expiry | 0.0128995 ± 0.0005308 | 0.00649008 ± 0.001268 |
| vega | near_expiry | 1.09368 ± 0.1932 | 0.735742 ± 0.0627 |
| theta | near_expiry | 6.67718 ± 1.034 | 3.27589 ± 1.08 |
| rho | near_expiry | 1.04022 ± 0.09612 | 0.693504 ± 0.1783 |
| price | near_strike_and_expiry | 0.165137 ± 0.03086 | 0.12138 ± 0.01753 |
| delta | near_strike_and_expiry | 0.0270372 ± 0.002232 | 0.0166393 ± 0.002854 |
| gamma | near_strike_and_expiry | 0.0171064 ± 0.0007098 | 0.00841857 ± 0.001691 |
| vega | near_strike_and_expiry | 0.920355 ± 0.1748 | 0.572572 ± 0.05682 |
| theta | near_strike_and_expiry | 8.41686 ± 1.288 | 3.57422 ± 1.513 |
| rho | near_strike_and_expiry | 0.90923 ± 0.08473 | 0.47498 ± 0.1419 |

Validation RMSE:

| Quantity | Uniform | Mixed |
|---|---:|---:|
| price | 0.124558 ± 0.01861 | 0.143611 ± 0.02375 |
| delta | 0.0111066 ± 0.001083 | 0.013132 ± 0.003205 |
| gamma | 0.00281168 ± 0.0002501 | 0.0029353 ± 0.0008951 |
| vega | 0.970833 ± 0.01515 | 0.982893 ± 0.1346 |
| theta | 1.33157 ± 0.1441 | 1.18684 ± 0.2436 |
| rho | 1.08783 ± 0.05182 | 1.22176 ± 0.2657 |

Validation shape violations (frequency / severity):

| Constraint | Uniform | Mixed |
|---|---|---|
| price_bounds | 0.0874 / 0.3622 | 0.0695 / 0.4226 |
| delta_bounds | 0.0729 / 0.07347 | 0.0729 / 0.03806 |
| negative_gamma | 0.0493 / 0.01563 | 0.0636 / 0.01245 |

---

**Addendum (2026-09-23, post-run audit).** This run used wall-clock budgets (180 s pilot / 300 s
continuation). On this host wall time proved an unstable compute proxy: identical runs in different
waves completed between 3,493 and 16,483 optimizer steps in the same 300 s (within-wave spread < 1%;
interleaved timings show 2-3x per-step wall drift at minute scale; the datasets contain no denormal
values and identical-shape batches cost the same when interleaved). The parity and penalty
comparisons are therefore step-imbalanced (parity ~4.6k steps vs control ~3.5k) and their attribution
is confounded; the uniform-vs-mixed continuation was step-matched (16,483 vs 16,294) but its pilots
were not (8,000 vs 7,259). A compute-matched re-run with fixed optimizer-step budgets and fresh
evaluation cases supersedes this study (protocol `deeponet_training_ablation_v2`, later folder in
`results/deeponet/training_ablation/`). The tables above are retained as measured.
