# DeepONet training ablations

Three single-factor comparisons under one two-stage training procedure (price-only pilot of 500 epochs, then supervised continuation of 800 epochs at Greek-loss weight 0.01, batch size 512), three seeds per arm. Budgets are matched update counts (identical epochs and batch size per run); equal update counts do not imply equal computation or equal wall time. Actual optimizer updates and runtime are reported separately. Model selection used training and validation data only. The fresh evaluation package (in-domain, stress, near-expiry/near-strike focus, paired call/put rows) was opened once after all selection decisions and is an observed benchmark, not an untouched test; a later independent assessment needs fresh cases. The final benchmark was evaluated once in an earlier study (`results/deeponet/primary/final_test_metrics.json`) and is already observed; the latest work did not access it again.

Penalty weight selected on validation: 0.1 (gate failed, fallback rule applied). Acceptability rule: Mean price RMSE at most 1.1 times control mean and every mean Greek RMSE at most 1.15 times control mean across the three seeds.

## Budget accounting (matched update counts)

Equal update counts do not imply equal computation or wall time.

| Run | Optimizer updates | Runtime (s) | Stop reason |
|---|---:|---:|---|
| control_22092026 | 12800 | 1093.5 | epochs_completed |
| control_22092027 | 12800 | 1092.1 | epochs_completed |
| control_22092028 | 12800 | 1093.6 | epochs_completed |
| mixed_22092026 | 12800 | 843.3 | epochs_completed |
| mixed_22092027 | 12800 | 844.6 | epochs_completed |
| mixed_22092028 | 12800 | 845.8 | epochs_completed |
| parity_22092026 | 12800 | 1378.2 | epochs_completed |
| parity_22092027 | 12800 | 1378.0 | epochs_completed |
| parity_22092028 | 12800 | 1378.3 | epochs_completed |
| penalty_all_22092026 | 12800 | 861.0 | epochs_completed |
| penalty_all_22092027 | 12800 | 844.5 | epochs_completed |
| penalty_all_22092028 | 12800 | 842.8 | epochs_completed |
| penalty_delta_22092026 | 12800 | 1276.1 | epochs_completed |
| penalty_delta_22092027 | 12800 | 1274.7 | epochs_completed |
| penalty_delta_22092028 | 12800 | 1274.2 | epochs_completed |
| penalty_gamma_22092026 | 12800 | 1349.5 | epochs_completed |
| penalty_gamma_22092027 | 12800 | 1349.5 | epochs_completed |
| penalty_gamma_22092028 | 12800 | 1347.0 | epochs_completed |
| penalty_price_22092026 | 12800 | 694.6 | epochs_completed |
| penalty_price_22092027 | 12800 | 694.7 | epochs_completed |
| penalty_price_22092028 | 12800 | 695.0 | epochs_completed |
| uniform_22092026 | 12800 | 879.1 | epochs_completed |
| uniform_22092027 | 12800 | 879.6 | epochs_completed |
| uniform_22092028 | 12800 | 879.9 | epochs_completed |

## 1. Call-put parity by construction

The parity arm predicts the shared midpoint (C+P)/2 and derives both prices from C-P = S exp(-q tau) - K exp(-r tau); the control arm predicts prices directly. Parity residuals are absolute call-put inconsistencies of each prediction relation on paired rows (mean over rows, mean over seeds):

| Quantity | Control | Parity |
|---|---:|---:|
| price | 0.107771 | 1.92798e-16 |
| delta | 0.00671303 | 2.54245e-17 |
| gamma | 0.000836801 | 0 |
| vega | 0.423287 | 0 |
| theta | 0.341272 | 7.16297e-16 |
| rho | 0.961077 | 6.55057e-16 |

Validation RMSE (mean ± SD over seeds):

| Quantity | Control | Parity |
|---|---:|---:|
| price | 0.144789 ± 0.002262 | 0.104216 ± 0.003369 |
| delta | 0.0118675 ± 0.001048 | 0.00942788 ± 0.0002803 |
| gamma | 0.00275705 ± 0.0003166 | 0.00255899 ± 4.572e-05 |
| vega | 1.10417 ± 0.03299 | 0.878999 ± 0.03598 |
| theta | 1.38525 ± 0.1976 | 1.1564 ± 0.08028 |
| rho | 1.21837 ± 0.1159 | 1.05561 ± 0.009834 |

Fresh in-domain RMSE (observed benchmark):

| Quantity | Control | Parity |
|---|---:|---:|
| price | 0.139041 ± 0.003181 | 0.100435 ± 0.006659 |
| delta | 0.0116486 ± 0.001054 | 0.00922245 ± 0.0001259 |
| gamma | 0.00234543 ± 0.0002449 | 0.00207214 ± 9.368e-05 |
| vega | 1.06356 ± 0.06785 | 0.918518 ± 0.02849 |
| theta | 1.61973 ± 0.2868 | 1.32748 ± 0.143 |
| rho | 1.16277 ± 0.08829 | 0.975728 ± 0.08874 |

Fresh stress RMSE (observed benchmark):

| Quantity | Control | Parity |
|---|---:|---:|
| price | 0.273775 ± 0.005402 | 0.233175 ± 0.0331 |
| delta | 0.0204529 ± 0.002111 | 0.0178506 ± 0.001551 |
| gamma | 0.00311649 ± 0.0002691 | 0.00301882 ± 0.0002284 |
| vega | 2.00958 ± 0.06169 | 1.94022 ± 0.1169 |
| theta | 1.60658 ± 0.173 | 1.44664 ± 0.1208 |
| rho | 2.86008 ± 0.2124 | 2.8951 ± 0.1234 |

Validation shape violations, frequency and maximum severity reported separately:

| Constraint | Control frequency | Parity frequency | Control max severity | Parity max severity |
|---|---:|---:|---:|---:|
| price_bounds | 0.0850 | 0.0469 | 0.3667 | 0.303 |
| delta_bounds | 0.0770 | 0.0776 | 0.07809 | 0.07331 |
| negative_gamma | 0.0467 | 0.0503 | 0.01776 | 0.02077 |

Conclusions from measured values:

- validation: parity improves 6 of 6 RMSEs (price, delta, gamma, vega, theta, rho).
- in_domain: parity improves 6 of 6 RMSEs (price, delta, gamma, vega, theta, rho).
- stress: parity improves 5 of 6 RMSEs (price, delta, gamma, vega, theta).
- Parity residual relations: call-put price residual -100.0% (better) (the parity arm is at float round-off by construction).
- Price-bound violation frequency on validation: price-bound frequency -44.8% (better), maximum severity: price-bound severity -17.4% (better).
- Parity arm passes the accuracy acceptance gate: **True**.

## 2. Bound-violation penalties

Each penalty arm adds one squared normalized violation term (all three for penalty_all) with weight 0.1 to the supervised loss during the continuation stage; stage 1 is identical for every arm. Frequency and maximum severity are separate measures:

| Arm | Price frequency | Delta frequency | Gamma frequency | Price max severity | Delta max severity | Gamma max severity | Price RMSE | Acceptable |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| control | 0.0850 | 0.0770 | 0.0467 | 0.3667 | 0.07809 | 0.01776 | 0.144789 | True |
| penalty_price | 0.0999 | 0.0687 | 0.0438 | 0.3749 | 0.07879 | 0.01738 | 0.147809 | True |
| penalty_delta | 0.0998 | 0.0565 | 0.0532 | 0.3678 | 0.07989 | 0.01981 | 0.140587 | True |
| penalty_gamma | 0.1024 | 0.0954 | 0.0107 | 0.4173 | 0.05229 | 0.00935 | 0.159963 | False |
| penalty_all | 0.0679 | 0.0809 | 0.0151 | 0.4614 | 0.04247 | 0.004212 | 0.158967 | False |

Fresh stress-set violations (observed benchmark):

| Arm | Price frequency | Delta frequency | Gamma frequency | Price max severity | Delta max severity | Gamma max severity |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.1323 | 0.0770 | 0.1408 | 1.381 | 0.06982 | 0.01646 |
| penalty_price | 0.1530 | 0.0716 | 0.1364 | 1.524 | 0.06636 | 0.01595 |
| penalty_delta | 0.1517 | 0.0544 | 0.1427 | 1.786 | 0.06133 | 0.01389 |
| penalty_gamma | 0.1370 | 0.1248 | 0.0898 | 1.281 | 0.06522 | 0.008494 |
| penalty_all | 0.1340 | 0.1006 | 0.0928 | 1.204 | 0.05894 | 0.008785 |

Validation RMSE (mean ± SD):

| Quantity | Control | penalty_price | penalty_delta | penalty_gamma | penalty_all |
|---|---:|---:|---:|---:|---:|
| price | 0.144789 ± 0.002262 | 0.147809 ± 0.004903 | 0.140587 ± 0.007035 | 0.159963 ± 0.003618 | 0.158967 ± 0.01125 |
| delta | 0.0118675 ± 0.001048 | 0.0125263 ± 0.002094 | 0.0125725 ± 0.0008307 | 0.0143388 ± 0.0008536 | 0.0153216 ± 0.0002867 |
| gamma | 0.00275705 ± 0.0003166 | 0.00284202 ± 0.0004573 | 0.00286473 ± 0.000384 | 0.00317951 ± 0.0002641 | 0.00349614 ± 9.034e-05 |
| vega | 1.10417 ± 0.03299 | 1.13613 ± 0.0246 | 1.0315 ± 0.02478 | 1.21882 ± 0.07268 | 1.24753 ± 0.05618 |
| theta | 1.38525 ± 0.1976 | 1.43126 ± 0.2738 | 1.39639 ± 0.2494 | 1.61666 ± 0.1763 | 1.76342 ± 0.1071 |
| rho | 1.21837 ± 0.1159 | 1.20714 ± 0.1008 | 1.19968 ± 0.09861 | 1.3071 ± 0.05404 | 1.41988 ± 0.08205 |

Conclusions from measured values and gates:

- penalty_price price_bounds frequency: penalty_price price_bounds frequency +17.6% (worse), maximum severity: penalty_price price_bounds severity +2.3% (worse).
- penalty_delta delta_bounds frequency: penalty_delta delta_bounds frequency -26.6% (better), maximum severity: penalty_delta delta_bounds severity +2.3% (worse).
- penalty_gamma negative_gamma frequency: penalty_gamma negative_gamma frequency -77.0% (better), maximum severity: penalty_gamma negative_gamma severity -47.4% (better).
- penalty_all price_bounds frequency: penalty_all price_bounds frequency -20.1% (better), maximum severity: penalty_all price_bounds severity +25.8% (worse).
- penalty_all delta_bounds frequency: penalty_all delta_bounds frequency +5.1% (worse), maximum severity: penalty_all delta_bounds severity -45.6% (better).
- penalty_all negative_gamma frequency: penalty_all negative_gamma frequency -67.6% (better), maximum severity: penalty_all negative_gamma severity -76.3% (better).
- Gates: penalty_price=pass, penalty_delta=pass, penalty_gamma=FAIL, penalty_all=FAIL.

## 3. Near-expiry and near-strike sampling

The mixed arm replaces half of each contract's uniform coordinates with concentrations near the strike, near expiry and near both, at identical dataset size and identical contracts. Both arms are normalized by one fixed set of Greek scales computed from the uniform arm's training labels (recorded in both runs' metadata), so the coordinate distribution is the only changed factor. Focus-set RMSE by region (fresh cases, mean ± SD):

| Quantity | Region | Uniform | Mixed |
|---|---|---:|---:|
| price | all | 0.151818 ± 0.0273 | 0.133947 ± 0.021 |
| delta | all | 0.0177883 ± 0.002813 | 0.0161042 ± 0.004023 |
| gamma | all | 0.00786114 ± 0.0006555 | 0.0053015 ± 0.0009666 |
| vega | all | 1.04065 ± 0.1201 | 0.967476 ± 0.1009 |
| theta | all | 4.36883 ± 0.9064 | 2.28547 ± 0.5859 |
| rho | all | 1.21335 ± 0.144 | 1.23165 ± 0.07146 |
| price | near_strike | 0.159281 ± 0.02572 | 0.111549 ± 0.01939 |
| delta | near_strike | 0.0208614 ± 0.004076 | 0.0162657 ± 0.003994 |
| gamma | near_strike | 0.0110552 ± 0.0009101 | 0.00692384 ± 0.001318 |
| vega | near_strike | 0.93387 ± 0.1086 | 0.675672 ± 0.1213 |
| theta | near_strike | 5.89214 ± 1.25 | 2.698 ± 0.8342 |
| rho | near_strike | 1.05844 ± 0.145 | 0.905336 ± 0.04243 |
| price | near_expiry | 0.17904 ± 0.02342 | 0.103944 ± 0.02443 |
| delta | near_expiry | 0.0267961 ± 0.004679 | 0.0163544 ± 0.003492 |
| gamma | near_expiry | 0.0136201 ± 0.001075 | 0.00785166 ± 0.00133 |
| vega | near_expiry | 1.12259 ± 0.1263 | 0.864998 ± 0.1735 |
| theta | near_expiry | 7.70472 ± 1.612 | 3.81856 ± 0.9695 |
| rho | near_expiry | 1.16078 ± 0.1758 | 0.771648 ± 0.1102 |
| price | near_strike_and_expiry | 0.196586 ± 0.02557 | 0.0762755 ± 0.0167 |
| delta | near_strike_and_expiry | 0.0310815 ± 0.006448 | 0.0183249 ± 0.003737 |
| gamma | near_strike_and_expiry | 0.0180277 ± 0.001411 | 0.0102409 ± 0.001745 |
| vega | near_strike_and_expiry | 1.0053 ± 0.09477 | 0.496774 ± 0.04896 |
| theta | near_strike_and_expiry | 9.73211 ± 2.072 | 4.2696 ± 1.282 |
| rho | near_strike_and_expiry | 0.954865 ± 0.1978 | 0.519834 ± 0.07553 |

Validation RMSE:

| Quantity | Uniform | Mixed |
|---|---:|---:|
| price | 0.139204 ± 0.0251 | 0.152723 ± 0.02654 |
| delta | 0.0126287 ± 0.001551 | 0.0166805 ± 0.003686 |
| gamma | 0.00310468 ± 0.0003556 | 0.00360066 ± 0.0007494 |
| vega | 1.07425 ± 0.1172 | 1.11084 ± 0.1462 |
| theta | 1.48696 ± 0.2472 | 1.3395 ± 0.2379 |
| rho | 1.22683 ± 0.1782 | 1.37149 ± 0.1407 |

Validation shape violations, frequency and maximum severity:

| Constraint | Uniform frequency | Mixed frequency | Uniform max severity | Mixed max severity |
|---|---:|---:|---:|---:|
| price_bounds | 0.0931 | 0.0736 | 0.4168 | 0.3622 |
| delta_bounds | 0.0775 | 0.0695 | 0.08314 | 0.06752 |
| negative_gamma | 0.0506 | 0.0719 | 0.01366 | 0.01617 |

Conclusions from measured values:

- mixed vs uniform price RMSE on focus all -11.8% (better).
- mixed vs uniform delta RMSE on focus all -9.5% (better).
- mixed vs uniform gamma RMSE on focus all -32.6% (better).
- mixed vs uniform vega RMSE on focus all -7.0% (better).
- mixed vs uniform theta RMSE on focus all -47.7% (better).
- mixed vs uniform rho RMSE on focus all +1.5% (worse).
- mixed vs uniform price RMSE on focus near_strike -30.0% (better).
- mixed vs uniform delta RMSE on focus near_strike -22.0% (better).
- mixed vs uniform gamma RMSE on focus near_strike -37.4% (better).
- mixed vs uniform vega RMSE on focus near_strike -27.6% (better).
- mixed vs uniform theta RMSE on focus near_strike -54.2% (better).
- mixed vs uniform rho RMSE on focus near_strike -14.5% (better).
- mixed vs uniform price RMSE on focus near_expiry -41.9% (better).
- mixed vs uniform delta RMSE on focus near_expiry -39.0% (better).
- mixed vs uniform gamma RMSE on focus near_expiry -42.4% (better).
- mixed vs uniform vega RMSE on focus near_expiry -22.9% (better).
- mixed vs uniform theta RMSE on focus near_expiry -50.4% (better).
- mixed vs uniform rho RMSE on focus near_expiry -33.5% (better).
- mixed vs uniform price RMSE on focus near_strike_and_expiry -61.2% (better).
- mixed vs uniform delta RMSE on focus near_strike_and_expiry -41.0% (better).
- mixed vs uniform gamma RMSE on focus near_strike_and_expiry -43.2% (better).
- mixed vs uniform vega RMSE on focus near_strike_and_expiry -50.6% (better).
- mixed vs uniform theta RMSE on focus near_strike_and_expiry -56.1% (better).
- mixed vs uniform rho RMSE on focus near_strike_and_expiry -45.6% (better).
- mixed vs uniform price RMSE on validation +9.7% (worse).
- mixed vs uniform delta RMSE on validation +32.1% (worse).
- mixed vs uniform gamma RMSE on validation +16.0% (worse).
- mixed vs uniform vega RMSE on validation +3.4% (worse).
- mixed vs uniform theta RMSE on validation -9.9% (better).
- mixed vs uniform rho RMSE on validation +11.8% (worse).

## Exceptions and gate results

Every measured regression and failed gate in this report:

- parity vs control rho RMSE on stress: regresses (control 2.86008 -> parity 2.8951, +1.2%).
- parity vs control negative_gamma violation frequency on validation: regresses (control 0.0467122 -> parity 0.050293, +7.7%).
- parity vs control negative_gamma maximum severity on validation: regresses (control 0.0177598 -> parity 0.0207678, +16.9%).
- penalty_gamma: fails the accuracy acceptance gate (Mean price RMSE at most 1.1 times control mean and every mean Greek RMSE at most 1.15 times control mean across the three seeds.).
- penalty_all: fails the accuracy acceptance gate (Mean price RMSE at most 1.1 times control mean and every mean Greek RMSE at most 1.15 times control mean across the three seeds.).
- penalty_price price_bounds frequency: regresses (control 0.0849609 -> penalty_price 0.0999349, +17.6%).
- penalty_price price_bounds severity: regresses (control 0.366671 -> penalty_price 0.374935, +2.3%).
- penalty_price price RMSE: regresses (control 0.144789 -> penalty_price 0.147809, +2.1%).
- penalty_price delta RMSE: regresses (control 0.0118675 -> penalty_price 0.0125263, +5.6%).
- penalty_delta delta_bounds severity: regresses (control 0.0780866 -> penalty_delta 0.0798915, +2.3%).
- penalty_delta delta RMSE: regresses (control 0.0118675 -> penalty_delta 0.0125725, +5.9%).
- penalty_gamma price RMSE: regresses (control 0.144789 -> penalty_gamma 0.159963, +10.5%).
- penalty_gamma delta RMSE: regresses (control 0.0118675 -> penalty_gamma 0.0143388, +20.8%).
- penalty_all price_bounds severity: regresses (control 0.366671 -> penalty_all 0.461413, +25.8%).
- penalty_all delta_bounds frequency: regresses (control 0.0769857 -> penalty_all 0.0808919, +5.1%).
- penalty_all price RMSE: regresses (control 0.144789 -> penalty_all 0.158967, +9.8%).
- penalty_all delta RMSE: regresses (control 0.0118675 -> penalty_all 0.0153216, +29.1%).
- mixed vs uniform rho RMSE on focus all: regresses (uniform 1.21335 -> mixed 1.23165, +1.5%).
- mixed vs uniform price RMSE on validation: regresses (uniform 0.139204 -> mixed 0.152723, +9.7%).
- mixed vs uniform delta RMSE on validation: regresses (uniform 0.0126287 -> mixed 0.0166805, +32.1%).
- mixed vs uniform gamma RMSE on validation: regresses (uniform 0.00310468 -> mixed 0.00360066, +16.0%).
- mixed vs uniform vega RMSE on validation: regresses (uniform 1.07425 -> mixed 1.11084, +3.4%).
- mixed vs uniform rho RMSE on validation: regresses (uniform 1.22683 -> mixed 1.37149, +11.8%).
- mixed vs uniform negative_gamma violation frequency on validation: regresses (uniform 0.0506185 -> mixed 0.0719401, +42.1%).
- mixed vs uniform negative_gamma maximum severity on validation: regresses (uniform 0.0136558 -> mixed 0.0161665, +18.4%).

Caveat: Corrected report (generated 2026-09-23 15:58 UTC). The original report.md in this folder is preserved unchanged; corrections cover terminology (matched update counts), gate-driven conclusions, and explicit exceptions.

Caveat: The previous sampling results in this study changed sampling AND normalization jointly: each arm used Greek scales computed from its own training labels. The isolated sampling effect is measured by the corrected sampling study, where both arms share the uniform arm's training-label scales.
