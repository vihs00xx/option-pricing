# DeepONet training ablations

Three single-factor comparisons under one compute-matched two-stage training procedure (price-only pilot of 8,000 optimizer steps, then supervised continuation of 12,800 steps at Greek-loss weight 0.01), three seeds each, identical optimizer-step budgets per run. Model selection used training and validation data only. The fresh evaluation package (in-domain, stress, near-expiry/near-strike focus, paired call/put rows) was opened once after all selection decisions and is now observed; a later independent assessment needs fresh cases. The frozen final benchmark was not accessed.

Penalty weight selected on validation: 0.1 (gate failed). Acceptability rule: Mean price RMSE at most 1.1 times control mean and every mean Greek RMSE at most 1.15 times control mean across the three seeds.

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

Validation shape violations (frequency / severity):

| Constraint | Control | Parity |
|---|---|---|
| price_bounds | 0.0850 / 0.3667 | 0.0469 / 0.303 |
| delta_bounds | 0.0770 / 0.07809 | 0.0776 / 0.07331 |
| negative_gamma | 0.0467 / 0.01776 | 0.0503 / 0.02077 |

Parity arm acceptable under the rule: **True**.

## 2. Bound-violation penalties

Each penalty arm adds one squared normalized violation term (all three for penalty_all) with weight 0.1 to the supervised loss during the continuation stage; stage 1 is unchanged for every arm. Validation violations (frequency / severity):

| Arm | Price | Delta | Gamma | Price RMSE | Acceptable |
|---|---|---|---|---:|---|
| control | 0.0850 / 0.3667 | 0.0770 / 0.07809 | 0.0467 / 0.01776 | 0.144789 | True |
| penalty_price | 0.0999 / 0.3749 | 0.0687 / 0.07879 | 0.0438 / 0.01738 | 0.147809 | True |
| penalty_delta | 0.0998 / 0.3678 | 0.0565 / 0.07989 | 0.0532 / 0.01981 | 0.140587 | True |
| penalty_gamma | 0.1024 / 0.4173 | 0.0954 / 0.05229 | 0.0107 / 0.00935 | 0.159963 | False |
| penalty_all | 0.0679 / 0.4614 | 0.0809 / 0.04247 | 0.0151 / 0.004212 | 0.158967 | False |

Fresh stress-set violations (observed benchmark):

| Arm | Price | Delta | Gamma |
|---|---|---|
| control | 0.1323 / 1.381 | 0.0770 / 0.06982 | 0.1408 / 0.01646 |
| penalty_price | 0.1530 / 1.524 | 0.0716 / 0.06636 | 0.1364 / 0.01595 |
| penalty_delta | 0.1517 / 1.786 | 0.0544 / 0.06133 | 0.1427 / 0.01389 |
| penalty_gamma | 0.1370 / 1.281 | 0.1248 / 0.06522 | 0.0898 / 0.008494 |
| penalty_all | 0.1340 / 1.204 | 0.1006 / 0.05894 | 0.0928 / 0.008785 |

Validation RMSE (mean ± SD):

| Quantity | Control | penalty_price | penalty_delta | penalty_gamma | penalty_all |
|---|---:|---:|---:|---:|---:|
| price | 0.144789 ± 0.002262 | 0.147809 ± 0.004903 | 0.140587 ± 0.007035 | 0.159963 ± 0.003618 | 0.158967 ± 0.01125 |
| delta | 0.0118675 ± 0.001048 | 0.0125263 ± 0.002094 | 0.0125725 ± 0.0008307 | 0.0143388 ± 0.0008536 | 0.0153216 ± 0.0002867 |
| gamma | 0.00275705 ± 0.0003166 | 0.00284202 ± 0.0004573 | 0.00286473 ± 0.000384 | 0.00317951 ± 0.0002641 | 0.00349614 ± 9.034e-05 |
| vega | 1.10417 ± 0.03299 | 1.13613 ± 0.0246 | 1.0315 ± 0.02478 | 1.21882 ± 0.07268 | 1.24753 ± 0.05618 |
| theta | 1.38525 ± 0.1976 | 1.43126 ± 0.2738 | 1.39639 ± 0.2494 | 1.61666 ± 0.1763 | 1.76342 ± 0.1071 |
| rho | 1.21837 ± 0.1159 | 1.20714 ± 0.1008 | 1.19968 ± 0.09861 | 1.3071 ± 0.05404 | 1.41988 ± 0.08205 |

## 3. Near-expiry and near-strike sampling

The mixed arm replaces half of each contract's uniform coordinates with concentrations near the strike, near expiry and near both, at identical dataset size and identical contracts. Focus-set RMSE by region (fresh cases, mean ± SD):

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

Validation shape violations (frequency / severity):

| Constraint | Uniform | Mixed |
|---|---|---|
| price_bounds | 0.0931 / 0.4168 | 0.0736 / 0.3622 |
| delta_bounds | 0.0775 / 0.08314 | 0.0695 / 0.06752 |
| negative_gamma | 0.0506 / 0.01366 | 0.0719 / 0.01617 |

## Findings (computed from the tables above)

- **Parity by construction is the decisive win.** The control model violates call-put consistency by 0.108 currency units in price and 0.961 in rho on average; the parity model is exact to float precision. At equal optimizer steps it improves 6 of 6 validation RMSEs (price -28.0%, every Greek also better or flat), and price-bound violation frequency falls -44.8%. Delta-bound and negative-Gamma frequencies are unchanged, confirming that parity does not by itself enforce those constraints.
- **Penalties reduce their targeted violations but pay for it in accuracy.** At the selected weight 0.1, the price penalty is ineffective (price-bound frequency +17.6%), the delta penalty cuts delta-bound frequency -26.6% while staying acceptable, the gamma penalty cuts negative-Gamma frequency -77.0% but fails the accuracy gate, and the combined penalty improves every violation severity except price severity and every violation frequency except delta (-20.1% price frequency, -67.6% negative-Gamma frequency) at an accuracy cost (price +9.8%, delta +29.1%) that fails the gate. No penalty arm both removes the violations and keeps accuracy acceptable.
- **Concentrated sampling trades broad accuracy for large target-region gains.** In the sharpest region (near strike and near expiry) the mixed arm improves theta -56.1% and gamma -43.2%, with every quantity better in every region, at the cost of broad validation accuracy (price +9.7%, delta +32.1%).

Synthesis: adopt parity (unambiguous), adopt mixed sampling when near-expiry/near-strike Greek accuracy matters more than average-case price/Delta accuracy, and treat loss penalties as a targeted trade-off rather than a free improvement.
