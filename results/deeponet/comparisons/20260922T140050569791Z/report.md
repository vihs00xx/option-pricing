# DeepONet validation comparison

Exploratory, single-seed comparison. The final benchmark was not accessed.

Both arms start from the same selected pilot checkpoint with fresh Adam, learning rate 0.0003, identical shuffle seed and 300 additional CPU seconds. Checkpoint selection minimizes validation price MSE normalized by strike. The shared pilot cost was approximately 143 seconds.

The price-only loss is mean squared price error divided by strike squared. The supervised arm adds the mean squared error across five Greeks, each divided by its training-label RMS squared, with coefficient 1. This is a fixed weighting choice, not a tuned optimum. Theta is per year; Vega and Rho are per unit decimal parameter change.

| RMSE | Pilot | Longer price-only | Price + Greeks |
|---|---:|---:|---:|
| price | 0.303353 | 0.196075 | 0.211373 |
| delta | 0.0434045 | 0.0300661 | 0.0187764 |
| gamma | 0.0105848 | 0.00863149 | 0.00343275 |
| vega | 2.66189 | 2.13294 | 1.45731 |
| theta | 4.08644 | 3.33752 | 1.76482 |
| rho | 5.88474 | 3.40681 | 1.52829 |

## Regional diagnostics

Bins are marginal groups, not causal effects; moneyness is S/K for both option types. Worst individual points and every regional metric are saved alongside this report.

| Arm | Region | Price RMSE | Gamma RMSE | Theta RMSE |
|---|---|---:|---:|---:|
| pilot | remaining_maturity: <0.1 | 0.389123 | 0.0198324 | 9.48332 |
| pilot | moneyness: 0.9-1.1 | 0.350196 | 0.0141306 | 5.99043 |
| pilot | option_type: call | 0.313696 | 0.00769878 | 3.60131 |
| pilot | option_type: put | 0.292644 | 0.0128377 | 4.5198 |
| price_only | remaining_maturity: <0.1 | 0.268875 | 0.0168856 | 7.85121 |
| price_only | moneyness: 0.9-1.1 | 0.221933 | 0.0120172 | 4.8331 |
| price_only | option_type: call | 0.188325 | 0.00591044 | 2.9121 |
| price_only | option_type: put | 0.20353 | 0.0106804 | 3.71454 |
| price_and_greeks | remaining_maturity: <0.1 | 0.182153 | 0.006281 | 3.93685 |
| price_and_greeks | moneyness: 0.9-1.1 | 0.211727 | 0.00513051 | 2.45726 |
| price_and_greeks | option_type: call | 0.217551 | 0.00239559 | 1.54608 |
| price_and_greeks | option_type: put | 0.20501 | 0.00422241 | 1.9593 |

## Last-checkpoint diagnostic

These are diagnostic outputs, not replacements for the price-selected checkpoints.

| Arm | Steps | Seconds | Selected epoch | Last price RMSE | Last Theta RMSE |
|---|---:|---:|---:|---:|---:|
| price_only | 18007 | 300.02 | 1080 | 0.221569 | 3.29518 |
| price_and_greeks | 7410 | 300.04 | 340 | 0.246717 | 1.53789 |

The protocol retains the inherited pilot configuration fields `loss` and `note`; the actual per-arm objective is specified above and by `label_columns_used`, `greek_scales`, and the archived source hash. No confidence intervals or multi-seed superiority claims are supported by this run.
