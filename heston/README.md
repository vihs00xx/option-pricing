# One-stock Heston: classical pricers against PINN and DeepONet

European calls and puts under the Heston model, price and 11 sensitivities (Delta, Gamma, Vega,
Vanna and Volga with respect to v0, Theta, rate rho, and d/dkappa, d/dtheta, d/dxi, d/drho), per
unit strike. The four Heston parameters are network inputs, so one network covers the whole domain:
S/K 0.7-1.3, tau 0.05-2 (short-maturity panel 0.005-0.05), r -0.02-0.10, q 0-0.06,
v0 and theta 0.01-0.36, kappa 0.5-5, xi 0.1-1, rho -0.9-0.3. The Feller condition is not imposed;
it fails for 22% of the domain.

## Best run per method on the sealed test set (252 cases)

Score = mean over the 12 quantities of RMSE / RMS(reference). Seconds per contract with all 12
quantities, one thread.

| Method | Setting | Test score | Price RMSE | Seconds per contract |
|---|---|---:|---:|---:|
| COS (Fang-Oosterlee, c2 truncation, L = 12) | 256 terms | 0.0019 | 7.9e-6 | 3.7e-3 (9.7e-4 batched) |
| ADI (Hundsdorfer-Verwer) | 160 x 80 x 80 | 0.0073 | 2.8e-5 | 1.8 |
| DeepONet | `D9_ensemble` | 0.073 | 6.5e-4 | 2.4e-4 batched |
| QE-RQMC | 16 x 2^16 points, dt 1/32 | 0.074 | 3.8e-5 | 64 |
| QE Monte Carlo | 10^6 paths, dt 1/8 | 0.063 | 3.5e-4 | 13 |
| PINN | `P10_anchored` | 0.144 | 1.4e-3 | 1.3e-5 batched |

COS is the pricer to use for one stock. The networks pay back their training against ADI and the
simulation methods, but not against COS. They are also not accurate enough to recover the Heston
parameters in calibration (COS recovers them to 3e-6).

## Layout

- `src/heston/`: model and domain (`model.py`), little-trap characteristic function
  (`characteristic.py`), Lewis reference with closed-form integrands for all 12 quantities
  (`reference.py`), its batched torch twin (`torch_reference.py`), COS (`cos.py`), ADI (`adi.py`),
  QE Monte Carlo and RQMC (`monte_carlo.py`), Black-Scholes (`black_scholes.py`).
- `src/neural/`: exact reduction and Black-Scholes base, networks (PINN, DeepONet), PDE residuals,
  labels and Greeks, training (Adam, RAD, L-BFGS, Newton-CG).
- `experiments/`: configs, the Phase 0 reference check, the test-set build, the classical
  benchmark, the ladders, selection, test evaluation, comparison and calibration.
- `results/`: `reference_check_v1` (Phase 0), `test_set_v1` (sealed), `classical_v1` (Phase 1),
  `neural/pinn_deeponet_v1` (Phase 2) and `neural/variance_cap_study`. Each has a `report.md`.

## Known limitations

- **Exactly at expiry the selected networks return NaN at the money.** At tau = 0 the
  Black-Scholes base divides zero log-moneyness by zero total volatility. Away from the strike they
  return the payoff, and no test or validation contract has tau = 0. The four-stock code
  returns the payoff explicitly; this study's code is left as it was sealed.
- **`evaluate_test` checks less than it freezes.** Its `verify_selection()` checks the
  evaluation sources and the two selected checkpoints, but not the recorded settings hashes or
  the other ladder checkpoints it also scores. `verify_results` checks the saved outputs and
  sources afterwards. The four-stock evaluator should use the two-stock check, which covers all
  three.

## Reproduce (from this folder; Python 3.13, CPU)

```bash
pip install -r requirements.txt
python -m pytest tests
python -m experiments.reference.check_reference && python -m experiments.reference.make_report
python -m experiments.test_set.freeze_test_set
python -m experiments.classical.benchmark && python -m experiments.classical.timing && python -m experiments.classical.make_report
python -m experiments.neural.run_ladders
python -m experiments.neural.select_models && python -m experiments.neural.evaluate_test
python -m experiments.neural.load_factor && python -m experiments.neural.compare && python -m experiments.neural.calibration
python -m experiments.neural.make_report && python -m experiments.neural.verify_results
```

The sealed test set and the frozen selection refuse to be rebuilt over existing results.
