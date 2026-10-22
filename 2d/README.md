# Option Pricing — 2D (two-stock baskets)

European arithmetic basket options on two correlated stocks under risk-neutral
GBM, priced with a PINN, a DeepONet, plain Monte Carlo, a Sobol RQMC engine and
an ADI grid solver, with the price and all ten Greeks. Higher dimensions live
in sibling folders.

## Best results

Best run of each method on the frozen two-stock test set (`results/test_set_v1`,
248 cases including short-maturity and extrapolation cases). Errors are RMSE
against the semi-analytical reference: the arithmetic basket has no closed form,
but its price is a one-dimensional integral of a closed-form integrand,
evaluated by adaptive quadrature to about 1e-10. Times are per contract for the
price and all ten Greeks on one CPU thread.

| Method | Best run | Price | Delta 1 | Gamma 11 | Vega 1 | Theta | Correlation | Time per contract |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Plain Monte Carlo | 1,000,000 paths | 0.0205 | 2.2e-4 | 4.4e-5 | 0.033 | 0.029 | 0.017 | 0.36 s |
| Sobol RQMC engine | 65,536 draws, geometric control | 0.0012 | 5.4e-6 | 2.3e-7 | 0.0025 | 0.0012 | 0.0014 | 0.83 s |
| ADI grid | 160 intervals, Richardson | 6.0e-5 | 7.9e-6 | 3.3e-6 | 9.1e-4 | 1.8e-4 | 0.0020 | 156 s |
| PINN | `P10_anchored` | 0.049 | 0.0022 | 2.5e-4 | 0.25 | 0.096 | 0.26 | 17 µs |
| DeepONet | `D9_ensemble` | 0.032 | 8.5e-4 | 2.2e-4 | 0.22 | 0.063 | 0.29 | 128 µs |

Neural times are in batches of 4,096 contracts. On the 128 cases inside the
training domain the price RMSE is 0.015 for the PINN and 0.0088 for the
DeepONet, both below plain Monte Carlo with 1,000,000 paths (0.020); most of
the neural error above comes from correlations and volatilities outside the
training domain. Training took hours of CPU time, so the networks pay off
only when many contracts are priced; ADI is the most accurate but works for two
stocks only. Full ladders, splits, Greek validation and
limits are in the [neural study report](results/neural/pinn_deeponet_v1/report.md).

The ADI time is the mean single-threaded duration per contract while 12 worker
processes ran at once (`experiments/neural/benchmark_adi.py`), so it was measured
under contention, unlike the isolated neural times; read the ratio between them
as approximate.

## Known limitations

- **Exactly at expiry the selected networks return NaN at the money.** At tau = 0 the
  Levy base divides zero log-moneyness by zero total volatility, and the similarity coordinate does the same. Away from the strike they
  return the payoff, and no test or validation contract has tau = 0. The four-stock code
  returns the payoff explicitly; this study's code is left as it was sealed.

## Code

| Path | Contents |
|---|---|
| `src/neural/` | `pinn.py`, `deeponet.py`, shared coordinates, PDE residuals, networks, trainer, quadrature reference and plain Monte Carlo |
| `src/pricing/` | Engine for any number of stocks: contract, geometric closed form, conditional estimator, tensor Gauss–Hermite, replicated RQMC, and the two-stock ADI solver |
| `src/basket/` | Frozen two-stock code of the exploratory study; its adaptive quadrature is the reference for the test set |
| `src/black_scholes.py` | Byte-identical copy of the 1D analytical module |
| `experiments/neural/` | Ladder training, selection, test-set scoring, Monte Carlo benchmark, report and verification |
| `experiments/pricing/validate_engine.py` | Engine validation on development cases |
| `experiments/test_set/freeze_test_set.py` | Generator of the frozen test set |
| `experiments/verify.py` | Integrity check for a saved results folder |
| `experiments/config/` | Study settings, the PINN and DeepONet ladders, and the test-set protocol |
| `tests/` | Regression tests |

**Neural pricers.** An exact reduction (homogeneity, forwards, discounting)
maps the ten contract inputs to three coordinates and three parameters. Both
networks learn the correction to the Levy moment-matching approximation, which
carries the payoff exactly. The PINN minimizes the residual of the pricing PDE
(plus reference prices at anchor contracts in its final level); the DeepONet
learns from reference prices and their derivatives plus the PDE residual. Each
was built as a ladder, one technique per level, and the best level on a
separate validation set was frozen before the test set was opened.

**Conditional estimator.** The log-return covariance is split as
`C = s²ttᵀ + AAᵀ` for a positive direction `t`, with `s² = 1/(tᵀC⁻¹t)`. Every
stock rises with the split-off factor, so given the remaining `d−1` Gaussians
the basket crosses the strike at one root and the conditional price, Deltas and
spot Hessian are closed-form. Only `d−1` dimensions are sampled (RQMC) or
integrated (tensor Gauss–Hermite). The default `t = σ` keeps the split-off
factor large when volatilities differ strongly; `t = 1` reproduces the
shared-factor estimator of the exploratory study. Vega, theta, rho and
correlation sensitivities are central bumps with common draws. The geometric
basket is the optional control variate.

**ADI.** Modified Craig–Sneyd on sinh-clustered grids with the spot on a node,
a far boundary that scales with volatility and maturity, Douglas damping, PDE
theta, optional Richardson extrapolation and bumped parameter sensitivities. It
is a two-stock grid baseline; grid cost grows exponentially with the number of
stocks.

## Results folders

| Folder | Contents |
|---|---|
| [`results/neural/pinn_deeponet_v1/`](results/neural/pinn_deeponet_v1/report.md) | PINN and DeepONet ladders, test-set scores, Monte Carlo benchmark and Greek validation |
| [`results/test_set_v1/`](results/test_set_v1/report.md) | Frozen two-stock test set: 248 cases and references; now observed (the neural pricers were scored on it) |
| [`results/engine_validation_v1/`](results/engine_validation_v1/report.md) | Engine against the independent two-stock quadrature; RQMC accuracy and calibration; ADI convergence |
| [`results/basket/two_asset_v1/`](results/basket/two_asset_v1/report.md) | Exploratory two-stock study (observed cases) |

Later model changes need a new test set.

## Reproduce

Run every command from this `2d/` folder. The studies used Windows,
Python 3.13.3, NumPy 2.1.0, SciPy 1.15.3 and PyTorch 2.8.0 (CPU).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m experiments.neural.verify_results
.\.venv\Scripts\python.exe -m experiments.verify results/test_set_v1
.\.venv\Scripts\python.exe -m experiments.verify results/engine_validation_v1
.\.venv\Scripts\python.exe -m experiments.basket.verify_results
```

Verification reads saved outputs and does not train anything;
`experiments.neural.verify_results` also re-prices the test set with the frozen
networks. Retraining commands are in the neural study report. Saved results pin
the source files that produced them, so changing those sources requires a new
results version.
