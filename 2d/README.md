# Option Pricing — 2D (two-stock baskets)

European basket options on two correlated stocks under risk-neutral GBM. The
neural methods will be compared against the classical baselines developed
here. Higher dimensions live in sibling folders.

## Two-stock classical baseline

The [two-asset study](results/basket/two_asset_v1/report.md) adds correlated
GBM, a geometric-basket analytical check, conditional quadrature, Sobol RQMC
prices and Greeks, and a 2D ADI solver. These exploratory cases are observed;
they cannot serve as an untouched neural test set.

| Path | Contents |
|---|---|
| `src/basket/` | Contract, geometric closed form, conditional quadrature, RQMC estimators and the MCS-ADI solver |
| `src/black_scholes.py` | Byte-identical copy of the 1D analytical module, used for ADI boundary values and pinned by the study's hashes |
| `experiments/basket/` | Study runner and evidence verifier |
| `experiments/config/classical/basket_settings.json` | Study settings and cases |
| `tests/` | Regression tests |
| `results/basket/two_asset_v1/` | Saved study outputs and metadata |

Known limitations: the ADI baseline converges slowly near expiry and at strong
negative correlation. Its default far boundary does not scale with volatility,
so check domain convergence before treating ADI output as a reference. The
two-asset quadrature is the accurate reference; it does not extend beyond two
stocks.

## Reproduce

Run every command from this `2d/` folder. The study used Windows,
Python 3.13.3, NumPy 2.1.0 and SciPy 1.15.3.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m experiments.basket.verify_results
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m experiments.basket.run_classical --output results/basket/new_run
```

The last command repeats the full study into a new folder; it refuses to
overwrite existing output without `--overwrite`.
