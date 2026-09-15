# Option Pricing

An undergraduate project comparing classical numerical methods, PINNs and DeepONet for European option pricing and Greeks.

## Current work

`brownian.py` simulates Brownian motion and geometric Brownian motion. It checks terminal moments and quadratic variation, and plots sample paths. This is a foundations exercise; it does not yet price options.

## Run locally

Install Python, then run these commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe brownian.py
```

Close each plot window to let the script continue.

## Planned work

- Analytical Black-Scholes prices and Greeks
- Monte Carlo pricing and confidence intervals
- Crank-Nicolson finite differences
- PINN and DeepONet comparisons
- Indian market-data case study
- Multi-asset basket options


