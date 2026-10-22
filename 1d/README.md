# Option Pricing — 1D study (frozen)

A reproducible comparison of European option prices and Greeks from analytical
Black–Scholes, conditional randomized quasi-Monte Carlo, Crank–Nicolson,
an anchored PINN and a parity-preserving DeepONet.

## Findings

On the [predefined shared benchmark](results/shared_comparison/v1/report.md),
classical methods are more accurate. Trained neural models evaluate batches
quickly but show larger near-expiry Greek errors and some financial-constraint
violations. Including training, labels and selection, no overall computational
advantage has been demonstrated. Analytical Black–Scholes is the exact
reference under the model assumptions.

The fixed-parameter PINN is compared with all methods on its call surface.
The varying-contract call/put comparison excludes that PINN; it does not
generalize across contract parameters. See the
[frozen protocol](results/shared_comparison/v1/protocol.md) for cases,
metrics, timing and checkpoint identities.

## Repository

| Path | Contents |
|---|---|
| `src/` | Pricing methods and neural models |
| `experiments/` | Training, evaluation, configurations and benchmark verification |
| `tests/` | Regression tests |
| `results/` | Selected models, datasets, raw comparison outputs and study evidence |
| `report/` | [Paper](report/main.pdf), [LaTeX](report/main.tex), figures and [Overleaf bundle](report/overleaf_bundle.zip) |

The [milestone record](results/milestone_1d/README.md) documents the saved
comparison. This study is frozen: tag `1d-v1.0` marks it at its original
repository-root paths. Treat this folder as read-only; new work belongs in the
higher-dimensional folders, starting with [`../2d/`](../2d/README.md).

## Reproduce

Run every command from this `1d/` folder; paths in the frozen evidence are
relative to it.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-benchmark.txt
.\.venv\Scripts\python.exe -m experiments.comparison.verify_milestone
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m experiments.comparison.summarize_shared
```

Verification and summary regeneration use saved predictions and do not
retrain models. These test cases are now observed; future model selection
needs new assessment cases. Timing depends on the machine and software
environment. Vega and Rho use unit decimal changes; Theta is calendar-time
change per year.
