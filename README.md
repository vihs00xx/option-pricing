# Option Pricing

Neural pricing methods (PINNs and DeepONets) compared with analytical and
classical numerical methods for European options, first on one stock and then
on baskets of correlated stocks.

## Layout

The repository holds two self-contained subprojects. Each has its own `src/`,
`experiments/`, `tests/` and `results/`, and each must be run from its own
folder: saved evidence is hash-pinned to paths relative to that folder.

| Folder | Status | Contents |
|---|---|---|
| [`1d/`](1d/README.md) | Frozen | Analytical Black–Scholes, conditional RQMC, Crank–Nicolson, anchored PINN and parity DeepONet; the shared comparison and the paper |
| [`multi_asset/`](multi_asset/README.md) | Active | Two-stock baskets now, with higher dimensions to follow |

Tag `1d-v1.0` marks the 1D study at its original repository-root paths,
before this split.

## Verify

```powershell
cd 1d
python -m experiments.comparison.verify_milestone
python -m pytest tests -q
cd ..\multi_asset
python -m experiments.basket.verify_results
python -m pytest tests -q
```

Verification reads saved outputs and does not retrain models. Run each test
suite from its own folder; both subprojects use a top-level package named
`src`, so running pytest from the repository root is not supported.
