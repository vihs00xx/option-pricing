# Frozen two-stock test set

Frozen at 2026-10-01T14:40:33.073268+00:00. Final assessment of two-stock pricing methods, neural and classical. No training, tuning or model selection may use these cases or their references.

## Cases

| Split | Cases | Definition |
|---|---:|---|
| interpolation | 128 | inside the training domain |
| short_maturity | 32 | remaining life below the training domain |
| extrapolation | 72 | one parameter outside the training domain |
| anchor | 16 | hand-chosen regimes |

Strike 100; calls and puts alternate within each split. Latin hypercube seed 20261010.

Training domain:

| Parameter | Range |
|---|---|
| moneyness_1 | 0.7 to 1.3 |
| moneyness_2 | 0.7 to 1.3 |
| tau | 0.05 to 2 |
| volatility_1 | 0.1 to 0.6 |
| volatility_2 | 0.1 to 0.6 |
| correlation | -0.8 to 0.9 |
| rate | -0.02 to 0.1 |
| dividend_1 | 0 to 0.06 |
| dividend_2 | 0 to 0.06 |
| weight_1 | 0.2 to 0.8 |

No case duplicates any of the 16 observed development contracts.

## References

Adaptive conditional quadrature of src/basket (shared-factor conditioning); analytic conditional Deltas and spot Hessian; central quadrature bumps for vega, theta, rho and correlation.

| Check | Worst case |
|---|---:|
| Quadrature error estimate | 8.3e-11 |
| Truncation tail bound | 2.2e-26 |
| Asset-conditioning price difference | 3.4e-13 |
| Gauss-Hermite engine, volatility direction, all quantities | 5.6e-05 |
| Gauss-Hermite engine, equal direction, all quantities | 7.8e-03 |
| Relative change from halving reference bumps | 9.2e-06 |

Relative differences divide by max(|reference|, floor), with floors 1e-7 for Gamma and 1e-4 otherwise. The engine cross-check is an independent factorization of the same model. With equal loadings and strongly unequal volatilities its shared factor is small and 128 nodes under-resolve some Gammas and correlation sensitivities; finite differences of the reference price confirm the reference in those cases.

Volatility-direction differences above 1e-6 relative:

| Case | Quantity | Reference | Engine minus reference | Relative |
|---|---|---:|---:|---:|
| extrapolation_correlation_042 | gamma_22 | 0.00450633 | 2.5e-07 | 5.6e-05 |
| short_maturity_027 | vega_2 | -3.37508e-10 | 7.8e-10 | 7.8e-06 |
| anchor_16_deep_itm_short_put | vega_2 | 3.579e-07 | 2.9e-10 | 2.9e-06 |
| short_maturity_032 | vega_1 | -2.66454e-10 | 2.3e-10 | 2.3e-06 |

These are either under-resolved engine integrals at extreme correlation (more nodes converge to the reference) or sensitivities that are zero to round-off. No pricing method has been evaluated on these cases.

## Rules

- Train, tune and select models only on data generated with other seeds; never on these cases or references.
- Before evaluating any model on these cases, freeze its checkpoint hash, settings and the evaluation protocol.
- Report every method on every split; do not drop cases after seeing results.
- Once any method is evaluated here, the cases are observed; later model changes need a new test set.

`protocol.sha256` identifies `protocol.json`, which pins `cases.json` and `references.csv`; `metadata.json` pins every output and the generating sources. Verify with `python -m experiments.verify results/test_set_v1`.
