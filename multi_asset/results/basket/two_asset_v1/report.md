# Two-stock classical verification

This is an exploratory baseline study; these cases are now observed and are not a future neural final test.

Arithmetic and geometric basket calls/puts use correlated GBM, constant dividends and positive weights summing to one.
The primary conditional estimator decomposes correlated log returns into a shared Gaussian factor with identical stock loadings and a residual factor. The shared factor is integrated analytically; only the residual is sampled or numerically integrated.
For log-return covariance C, the shared-factor variance is det(C)/(C11+C22-2C12), and the residual loadings are (C11-C12, C12-C22)/sqrt(C11+C22-2C12). Their outer product plus the shared variance times an all-ones matrix reconstructs C.
A separate estimator conditions on the second stock directly and analytically integrates the first stock. Agreement between these distinct quadratures provides an independent representation check. Singular correlations and zero volatilities use the asset-conditional price path; regular Greek estimation requires positive stock volatilities and |correlation| < 1.
The +/-12 normal truncation has a separately computed price-tail bound. The quadrature error is a numerical estimate, not a rigorous total-error guarantee.

## Price estimates at the largest budget

| Estimator | Total points incl. pilot | Price RMSE | Max absolute error | Marginal CI coverage |
|---|---:|---:|---:|---:|
| mc_raw | 1048576 | 0.0320971 | 0.121947 | 0.958 |
| sobol_raw | 1048576 | 0.00124813 | 0.00514059 | 1.000 |
| sobol_asset_conditional | 1048576 | 0.00137379 | 0.00574448 | 1.000 |
| sobol_conditional | 1048576 | 0.000183255 | 0.000687108 | 1.000 |
| sobol_conditional_control | 1056768 | 5.65623e-05 | 0.000254522 | 1.000 |

Pilot control coefficients are fitted from independent replicate means. Sobol uncertainty is estimated across independent scrambles, never across points within a net.
`replicate_means.npz` preserves the independent means used for uncertainty calculations; its price rows follow `price_runs.csv`, and Greek groups and quantity order are stored in the archive.
Gaussian-point counts do not equal CPU work: conditional formulas, geometric controls and Greek bumps have different costs. Recorded timings are exploratory, without a controlled cross-method timing protocol.

## Greek errors at 262,144 production points

| Quantity | Conditional RMSE | Conditional + control RMSE |
|---|---:|---:|
| delta_1 | 8.34678e-07 | 7.74562e-07 |
| delta_2 | 9.27076e-06 | 2.60253e-06 |
| gamma_11 | 2.6504e-08 | 2.07378e-08 |
| gamma_22 | 1.86943e-08 | 3.83682e-09 |
| gamma_12 | 1.82773e-08 | 6.55925e-09 |
| vega_1 | 0.00528418 | 0.00138549 |
| vega_2 | 0.00860141 | 0.00222024 |
| theta | 0.000755104 | 0.000204895 |
| rho | 5.26629e-05 | 6.02693e-05 |
| correlation | 0.00116331 | 0.000353877 |

Delta and the full spot Hessian use analytical conditional derivatives. Vega, Theta, rate Rho and correlation sensitivity use central bumps with common random numbers.
Their intervals exclude bump bias. `reference_checks.csv` records the effect of halving the reference bumps. Greek intervals are marginal, not simultaneous.

## ADI scope

Modified Craig-Sneyd (theta=1/3), central differences and two initial implicit-Euler-damped intervals. The zero-stock faces are exact vanilla prices; the far faces use asymptotic values.
`adi_convergence.csv` separates mesh refinement from a domain extension at constant spatial spacing. Near-expiry Hessians require substantially finer grids than regular cases. This implementation is an independent baseline, not yet a production reference across all parameter ranges.

## Sources

- Bayer, Siebenmorgen and Tempone: https://arxiv.org/abs/1607.05572
- In 't Hout and Foulon (ADI with mixed derivatives): https://arxiv.org/abs/0811.3427
- Glau and Wunderlich: https://doi.org/10.1016/j.amc.2022.127355
