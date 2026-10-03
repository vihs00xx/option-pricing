# d-stock pricing engine validation

Development cases only: the eight observed two_asset_v1 contracts and four one-, three- and four-stock contracts. No frozen test-set case is used.

## Agreement with the independent two-asset quadrature

The new engine integrates its conditional values with 128-node Gauss-Hermite; the reference is the adaptive quadrature of `src/basket`, which uses a different factor decomposition.

| Quantity | Max abs difference (equal) | Max abs difference (volatility) | Reference bump change |
|---|---:|---:|---:|
| price | 3.38e-14 | 4.62e-14 | 0.00e+00 |
| delta_1 | 1.22e-15 | 1.05e-15 | 0.00e+00 |
| delta_2 | 1.17e-15 | 1.11e-15 | 0.00e+00 |
| gamma_11 | 6.94e-18 | 1.39e-17 | 0.00e+00 |
| gamma_12 | 1.39e-17 | 2.78e-17 | 0.00e+00 |
| gamma_22 | 2.08e-17 | 2.08e-17 | 0.00e+00 |
| vega_1 | 7.31e-11 | 1.93e-10 | 6.73e-07 |
| vega_2 | 2.37e-10 | 4.97e-10 | 3.63e-07 |
| theta | 1.70e-09 | 2.27e-09 | 7.23e-07 |
| rho | 4.26e-11 | 8.23e-11 | 1.01e-06 |
| correlation_12 | 1.34e-10 | 1.66e-10 | 3.55e-08 |

## Price estimates, 64 replicates of 16,384 points

| Method | Price RMSE | Max abs error | RMS z | Coverage | Median seconds |
|---|---:|---:|---:|---:|---:|
| equal_control | 5.832e-05 | 2.156e-04 | 1.07 | 0.875 | 2.56 |
| volatility | 7.703e-05 | 3.242e-04 | 0.80 | 0.958 | 2.91 |
| volatility_control | 8.757e-05 | 4.050e-04 | 1.36 | 0.750 | 3.46 |
| two_asset_v1_conditional_control | 1.183e-04 | 5.196e-04 | 1.56 | 0.792 | 1.12 |
| equal | 1.219e-04 | 3.840e-04 | 0.78 | 0.917 | 2.17 |
| two_asset_v1_conditional | 2.905e-04 | 1.268e-03 | 1.37 | 0.750 | 0.73 |

RMS z is the root-mean-square of error divided by the reported standard error; about one indicates calibrated uncertainty. Each row has 24 runs, but all cases share the same three seeds, so the runs are strongly correlated and their coverage is not a calibration test; see the calibration section below.

## Uncertainty calibration over independent seeds

| Case | Variant | Quantities | Runs | RMS z | Outside 95% interval |
|---|---|---|---:|---:|---:|
| unequal | equal | price | 200 | 1.12 | 0.080 |
| unequal | equal_control | price | 200 | 1.02 | 0.055 |
| unequal | volatility | price | 200 | 1.08 | 0.065 |
| unequal | volatility_control | price | 200 | 1.03 | 0.040 |
| three_stock_mixed | equal | all 18 | 100 | 0.97 | 0.038 |
| three_stock_mixed | equal_control | all 18 | 100 | 1.03 | 0.040 |
| three_stock_mixed | volatility | all 18 | 100 | 1.06 | 0.052 |
| three_stock_mixed | volatility_control | all 18 | 100 | 1.09 | 0.066 |

Every seed is distinct. Three-stock references are the finest Gauss-Hermite values. Intervals use Student-t critical values over independent replicates.

## Sensitivities, 64 replicates of 4,096 points

| Quantity | equal | equal_control | volatility | volatility_control | RMS z (all) |
|---|---:|---:|---:|---:|---:|
| price | 9.48e-04 | 2.57e-04 | 3.45e-04 | 1.12e-04 | 1.24 |
| delta_1 | 7.41e-07 | 5.47e-07 | 1.06e-06 | 5.93e-07 | 0.77 |
| delta_2 | 9.79e-06 | 2.85e-06 | 4.39e-06 | 1.29e-06 | 1.22 |
| gamma_11 | 2.55e-08 | 2.49e-08 | 2.38e-08 | 2.56e-08 | 0.80 |
| gamma_12 | 1.23e-08 | 7.14e-09 | 8.71e-09 | 7.92e-09 | 0.90 |
| gamma_22 | 1.38e-08 | 5.03e-09 | 6.53e-09 | 5.02e-09 | 0.96 |
| vega_1 | 5.67e-03 | 2.34e-03 | 4.22e-04 | 4.36e-04 | 1.13 |
| vega_2 | 9.07e-03 | 3.44e-03 | 1.84e-03 | 6.04e-04 | 1.48 |
| theta | 7.79e-04 | 2.50e-04 | 2.62e-04 | 8.20e-05 | 1.28 |
| rho | 3.18e-05 | 6.00e-05 | 5.80e-05 | 3.77e-05 | 0.77 |
| correlation_12 | 1.16e-03 | 2.42e-04 | 9.46e-04 | 2.40e-04 | 1.15 |

Deltas and the spot Hessian are exact conditional derivatives; vega, theta, rho and correlation sensitivities are central bumps with common draws, and their intervals exclude bump bias.

## One, three and four stocks

| Case | d | Nodes | Evaluations | Max change from finest | Max Black-Scholes error | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| one_stock_put | 1 | 16 | 1 | 0.0e+00 | 3.3e-07 | 0.01 |
| one_stock_put | 1 | 24 | 1 | 0.0e+00 | 3.3e-07 | 0.01 |
| one_stock_put | 1 | 32 | 1 | 0.0e+00 | 3.3e-07 | 0.00 |
| one_stock_put | 1 | 40 | 1 | 0.0e+00 | 3.3e-07 | 0.01 |
| three_stock_mixed | 3 | 16 | 256 | 9.4e-11 | | 0.03 |
| three_stock_mixed | 3 | 24 | 576 | 1.5e-10 | | 0.04 |
| three_stock_mixed | 3 | 32 | 1,024 | 2.7e-10 | | 0.05 |
| three_stock_mixed | 3 | 40 | 1,600 | 0.0e+00 | | 0.07 |
| four_stock_mixed | 4 | 16 | 4,096 | 3.3e-11 | | 0.21 |
| four_stock_mixed | 4 | 24 | 13,824 | 7.8e-11 | | 0.77 |
| four_stock_mixed | 4 | 32 | 32,768 | 5.8e-11 | | 3.62 |
| four_stock_mixed | 4 | 40 | 64,000 | 0.0e+00 | | 5.11 |
| four_stock_equicorrelated_put | 4 | 16 | 4,096 | 3.9e-11 | | 0.11 |
| four_stock_equicorrelated_put | 4 | 24 | 13,824 | 3.4e-11 | | 0.45 |
| four_stock_equicorrelated_put | 4 | 32 | 32,768 | 2.9e-11 | | 1.36 |
| four_stock_equicorrelated_put | 4 | 40 | 64,000 | 0.0e+00 | | 2.53 |

The one-stock Black-Scholes error of vega, theta and rho is bump bias of the central differences.

| Case | Variant | Price RMSE vs finest Gauss-Hermite | RMS z (all quantities) | Median seconds |
|---|---|---:|---:|---:|
| three_stock_mixed | equal | 3.13e-04 | 1.75 | 2.68 |
| three_stock_mixed | equal_control | 2.51e-04 | 1.75 | 7.14 |
| three_stock_mixed | volatility | 2.43e-04 | 1.69 | 3.95 |
| three_stock_mixed | volatility_control | 2.48e-04 | 1.58 | 5.91 |
| four_stock_mixed | equal | 7.69e-04 | 1.00 | 4.32 |
| four_stock_mixed | equal_control | 3.77e-04 | 1.06 | 9.26 |
| four_stock_mixed | volatility | 5.70e-04 | 0.96 | 8.49 |
| four_stock_mixed | volatility_control | 2.10e-04 | 0.96 | 9.93 |
| four_stock_equicorrelated_put | equal | 1.58e-05 | 0.99 | 3.78 |
| four_stock_equicorrelated_put | equal_control | 1.58e-05 | 1.05 | 6.08 |
| four_stock_equicorrelated_put | volatility | 1.58e-05 | 1.00 | 3.81 |
| four_stock_equicorrelated_put | volatility_control | 1.58e-05 | 1.00 | 6.20 |

## Two-stock ADI

Sinh-clustered grids with the spot on a node, a far boundary scaled with volatility and maturity, and steps equal to intervals. Richardson combines the grid with its nested doubling.

| Case | Intervals | Price error | Richardson price error | two_asset_v1 ADI price error | Max spot-Greek/theta rel. error (Richardson) | Seconds (Richardson) |
|---|---:|---:|---:|---:|---:|---:|
| standard | 40 | -9.26e-03 | 2.86e-04 | -7.20e-02 | 1.4e-04 | 0.33 |
| standard | 80 | -2.10e-03 | -8.30e-05 | -1.78e-02 | 1.0e-04 | 1.95 |
| standard | 160 | -5.35e-04 | 1.74e-06 | -4.43e-03 | 3.3e-06 | 20.26 |
| unequal | 40 | -8.52e-03 | 1.35e-04 | | 4.4e-04 | 0.32 |
| unequal | 80 | -1.94e-03 | -7.85e-06 | | 1.5e-04 | 3.56 |
| unequal | 160 | -4.77e-04 | 2.12e-06 | | 2.1e-05 | 19.56 |
| negative_correlation | 40 | -1.33e-02 | 2.35e-03 | 4.99e-02 | 1.7e-03 | 0.32 |
| negative_correlation | 80 | -3.11e-03 | 1.96e-04 | 1.87e-02 | 1.3e-04 | 1.95 |
| negative_correlation | 160 | -6.59e-04 | 3.00e-05 | 5.39e-03 | 9.9e-06 | 13.79 |
| near_expiry | 40 | -2.99e-03 | 4.73e-05 | -5.23e-01 | 1.2e-03 | 0.31 |
| near_expiry | 80 | -8.14e-04 | 1.02e-05 | -1.73e-01 | 1.9e-04 | 1.42 |
| near_expiry | 160 | -1.76e-04 | -7.62e-06 | -3.53e-02 | 9.9e-05 | 12.98 |
| high_volatility | 40 | -3.95e-02 | -2.13e-04 | | 2.7e-04 | 0.48 |
| high_volatility | 80 | -8.93e-03 | -9.45e-05 | | 2.1e-05 | 1.64 |
| high_volatility | 160 | -2.26e-03 | -6.79e-07 | | 2.0e-06 | 14.34 |
| unequal_put | 40 | -8.52e-03 | 1.35e-04 | | 4.4e-04 | 0.84 |
| unequal_put | 80 | -1.94e-03 | -7.85e-06 | | 1.5e-04 | 1.57 |
| unequal_put | 160 | -4.77e-04 | 2.12e-06 | | 2.1e-05 | 18.05 |

Parameter sensitivities (Richardson, 80 intervals, bumped re-solves on a fixed grid):

| Case | vega_1 | vega_2 | rho | correlation_12 |
|---|---:|---:|---:|---:|
| standard | -3.7e-04 | -1.9e-04 | 6.6e-05 | 2.3e-04 |
| unequal | -1.5e-04 | 9.4e-06 | 5.0e-05 | 2.2e-05 |
| negative_correlation | 1.3e-04 | -1.2e-03 | 9.1e-05 | -6.5e-04 |
| near_expiry | 1.2e-04 | 3.0e-05 | -2.6e-08 | -4.7e-05 |
| high_volatility | -1.4e-04 | 8.1e-05 | 1.6e-04 | 1.6e-04 |
| unequal_put | -1.5e-04 | 9.4e-06 | 5.0e-05 | 2.2e-05 |

Errors are signed differences from the quadrature reference. Puts are the call solution plus the exact parity term, so a put shares its call's errors (unequal and unequal_put). Grid solves are a two-stock baseline; their cost grows exponentially with the number of stocks.
