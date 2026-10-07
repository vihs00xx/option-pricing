# Phase 1: classical benchmark on heston_test_v1

COS, ADI, QE Monte Carlo and QE-RQMC at every setting on the 252 sealed test cases (128 interpolation, 32 short maturity, 72 extrapolation, 20 anchors), price and 11 sensitivities per unit strike, against the Lewis reference. Settings were fixed in advance on the Phase 0 development contracts and were not tuned on the test cases. Seconds are per contract with all 12 quantities, one thread, nothing else running, measured on 12 test cases spread evenly over maturity (`timing.json`; the benchmark workers' own seconds shared the CPU with the neural training).

Score = mean over the 12 quantities of RMSE / RMS(reference), all cases.

| Method | Setting | Score | Price RMSE | Delta RMSE | Gamma RMSE | Vega RMSE | Volga RMSE | Theta RMSE | d_xi RMSE | Seconds per contract |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COS (c2 rule, L = 12) | `cos_32` | 0.081 | 0.00041 | 0.013 | 0.15 | 0.011 | 0.27 | 0.00043 | 0.00092 | 0.003 |
| COS (c2 rule, L = 12) | `cos_64` | 0.04 | 7.9e-05 | 0.0035 | 0.18 | 0.0021 | 0.1 | 6.8e-05 | 0.00015 | 0.0031 |
| COS (c2 rule, L = 12) | `cos_128` | 0.012 | 2.2e-05 | 0.00084 | 0.041 | 0.00094 | 0.046 | 2.9e-05 | 9.7e-05 | 0.0035 |
| COS (c2 rule, L = 12) | `cos_256` | 0.0019 | 7.9e-06 | 9.3e-05 | 0.0095 | 0.0004 | 0.0035 | 3.6e-05 | 3.3e-05 | 0.0037 |
| COS (c2 rule, L = 12) | `cos_512` | 0.00082 | 8e-06 | 1.6e-05 | 0.00036 | 0.00038 | 0.0025 | 3.6e-05 | 3.5e-05 | 0.0048 |
| COS (c2 rule, L = 12) | `cos_1024` | 0.00081 | 8e-06 | 1.6e-05 | 4.9e-05 | 0.00038 | 0.0024 | 3.6e-05 | 3.5e-05 | 0.0052 |
| ADI (Hundsdorfer-Verwer) | `adi_40x20x20` | 0.1 | 0.00043 | 0.0063 | 0.3 | 0.01 | 1.2 | 0.0012 | 0.0005 | 0.12 |
| ADI (Hundsdorfer-Verwer) | `adi_80x40x40` | 0.023 | 0.00011 | 0.0021 | 0.061 | 0.0018 | 0.15 | 0.00036 | 0.00018 | 0.37 |
| ADI (Hundsdorfer-Verwer) | `adi_160x80x80` | 0.0073 | 2.8e-05 | 0.00061 | 0.018 | 0.00056 | 0.055 | 0.00011 | 5.9e-05 | 1.8 |
| QE Monte Carlo | `mc_10000_dt8` | 0.69 | 0.003 | 0.0061 | 0.097 | 0.0078 | 13 | 0.0055 | 0.0022 | 0.097 |
| QE Monte Carlo | `mc_100000_dt8` | 0.28 | 0.001 | 0.0022 | 0.077 | 0.0029 | 5.9 | 0.0014 | 0.00079 | 1.2 |
| QE Monte Carlo | `mc_1000000_dt8` | 0.063 | 0.00035 | 0.0012 | 0.071 | 0.002 | 0.99 | 0.00081 | 0.00034 | 13 |
| QE Monte Carlo | `mc_10000_dt32` | 0.76 | 0.0033 | 0.0065 | 0.082 | 0.0079 | 15 | 0.0035 | 0.0024 | 0.39 |
| QE Monte Carlo | `mc_100000_dt32` | 0.19 | 0.00099 | 0.0019 | 0.032 | 0.0023 | 3.7 | 0.0015 | 0.00075 | 5.3 |
| QE Monte Carlo | `mc_1000000_dt32` | 0.11 | 0.00029 | 0.00063 | 0.023 | 0.0013 | 2.3 | 0.00051 | 0.0003 | 58 |
| QE-RQMC (Sobol, bridge, 16 scrambles) | `rqmc_2^8x16_dt32` | 1.4 | 0.0017 | 0.0034 | 0.12 | 0.0094 | 29 | 0.0024 | 0.0036 | 0.64 |
| QE-RQMC (Sobol, bridge, 16 scrambles) | `rqmc_2^10x16_dt32` | 0.43 | 0.00046 | 0.0015 | 0.059 | 0.0041 | 8.8 | 0.00072 | 0.0014 | 1.1 |
| QE-RQMC (Sobol, bridge, 16 scrambles) | `rqmc_2^12x16_dt32` | 0.34 | 0.00019 | 0.00068 | 0.033 | 0.004 | 7.4 | 0.00035 | 0.00054 | 3 |
| QE-RQMC (Sobol, bridge, 16 scrambles) | `rqmc_2^14x16_dt32` | 0.1 | 6.1e-05 | 0.00032 | 0.025 | 0.0011 | 2.2 | 0.00018 | 0.00029 | 11 |
| QE-RQMC (Sobol, bridge, 16 scrambles) | `rqmc_2^16x16_dt32` | 0.074 | 3.8e-05 | 0.00019 | 0.022 | 0.00057 | 1.6 | 0.00013 | 0.00012 | 64 |

COS on the whole test set at once (vectorised): cos_32 87 us, cos_64 157 us, cos_128 417 us, cos_256 970 us, cos_512 2496 us, cos_1024 5769 us per contract.
 The NumPy Lewis reference with all 12 quantities takes 0.0039 s per contract.

## By split (score)

| Setting | interpolation | short_maturity | extrapolation | anchor |
|---|---:|---:|---:|---:|
| `cos_32` | 0.06 | 0.0084 | 0.038 | 0.17 |
| `cos_64` | 0.013 | 0.00058 | 0.009 | 0.1 |
| `cos_128` | 0.00096 | 1.1e-06 | 0.0013 | 0.03 |
| `cos_256` | 1.9e-05 | 8.7e-08 | 4.6e-05 | 0.0048 |
| `cos_512` | 1.7e-05 | 8.7e-08 | 2.4e-05 | 0.0019 |
| `cos_1024` | 1.7e-05 | 8.7e-08 | 2.4e-05 | 0.0019 |
| `adi_40x20x20` | 0.051 | 0.036 | 0.066 | 0.16 |
| `adi_80x40x40` | 0.02 | 0.0087 | 0.037 | 0.036 |
| `adi_160x80x80` | 0.0074 | 0.0017 | 0.016 | 0.0068 |
| `mc_10000_dt8` | 0.82 | 0.42 | 1 | 0.64 |
| `mc_100000_dt8` | 0.33 | 0.065 | 0.16 | 0.29 |
| `mc_1000000_dt8` | 0.17 | 0.026 | 0.13 | 0.058 |
| `mc_10000_dt32` | 2.4 | 0.18 | 0.51 | 0.7 |
| `mc_100000_dt32` | 0.37 | 0.061 | 0.14 | 0.18 |
| `mc_1000000_dt32` | 0.069 | 0.017 | 0.074 | 0.11 |
| `rqmc_2^8x16_dt32` | 2.7 | 0.22 | 0.56 | 1.3 |
| `rqmc_2^10x16_dt32` | 0.96 | 0.11 | 0.4 | 0.42 |
| `rqmc_2^12x16_dt32` | 0.89 | 0.04 | 0.16 | 0.34 |
| `rqmc_2^14x16_dt32` | 0.2 | 0.052 | 0.1 | 0.1 |
| `rqmc_2^16x16_dt32` | 0.19 | 0.014 | 0.026 | 0.077 |

## Monte Carlo and RQMC error bars

| Setting | median abs z | share inside 1.96 SE | RMS z (outlier-dominated) | values with SE = 0 (of which error > 1e-10) |
|---|---:|---:|---:|---:|
| `mc_10000_dt8` | 0.69 | 0.93 | 3.5e+06 | 24 (23) |
| `mc_100000_dt8` | 0.8 | 0.88 | 1.1e+07 | 25 (23) |
| `mc_1000000_dt8` | 1.1 | 0.73 | 3.5e+07 | 22 (21) |
| `mc_10000_dt32` | 0.7 | 0.93 | 3.5e+06 | 36 (35) |
| `mc_100000_dt32` | 0.68 | 0.94 | 1.1e+07 | 25 (23) |
| `mc_1000000_dt32` | 0.73 | 0.93 | 3.5e+07 | 13 (12) |
| `rqmc_2^8x16_dt32` | 0.7 | 0.91 | 1.8e+06 | 26 (24) |
| `rqmc_2^10x16_dt32` | 0.73 | 0.9 | 4.2e+06 | 34 (32) |
| `rqmc_2^12x16_dt32` | 0.8 | 0.87 | 1.1e+07 | 32 (30) |
| `rqmc_2^14x16_dt32` | 0.82 | 0.83 | 1.5e+07 | 13 (11) |
| `rqmc_2^16x16_dt32` | 0.98 | 0.75 | 4e+07 | 13 (11) |

RQMC caveat: the standard error over 16 scrambles is unreliable when a sensitivity comes from rare paths. On short maturities deep in or out of the money (for example Gamma of a deep in-the-money call with tau = 0.006, or Vanna of a deep in-the-money put), few or no scrambles hit the paths that cross the strike, so the estimate and its SE are both near zero while the reference is not, and z runs to hundreds or more. Plain Monte Carlo has the same SE = 0 cases at small path counts, and they disappear with more paths. With 2e5 paths it reproduces the RQMC outliers within its error bars. The RMS z column is dominated by these cases; the median abs z and the share inside 1.96 are not, and neither are the RMSEs.

## Time-step bias (QE, 10^6 paths)

| Quantity | dt 1/8 vs reference: median abs z / inside 1.96 | dt 1/32 vs reference | dt 1/8 minus dt 1/32 | RMSE dt 1/8 | RMSE dt 1/32 |
|---|---:|---:|---:|---:|---:|
| price | 0.89 / 0.81 | 0.72 / 0.97 | 0.81 / 0.88 | 0.00035 | 0.00029 |
| delta | 0.93 / 0.79 | 0.74 / 0.92 | 0.81 / 0.88 | 0.0012 | 0.00063 |
| gamma | 0.83 / 0.86 | 0.69 / 0.91 | 0.67 / 0.92 | 0.071 | 0.023 |
| vega | 1.5 / 0.6 | 0.68 / 0.95 | 1.2 / 0.72 | 0.002 | 0.0013 |
| vanna | 0.79 / 0.88 | 0.74 / 0.94 | 0.77 / 0.91 | 0.045 | 0.023 |
| volga | 0.78 / 0.87 | 0.66 / 0.95 | 0.81 / 0.9 | 0.99 | 2.3 |
| theta_cal | 1.4 / 0.66 | 0.76 / 0.95 | 1 / 0.77 | 0.00081 | 0.00051 |
| rho_rate | 1.1 / 0.73 | 0.71 / 0.93 | 0.9 / 0.83 | 0.00094 | 0.0005 |
| d_kappa | 2.3 / 0.46 | 0.9 / 0.83 | 1.4 / 0.62 | 0.00013 | 9.8e-05 |
| d_theta | 1.3 / 0.65 | 0.75 / 0.91 | 1 / 0.76 | 0.0024 | 0.003 |
| d_xi | 1 / 0.77 | 0.72 / 0.93 | 0.85 / 0.85 | 0.00034 | 0.0003 |
| d_rho | 1.3 / 0.71 | 0.75 / 0.92 | 0.97 / 0.75 | 0.00095 | 0.0005 |

Without bias, the median abs z is about 0.67 and 95% lie inside 1.96. Larger values are bias: the time step, and for the bumped sensitivities the bump. The 'dt 1/8 minus dt 1/32' column compares the two step sizes directly, so it isolates the time-step part. Cases with SE = 0 are left out (counted above).

## Settings

- COS: 32, 64, 128, 256, 512, 1024 terms; truncation [c1 +- 12 sqrt(c2)] with exact cumulants, frozen on the development contracts (Phase 0); Delta, Gamma, Vega, Vanna, Volga from the expansion, rate rho by the chain rule, Theta from the PDE identity, kappa/theta/xi/rho by central bumps with [a, b] fixed.
- ADI: Hundsdorfer-Verwer, theta = 1/2 + sqrt(3)/6, two Douglas (theta = 1) half-step pairs first; (S intervals, v intervals, time steps) = (40, 20, 20), (80, 40, 40), (160, 80, 80), chosen on development contracts; S grid sinh-clustered at the strike with the spot on a node; v grid sinh-clustered at 0 up to v_max = 15.28 (Phase 0) with v0 on a node; second-order upwinding of the v-convection above v = 1; V = 0 at S = 0, V_SS = 0 at S_max, the degenerate PDE at v = 0, V_v = 0 at v_max. Grid Greeks; rate rho and parameter sensitivities by central bumps re-solved on the same grid; Theta from the PDE.
- QE Monte Carlo (Andersen 2008, psi_c = 1.5, martingale correction): 10^4, 10^5, 10^6 paths at dt = 1/8 and 1/32 (at least 2 steps). Pathwise Delta and rate rho; Gamma from pathwise Deltas at S +- 1%; Vega, Vanna, Volga, Theta and the parameter sensitivities by central bumps (2%; 0.02 for rho) with common random numbers.
- QE-RQMC: scrambled Sobol, Brownian bridge for both Brownian motions, 16 scrambles of 2^8, 2^10, 2^12, 2^14, 2^16 points, dt = 1/32; same estimators; standard errors over scrambles.

Files: `benchmark.csv`, `benchmark_by_split.csv`, `values.npz` (every value and SE per case), `time_step_bias.csv`, `timing.json`, `settings.json`.
