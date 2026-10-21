# 2-stock Heston basket: test-set scores

Scored 2026-10-21T03:00:16.201048+00:00 on results/test_set_v1 (220 cases). Score = mean over price and every Greek of RMSE / RMS(reference); noise floor = the same ratio for the references' own standard errors. Network seconds are per contract in one batch with every Greek; classical seconds per contract on one process.

| Method | Score | Price RMSE | Seconds per contract | Noise floor |
|---|---:|---:|---:|---:|
| QE-RQMC 8 x 1024 | 0.02838 | 0.00781 | 1.59e+00 | 0.0045 |
| QE Monte Carlo 8 x 1024 | 0.08165 | 0.0998 | 1.54e+00 | 0.0045 |
| DeepONet D9_ensemble_seed0 | 0.1121 | 0.11 | 2.02e-03 | 0.0045 |
| DeepONet D8_envelope_seed2 | 0.1261 | 0.128 | 5.21e-04 | 0.0045 |
| DeepONet D8_envelope_seed0 | 0.1282 | 0.112 | 6.20e-04 | 0.0045 |
| DeepONet D8_envelope_seed3 | 0.1327 | 0.124 | 4.75e-04 | 0.0045 |
| DeepONet D8_envelope_seed1 | 0.1328 | 0.134 | 5.54e-04 | 0.0045 |
| DeepONet D5_similarity_seed0 | 0.134 | 0.133 | 4.04e-04 | 0.0045 |
| DeepONet D4_physics_seed0 | 0.1346 | 0.141 | 4.37e-04 | 0.0045 |
| DeepONet D3_differential_seed0 | 0.1426 | 0.142 | 4.25e-04 | 0.0045 |
| DeepONet D2_hard_payoff_seed0 | 0.1527 | 0.155 | 4.38e-04 | 0.0045 |
| PINN P2_hard_payoff_seed0 | 0.2863 | 0.407 | 3.20e-04 | 0.0045 |
| PINN P4_variance_boundary_seed0 | 0.3019 | 0.72 | 3.33e-04 | 0.0045 |
| PINN P5_adaptive_seed0 | 0.3383 | 0.9 | 2.82e-04 | 0.0045 |
| DeepONet D1_reduced_seed0 | 0.3957 | 0.57 | 2.57e-04 | 0.0045 |
| DeepONet D0_vanilla_seed0 | 0.5403 | 2.59 | 2.48e-04 | 0.0045 |

By split (score):

| Method | anchor | extrapolation | interpolation | short_maturity |
|---|---:|---:|---:|---:|
| QE-RQMC 8 x 1024 | 0.01981 | 0.05059 | 0.01647 | 0.006266 |
| QE Monte Carlo 8 x 1024 | 0.06036 | 0.1152 | 0.07063 | 0.202 |
| DeepONet D9_ensemble_seed0 | 0.176 | 0.1415 | 0.0939 | 0.2001 |
| DeepONet D8_envelope_seed2 | 0.2383 | 0.1575 | 0.1023 | 0.2134 |
| DeepONet D8_envelope_seed0 | 0.1632 | 0.1587 | 0.1131 | 0.1922 |
| DeepONet D8_envelope_seed3 | 0.1856 | 0.1737 | 0.1124 | 0.2471 |
| DeepONet D8_envelope_seed1 | 0.1972 | 0.1736 | 0.1097 | 0.2395 |
| DeepONet D5_similarity_seed0 | 0.1491 | 0.1811 | 0.1123 | 0.2808 |
| DeepONet D4_physics_seed0 | 0.1528 | 0.1747 | 0.1192 | 0.3565 |
| DeepONet D3_differential_seed0 | 0.1447 | 0.1979 | 0.1201 | 0.3733 |
| DeepONet D2_hard_payoff_seed0 | 0.1508 | 0.2048 | 0.1334 | 0.2935 |
| PINN P2_hard_payoff_seed0 | 0.3667 | 0.3241 | 0.2748 | 0.4269 |
| PINN P4_variance_boundary_seed0 | 0.3466 | 0.3493 | 0.295 | 0.5501 |
| PINN P5_adaptive_seed0 | 0.3788 | 0.3794 | 0.341 | 0.5882 |
| DeepONet D1_reduced_seed0 | 0.3752 | 0.4164 | 0.2924 | 5.83 |
| DeepONet D0_vanilla_seed0 | 0.4236 | 0.6526 | 0.528 | 6.032 |
