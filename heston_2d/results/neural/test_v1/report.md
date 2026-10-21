# 2-stock Heston basket: test-set scores

Scored 2026-10-21T17:33:31.588480+00:00 on results/test_set_v1 (220 cases). Score = mean over price and every Greek of RMSE / RMS(reference); noise floor = the same ratio for the references' own standard errors. Network seconds are per contract in one batch with every Greek; classical seconds per contract on one process.

| Method | Score | Price RMSE | Seconds per contract | Noise floor |
|---|---:|---:|---:|---:|
| QE-RQMC 8 x 1024 | 0.02838 | 0.00781 | 1.59e+00 | 0.0045 |
| QE Monte Carlo 8 x 1024 | 0.08165 | 0.0998 | 1.54e+00 | 0.0045 |
| DeepONet D9_ensemble_seed0 | 0.1121 | 0.11 | 5.52e-04 | 0.0045 |
| DeepONet D8_envelope_seed2 | 0.1261 | 0.128 | 1.28e-04 | 0.0045 |
| DeepONet D8_envelope_seed0 | 0.1282 | 0.112 | 1.41e-04 | 0.0045 |
| DeepONet D10_permutation_seed0 | 0.13 | 0.12 | 1.42e-04 | 0.0045 |
| DeepONet D8_envelope_seed3 | 0.1327 | 0.124 | 1.39e-04 | 0.0045 |
| DeepONet D8_envelope_seed1 | 0.1328 | 0.134 | 1.34e-04 | 0.0045 |
| DeepONet D5_similarity_seed0 | 0.134 | 0.133 | 1.04e-04 | 0.0045 |
| DeepONet D6_lbfgs_seed0 | 0.134 | 0.133 | 1.06e-04 | 0.0045 |
| DeepONet D4_physics_seed0 | 0.1346 | 0.141 | 1.01e-04 | 0.0045 |
| PINN P11_permutation_seed0 | 0.1393 | 0.116 | 1.08e-04 | 0.0045 |
| DeepONet D3_differential_seed0 | 0.1426 | 0.142 | 1.03e-04 | 0.0045 |
| PINN P10_anchored_seed0 | 0.144 | 0.137 | 1.02e-04 | 0.0045 |
| DeepONet D2_hard_payoff_seed0 | 0.1527 | 0.155 | 1.02e-04 | 0.0045 |
| PINN P9_envelope_seed0 | 0.202 | 0.197 | 1.06e-04 | 0.0045 |
| PINN P6_lbfgs_seed0 | 0.2034 | 0.221 | 7.90e-05 | 0.0045 |
| PINN P2_hard_payoff_seed0 | 0.2863 | 0.407 | 7.53e-05 | 0.0045 |
| PINN P4_variance_boundary_seed0 | 0.3019 | 0.72 | 8.15e-05 | 0.0045 |
| PINN P3_time_balanced_seed0 | 0.3161 | 0.508 | 7.81e-05 | 0.0045 |
| PINN P5_adaptive_seed0 | 0.3383 | 0.9 | 7.90e-05 | 0.0045 |
| DeepONet D1_reduced_seed0 | 0.3957 | 0.57 | 6.67e-05 | 0.0045 |
| DeepONet D0_vanilla_seed0 | 0.5403 | 2.59 | 7.09e-05 | 0.0045 |
| PINN P0_vanilla_seed0 | 0.698 | 5.4 | 3.15e-05 | 0.0045 |
| PINN P1_reduced_seed0 | 1.134 | 5.74 | 4.57e-05 | 0.0045 |

By split (score):

| Method | anchor | extrapolation | interpolation | short_maturity |
|---|---:|---:|---:|---:|
| QE-RQMC 8 x 1024 | 0.01981 | 0.05059 | 0.01647 | 0.006266 |
| QE Monte Carlo 8 x 1024 | 0.06036 | 0.1152 | 0.07063 | 0.202 |
| DeepONet D9_ensemble_seed0 | 0.176 | 0.1415 | 0.0939 | 0.2001 |
| DeepONet D8_envelope_seed2 | 0.2383 | 0.1575 | 0.1023 | 0.2134 |
| DeepONet D8_envelope_seed0 | 0.1632 | 0.1587 | 0.1131 | 0.1922 |
| DeepONet D10_permutation_seed0 | 0.1514 | 0.1689 | 0.11 | 0.2107 |
| DeepONet D8_envelope_seed3 | 0.1856 | 0.1737 | 0.1124 | 0.2471 |
| DeepONet D8_envelope_seed1 | 0.1972 | 0.1736 | 0.1097 | 0.2395 |
| DeepONet D5_similarity_seed0 | 0.1491 | 0.1811 | 0.1123 | 0.2808 |
| DeepONet D6_lbfgs_seed0 | 0.1491 | 0.1811 | 0.1123 | 0.2808 |
| DeepONet D4_physics_seed0 | 0.1528 | 0.1747 | 0.1192 | 0.3565 |
| PINN P11_permutation_seed0 | 0.1834 | 0.1681 | 0.1273 | 0.352 |
| DeepONet D3_differential_seed0 | 0.1447 | 0.1979 | 0.1201 | 0.3733 |
| PINN P10_anchored_seed0 | 0.2104 | 0.1728 | 0.129 | 0.3528 |
| DeepONet D2_hard_payoff_seed0 | 0.1508 | 0.2048 | 0.1334 | 0.2935 |
| PINN P9_envelope_seed0 | 0.32 | 0.2343 | 0.1883 | 0.4272 |
| PINN P6_lbfgs_seed0 | 0.2302 | 0.2306 | 0.1954 | 0.3509 |
| PINN P2_hard_payoff_seed0 | 0.3667 | 0.3241 | 0.2748 | 0.4269 |
| PINN P4_variance_boundary_seed0 | 0.3466 | 0.3493 | 0.295 | 0.5501 |
| PINN P3_time_balanced_seed0 | 0.323 | 0.3552 | 0.3122 | 0.4591 |
| PINN P5_adaptive_seed0 | 0.3788 | 0.3794 | 0.341 | 0.5882 |
| DeepONet D1_reduced_seed0 | 0.3752 | 0.4164 | 0.2924 | 5.83 |
| DeepONet D0_vanilla_seed0 | 0.4236 | 0.6526 | 0.528 | 6.032 |
| PINN P0_vanilla_seed0 | 0.6435 | 0.7482 | 0.6954 | 3.952 |
| PINN P1_reduced_seed0 | 1.059 | 1.221 | 1.046 | 13.23 |
