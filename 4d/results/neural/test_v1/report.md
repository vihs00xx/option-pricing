# Four-stock basket: test-set scores

Scored 2026-10-21T03:00:41.241830+00:00 on results/test_set_v1 (231 cases). Score = mean over price and the 26 Greeks of RMSE / RMS(reference); references are converged Gauss-Hermite values. Network seconds are per contract in one batch with every Greek; classical seconds per contract on one process.

| Method | Score | Price RMSE | Seconds per contract |
|---|---:|---:|---:|
| RQMC 16 x 4096 (control) | 0.0004528 | 0.000838 | 5.06e+00 |
| DeepONet D9_ensemble_seed0 | 0.02091 | 0.0382 | 3.32e-03 |
| DeepONet D8_envelope_seed1 | 0.02255 | 0.0467 | 7.45e-04 |
| DeepONet D8_envelope_seed3 | 0.02364 | 0.0419 | 8.90e-04 |
| DeepONet D8_envelope_seed0 | 0.02398 | 0.0494 | 7.46e-04 |
| DeepONet D8_envelope_seed2 | 0.02429 | 0.0428 | 8.50e-04 |
| Monte Carlo 100,000 paths | 0.02603 | 0.0414 | 5.90e-01 |
| DeepONet D6_lbfgs_seed0 | 0.03421 | 0.0675 | 6.48e-04 |
| DeepONet D5_similarity_seed0 | 0.03845 | 0.074 | 8.27e-04 |
| DeepONet D4_physics_seed0 | 0.04266 | 0.0852 | 5.76e-04 |
| DeepONet D3_differential_seed0 | 0.0443 | 0.0856 | 6.20e-04 |
| DeepONet D2_hard_payoff_seed0 | 0.05582 | 0.127 | 5.08e-04 |
| PINN P5_similarity_seed0 | 0.05864 | 0.139 | 6.15e-04 |
| PINN P4_adaptive_seed0 | 0.07703 | 0.202 | 4.39e-04 |
| DeepONet D1_reduced_seed0 | 0.3515 | 0.487 | 5.57e-04 |
| DeepONet D0_vanilla_seed0 | 0.4251 | 0.818 | 3.36e-04 |
| PINN P0_vanilla_seed0 | 0.5049 | 2.4 | 1.70e-04 |
| PINN P1_reduced_seed0 | 0.6326 | 2.13 | 3.48e-04 |

By split (score):

| Method | anchor | extrapolation | interpolation | short_maturity |
|---|---:|---:|---:|---:|
| RQMC 16 x 4096 (control) | 0.0002435 | 0.0007237 | 0.0002957 | 0.0002398 |
| DeepONet D9_ensemble_seed0 | 0.02301 | 0.02956 | 0.01857 | 0.005572 |
| DeepONet D8_envelope_seed1 | 0.01965 | 0.03359 | 0.02044 | 0.005821 |
| DeepONet D8_envelope_seed3 | 0.0228 | 0.03333 | 0.02193 | 0.006502 |
| DeepONet D8_envelope_seed0 | 0.02684 | 0.03335 | 0.02136 | 0.005997 |
| DeepONet D8_envelope_seed2 | 0.02552 | 0.03622 | 0.02051 | 0.006056 |
| Monte Carlo 100,000 paths | 0.01469 | 0.02871 | 0.01882 | 0.04086 |
| DeepONet D6_lbfgs_seed0 | 0.03627 | 0.05024 | 0.02904 | 0.01792 |
| DeepONet D5_similarity_seed0 | 0.03592 | 0.05732 | 0.03351 | 0.0196 |
| DeepONet D4_physics_seed0 | 0.039 | 0.06188 | 0.04041 | 0.02092 |
| DeepONet D3_differential_seed0 | 0.03934 | 0.06625 | 0.04133 | 0.02112 |
| DeepONet D2_hard_payoff_seed0 | 0.04542 | 0.08673 | 0.05176 | 0.02024 |
| PINN P5_similarity_seed0 | 0.04775 | 0.07952 | 0.05794 | 0.02385 |
| PINN P4_adaptive_seed0 | 0.0635 | 0.1072 | 0.07551 | 0.02638 |
| DeepONet D1_reduced_seed0 | 0.3792 | 0.3293 | 0.2228 | 1.498 |
| DeepONet D0_vanilla_seed0 | 0.4062 | 0.4735 | 0.3414 | 1.577 |
| PINN P0_vanilla_seed0 | 0.4643 | 0.5113 | 0.4421 | 1.34 |
| PINN P1_reduced_seed0 | 0.6251 | 0.5363 | 0.5041 | 2.761 |
