# Selected DeepONet result

Weight 0.01 was selected by the paired three-seed validation study before the final test was opened. The seed-22092027 checkpoint had the lowest validation price RMSE within that family. Its path and SHA-256 hash are frozen in `experiments/config/neural/deeponet_model_registry.json`. Weight 1 remains the Greek-focused reference.

The two newly trained weight-0.01 runs started from exactly the same seed-specific pilot checkpoints as their weight-1 partners. Each received 300 seconds of continuation training. Weight 0.01 had lower price RMSE in all three pairs. Its three-seed mean price RMSE was 0.1614 versus 0.2575 for weight 1. Mean Gamma, Theta, and Rho RMSE stayed within the prespecified 20% tolerance. The weight-0.01 Greek results varied more across seeds, especially Theta; this selection should not be described as uniformly more stable.

After recording the selection, all three registry models were evaluated once on the 68-case held-out benchmark. Values below are RMSE against analytical Black--Scholes prices and Greeks.

| Quantity | Selected weight 0.01 | Weight 1 reference | Price-only reference |
|---|---:|---:|---:|
| Price | 0.27594 | 0.47159 | 0.54458 |
| Delta | 0.01620 | 0.02741 | 0.06100 |
| Gamma | 0.00232 | 0.00488 | 0.01489 |
| Vega | 2.24314 | 3.07879 | 3.53239 |
| Theta | 1.22860 | 1.86317 | 3.23091 |
| Rho | 3.25384 | 4.20391 | 9.58222 |

For the selected model, price RMSE was 0.0933 on 16 anchor cases, 0.1334 on 32 interpolation cases, and 0.4727 on 20 extrapolation cases. The largest extrapolation weaknesses were Vega (RMSE 3.973) and Rho (RMSE 5.687). The full group metrics and maximum errors are in `final_test_metrics.json`.

The 68 cases provide a useful external check within the Black--Scholes model, but the point estimates here are not confidence intervals and do not establish market-price performance. Model and checkpoint selection used validation data only; final-test results did not change that choice.
