# Frozen fresh-comparison protocol

Frozen at: 2026-09-26T11:24:00.043762+00:00

## Model selection

The anchored PINN retains its registered checkpoint hash. The primary refined parity checkpoint is parity_22092027, selected from previously chosen refinement checkpoints by existing validation price RMSE before generating any predictions on these cases. The other two parity seeds are secondary checks. No retraining or selection on new labels is permitted.

## Cases

- Fixed panel: 256 calls at K=100, T=1, r=0.05, q=0, sigma=0.2. All five methods report price, Delta, Gamma and Theta. There are 128 common-domain points, 64 tails outside DeepONet's moneyness training range but inside the PINN core, and 64 short-maturity points. The region-specific rows are essential to interpretation.
- Varying panel: 256 queries in 128 paired call/put states. The PINN is excluded. Other methods report price and all five Greeks. There are 128 interpolation, 32 short-maturity and 96 extrapolation queries. Maturity extrapolation changes both T and tau; it is not a single raw-feature extrapolation.
- Seeded Latin hypercube construction and every raw case are frozen in cases.json/cases.csv. Cases are checked against available earlier DeepONet NPZ financial queries, ignoring redundant T and rounding to 10 decimal places.

## Numerical settings

Conditional RQMC uses 64 scrambles, each with 4,096 production and 2,048 pilot points, conditional fraction 0.25. CN uses strike-clustered grids, 800 base space/time intervals, a doubled grid for Richardson, and two full Rannacher steps. The historical settings object is retained to identify these solver settings; its old benchmark metadata does not describe the new test's creation date or cases.

## Evaluation

RMSE, MAE, bias, worst absolute errors, relative errors above quantity-specific floors, and constraint frequency/severity are reported. Separate rows cover regions, option type, moneyness and maturity. The shape tolerance is 1e-5 in each quantity's units. No interval-coverage claim follows from one RQMC estimate per case.

## Timing

Each panel/method uses a fresh CPU process, float64, and two configured library threads. Batches of 1, 8 and 32 have one warm-up and five repetitions; median and interquartile spread are recorded. Loading is excluded; query conversion and computation are included. Fixed-panel CN omits unused bump solves. Conditional RQMC internally computes six outputs. Independent-query timings do not measure cached CN surface reuse. RSS is sampled at 1 ms, including runtime overhead; it is not an exact isolated model-memory measure.

## Integrity and interpretation

protocol.sha256 identifies protocol.json; cases, checkpoints, and implementation sources are hash-pinned. The run refuses changed inputs or implementations. Accuracy outputs are sealed on completion; no frozen model files are modified. The new cases become observed after this run. Analytical BS is the reference, so its error is zero by definition. Results are not an accuracy-matched speed contest or an end-to-end training-cost claim.
