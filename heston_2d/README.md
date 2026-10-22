# Two-stock Heston baskets

This study prices European arithmetic basket options on two stocks, each with its own Heston
variance, with PINNs and DeepONets. Each model returns the price and all 19 Greeks: Deltas, Gammas,
theta, rate rho, vega (d/dv0), the sensitivities to each stock's kappa, theta, xi and leverage, and
the sensitivities to both correlations. The study combines the reduced coordinates, Levy base and
ablation ladders of the four-stock study (`../4d`) with the QE simulation and variance boundary of
the one-stock Heston study (`../heston`). `../heston_4d` runs the same code with a four-stock
protocol, because the protocol alone sets the number of stocks.

## Model

Stock i has dS_i / S_i = (r - q_i) dt + sqrt(v_i) dW_i and dv_i = kappa_i (theta_i - v_i) dt +
xi_i sqrt(v_i) dZ_i. The variance drivers Z have correlation C (`vol_correlation`); W_i = rho_i Z_i
+ sqrt(1 - rho_i^2) Y_i with Y independent of Z and correlation P, so the stock correlation is
R = (rho rho') o C + (rbar rbar') o P. Contracts give R and C; every (W, Z) correlation is valid by
construction, and one stock is the Heston model. The domain draws C and P, so R is always valid.

## References

There is no closed form or quadrature for a Heston basket. The engine (`src/heston_basket/`)
simulates only the variances, by Andersen's QE scheme with martingale correction; given the
variance paths the log returns are Gaussian, and the basket option is the four-stock conditional
estimator (one factor in closed form, the residual factors sampled). Spot Greeks are analytic per
path; the other sensitivities are common-random-number bumps. The tests check it against the
one-stock Lewis reference in `../heston` and against the Black-Scholes basket as xi -> 0. The
sealed test set uses replicated Sobol RQMC at QE step 1/64 with a standard error for every value,
an independent second run, a doubled-step bias check and a bump-halving check.

## Test set

The [sealed test set](results/test_set_v1/report.md) has 220 cases: 128 interpolation, 32 short
maturity (tau 0.005 to 0.05), 46 one-parameter extrapolation and 14 anchor regimes. The Feller
condition is not imposed. Each case's points per replicate rise from 4,096 to 16,384 until the price
standard error is at most 5e-4 (strike 100). Cases still above that target are flagged, not replaced.
An independent second run gives RMS z = 1.04 over all 4,400 values.

## Results

The score is the mean over the price and every Greek of RMSE / RMS(reference). The noise floor is
the same ratio for the references' own standard errors (0.0045). Below it, no method can be told
apart from the reference.

| Method | Score | Price RMSE | Seconds per contract |
|---|---:|---:|---:|
| QE-RQMC, 8 x 1,024 Sobol paths | 0.028 | 0.0078 | 1.6 |
| QE Monte Carlo, 8 x 1,024 paths | 0.082 | 0.100 | 1.5 |
| DeepONet `D9_ensemble` (four seeds) | 0.112 | 0.110 | 5.5e-4 |
| PINN `P11_permutation` | 0.139 | 0.116 | 1.1e-4 |
| PINN `P10_anchored` | 0.144 | 0.137 | 1.0e-4 |
| DeepONet `D0_vanilla` | 0.540 | 2.59 | 7.1e-5 |
| PINN `P0_vanilla` | 0.698 | 5.40 | 3.2e-5 |

- **Ablations.** The ladders reduce the score by a factor of about 5 compared with the textbook
  networks. As with four Black–Scholes stocks, the hard payoff constraint gives the largest single
  gain. For the PINN, L-BFGS refinement (`P6`) and anchoring (`P10`) also give large gains.
- **Against simulation.** The networks are about 3,000 to 15,000 times faster per contract than
  the simulation baselines. Their price error is similar to that of QE Monte Carlo with 8,192 paths,
  but their Greek errors are larger, so their overall score is worse.
- **By split.** Interpolation is the best split for the networks (0.094 for `D9`). Short maturities
  and anchors are the hardest (0.20 and 0.18).

The full table, with every level and the scores by split, is in
[results/neural/test_v1/report.md](results/neural/test_v1/report.md).

## Reproduce

Run from this folder:

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m experiments.reference.reference_check --workers 8     # Phase 0, development contracts only
python -m experiments.reference.reference_check --report        # results/reference_check_v1/report.md
python -m experiments.test_set.freeze_test_set --workers 8      # seals results/test_set_v1
python -m experiments.verify results/test_set_v1
python -m experiments.neural.run_ladders --threads 8            # every ladder level, one at a time
python -m experiments.evaluate_test
```

An interrupted level resumes from its last check.

## Layout

- `src/heston_basket/`: the model (`model.py`) and the conditional QE engine (`engine.py`).
- `src/neural/`: coordinates and domain sampling, networks (with sqrt(v + 0.01) inputs), the Heston
  basket PDE residuals, simulation labels and autograd Greeks, training with the variance boundary.
- `src/pricing/`: the Black-Scholes basket engine of `../4d`, used by the tests.
- `experiments/config/`: the test-set protocol (which also sets the number of stocks), the ladders
  and the study settings. `experiments/reference/`, `experiments/test_set/` and
  `experiments/neural/` hold the Phase 0 check, the test-set build and the training scripts.
- `tests/`: engine, labels, coordinates, PDE, models and short training runs.
