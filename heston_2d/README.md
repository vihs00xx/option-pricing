# Two-stock Heston baskets: code for the lab PC

PINN and DeepONet pricers for European arithmetic basket options on two stocks, each with its own
Heston variance, with the price and every Greek: Deltas, Gammas, theta, rate rho, vega (d/dv0)
and the sensitivities to every stock's kappa, theta, xi and leverage and to both correlations.
It combines the four-stock study (`../4d`: reduced coordinates, Levy base, ladders, scripts) with
the one-stock Heston study (`../heston`: QE simulation, variance boundary). `../heston_4d` is the
same code with a four-stock protocol. Nothing has been run at full size yet: no test set is sealed
and no network is trained.

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

## Run order on the lab PC (from this folder)

```bash
pip install -r requirements.txt
python -m pytest tests -q                                       # about 1 minute
python -m experiments.reference.reference_check --workers 8     # Phase 0, development contracts only
python -m experiments.reference.reference_check --report        # results/reference_check_v1/report.md
python -m experiments.test_set.freeze_test_set --workers 8      # seals results/test_set_v1
python -m experiments.verify results/test_set_v1
python -m experiments.neural.run_ladders --threads 8            # every ladder level, one at a time
```

Read the Phase 0 report before sealing the test set: the time-step table should show the QE bias at
1/64 below the price standard-error target, and the points ladder how long each case takes. Before
training, check that the evaluation references' noise floor (recorded in every level's summary) is
well below the networks' errors; raise `reference.evaluation` in `experiments/config/neural/study.json`
if not.

## Expected cost (measured on the laptop, 4 threads; scale for the lab PC)

| Step | Cost |
|---|---|
| Test-set reference | about 100 s per run of 16 x 16,384 paths with every Greek; a case climbs to at most 65,536 points and adds three check runs, so about 10 to 30 minutes per case and process |
| Training labels | 0.017 s per contract: about 9 minutes for 32,768 contracts |
| Monitor and validation references | 2.4 s per contract: about 40 minutes for 1,024 contracts on one process, spread over every core |
| PINN Adam step (P5) | 0.8 s: about 3 hours per 15,000-step level |
| Differential DeepONet step (D3) / physics-informed (D4) | 0.13 s / 0.6 s: 0.5 / 2.7 hours per level |

## Layout

- `src/heston_basket/`: the model (`model.py`) and the conditional QE engine (`engine.py`).
- `src/neural/`: coordinates and domain sampling, networks (with sqrt(v + 0.01) inputs), the Heston
  basket PDE residuals, simulation labels and autograd Greeks, training with the variance boundary.
- `src/pricing/`: the Black-Scholes basket engine of `../4d`, used by the tests.
- `experiments/config/`: the test-set protocol (which also sets the number of stocks), the ladders
  and the study settings. `experiments/reference/`, `experiments/test_set/` and
  `experiments/neural/` hold the Phase 0 check, the test-set build and the training scripts.
- `tests/`: engine, labels, coordinates, PDE, models and short training runs.
