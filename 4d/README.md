# Four-stock Black–Scholes baskets

This study prices European arithmetic basket options on four correlated stocks under
Black–Scholes with PINNs and DeepONets. Each model returns the price and all 26 Greeks: 4 Deltas,
10 Gammas, 4 Vegas, theta, rho and the 6 correlation sensitivities. One network covers the whole
contract domain. Its inputs are the moneyness of each stock (0.7 to 1.3), the maturity (0.05 to 2),
the volatilities (0.1 to 0.6), the rate, the dividends, the basket weights (each at least 0.1) and
a correlation matrix whose off-diagonal entries lie in [-0.5, 0.9] with smallest eigenvalue at
least 0.05.

## Reference

The arithmetic basket has no closed form. The reference conditions on one Gaussian factor, which
gives a closed-form Black–Scholes-type integrand, and integrates over the three remaining factors by
tensor Gauss–Hermite quadrature. The node count rises (32, 40, 48, 64, 80) until every one of the
27 quantities changes by less than the tolerance, and the next level must confirm that. Deltas and
Gammas are analytic, and the other Greeks are central common-random-number bumps. The Phase 0
check ([report](results/reference_check_v1/report.md)) shows that quadrature reaches 1e-8 on all
27 quantities within seconds for almost every contract, so it replaces RQMC as the reference at
this dimension.

## Test set

The [sealed test set](results/test_set_v1/report.md) has 231 cases: 128 interpolation,
32 short maturity (tau 0.005 to 0.05), 52 one-parameter extrapolation and 19 anchor regimes.
It was frozen before any network was trained.

## Results

The score is the mean over the price and the 26 Greeks of RMSE / RMS(reference). Networks are timed
per contract in one batch with every Greek. Classical methods are timed on one process.

| Method | Score | Price RMSE | Seconds per contract |
|---|---:|---:|---:|
| RQMC, 16 x 4,096 points with control variate | 0.00045 | 0.00084 | 5.1 |
| DeepONet `D9_ensemble` (four seeds) | 0.0209 | 0.038 | 1.1e-3 |
| Monte Carlo, 100,000 paths | 0.0260 | 0.041 | 0.59 |
| PINN `P11_permutation` | 0.0320 | 0.064 | 2.6e-4 |
| PINN `P10_anchored` | 0.0337 | 0.062 | 2.7e-4 |
| DeepONet `D0_vanilla` | 0.425 | 0.82 | 1.2e-4 |
| PINN `P0_vanilla` | 0.505 | 2.40 | 1.1e-4 |

- **Ablations.** The ladders bring the error down by a factor of 15 to 20 compared with the textbook
  networks (`P0`, `D0`). The largest single gain comes from the hard payoff constraint on a
  closed-form base solution (`P2`, `D2`). Differential training, similarity coordinates, L-BFGS,
  the payoff envelope and ensembling each add a smaller improvement.
- **Against simulation.** The DeepONet ensemble is more accurate than Monte Carlo with 100,000
  paths and about 500 times faster per contract. RQMC with a control variate is still about
  50 times more accurate, but it costs 5 s per contract.
- **By split.** Short maturities are the easiest split for the finished networks (score 0.006 for
  `D9`), which is consistent with the hard payoff constraint being exact at expiry. Extrapolation is the
  hardest (0.030).

The full table, with every level and the scores by split, is in
[results/neural/test_v1/report.md](results/neural/test_v1/report.md).

## Reproduce

Run from this folder:

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m experiments.verify results/test_set_v1
python -m experiments.neural.run_ladders --threads 8
python -m experiments.evaluate_test
```

`run_ladders` trains every level of both ladders in order. An interrupted level resumes from its
last check.

## Layout

- `src/pricing/`: engine for any number of stocks. It holds the conditional estimator, the
  geometric-basket closed form, RQMC and the two-stock ADI solver.
- `src/neural/`: coordinates, networks, PDE residuals, the quadrature reference, training
  (Adam, L-BFGS and Nyström Newton-CG) and Monte Carlo.
- `experiments/reference/`: the Phase 0 convergence, RQMC and cost studies.
- `experiments/test_set/`: the test-set build. `experiments/neural/`: training, ensembles and the
  ladder runner. `experiments/evaluate_test.py`: test-set scoring.
- `tests/`: pricing, the root-solver fix, neural models and the two-stock reproduction.
