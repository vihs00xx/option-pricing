# Neural Pricing of Multi-Asset Options and Their Greeks

This repository studies whether physics-informed neural networks (PINNs) and deep operator
networks (DeepONets) can price European options and estimate every Greek accurately enough
to compete with classical methods. It also measures how this changes as the number of underlying
stocks grows. The study starts from one stock under Black–Scholes and extends to baskets of
two, four and eight correlated stocks, and then to stochastic volatility (Heston) with one,
two and four stocks.

## Methodology

- **Networks.** Each network is a single model of the whole contract domain: moneyness, maturity,
  rate, dividends, volatilities or Heston parameters, correlations and basket weights are all
  inputs. Greeks are obtained by automatic differentiation of the network's price.
- **Ablation ladders.** PINNs (levels `P0` to `P11`) and DeepONets (`D0` to `D10`) are built up one
  technique at a time: reduced coordinates, a hard payoff constraint, a closed-form base solution,
  similarity coordinates, adaptive sampling, L-BFGS and Newton-CG refinement, a payoff envelope,
  anchoring, permutation symmetry and seed ensembles. Each level is scored, so the contribution of
  every technique is measured.
- **References.** Every study uses the most accurate classical reference that its dimension allows.
  These are closed forms in 1D, adaptive quadrature for two-stock baskets and tensor Gauss–Hermite
  quadrature for four stocks. Eight stocks and every Heston basket use randomized quasi-Monte Carlo
  (RQMC) with a standard error for each value. The classical baselines are Monte Carlo, RQMC,
  Crank–Nicolson or ADI grids, and COS for one-stock Heston.
- **Sealed test sets.** Each study freezes its test set before any training. A test set has four
  splits: interpolation, short maturity, one-parameter extrapolation and hand-chosen anchor
  regimes. Model selection uses separate validation contracts only. Every saved output is
  pinned by SHA-256 so that results can be verified without retraining.
- **Score.** The score is the mean over the price and all Greeks of RMSE / RMS(reference). Lower is
  better. Timings are per contract with every Greek.

## Results

Best run of each family on each sealed test set:

| Study | Model, stocks | Greeks | Best classical | Best PINN | Best DeepONet |
|---|---|---:|---|---|---|
| [`1d/`](1d/README.md) | Black–Scholes, 1 | all | Analytical (exact) | See the [paper](1d/report/main.pdf) | See the [paper](1d/report/main.pdf) |
| [`2d/`](2d/README.md) | Black–Scholes, 2 | 10 | ADI, price RMSE 6.0e-5 | `P10`, price RMSE 0.049 | `D9`, price RMSE 0.032 |
| [`4d/`](4d/README.md) | Black–Scholes, 4 | 26 | RQMC, score 0.00045 | `P11`, score 0.032 | `D9`, score 0.021 |
| [`8d/`](8d/README.md) | Black–Scholes, 8 | 82 | Not yet run | Not yet run | Not yet run |
| [`heston/`](heston/README.md) | Heston, 1 | 11 | COS, score 0.0019 | `P10`, score 0.144 | `D9`, score 0.073 |
| [`heston_2d/`](heston_2d/README.md) | Heston, 2 | all | QE-RQMC, score 0.028 | `P11`, score 0.139 | `D9`, score 0.112 |
| [`heston_4d/`](heston_4d/README.md) | Heston, 4 | all | Training in progress | Training in progress | Training in progress |

The trained networks price a contract with every Greek in 0.02 to 1 ms. That is three to four
orders of magnitude faster than the simulation references. In every finished basket and Heston
study, the DeepONet ensemble is the most accurate network. On four Black–Scholes stocks it is
more accurate than Monte Carlo with 100,000 paths, which takes about 0.6 s per contract. In low
dimension,
where quadrature, grids or COS are available, the classical methods remain more accurate,
so the networks pay back their training cost only when many contracts are priced. Each study's
README and its `results/*/report.md` files give the full ladders, the scores by split and the
known limitations.

## Layout

Each study is a self-contained folder with its own `src/`, `experiments/`, `tests/` and
`results/`. Run each one from its own folder, because saved evidence is pinned to paths relative
to that folder. Tag `1d-v1.0` marks the 1D study at its original repository-root paths.

## Verify

```bash
cd 1d
python -m experiments.comparison.verify_milestone
python -m pytest tests -q
cd ../2d
python -m experiments.basket.verify_results
python -m pytest tests -q
```

Verification reads the saved outputs and does not retrain any model. Every study uses a
top-level package named `src`, so run each test suite from its own folder rather than from the
repository root.
