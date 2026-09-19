# Option Pricing

An undergraduate project comparing numerical methods for European option pricing and Greeks. The current GitHub baseline contains analytical Black--Scholes, Monte Carlo, randomized quasi-Monte Carlo, and conditional smoothing experiments. Finite-difference and neural methods are planned extensions.

## Project structure

```text
src/                         Pricing implementations
experiments/analytical/      Black--Scholes reference-grid experiment
experiments/monte_carlo/     Monte Carlo and convergence experiments
experiments/conditional/     Conditional smoothing studies
experiments/config/          Experiment configurations
results/analytical/          Analytical reference outputs
results/monte_carlo/         Monte Carlo tables, figures, and archive
results/conditional/         Conditional estimator tables and figures
tests/                       Unit tests
```

## Setup

From the project folder, install the listed dependencies and run the focused experiments:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m experiments.analytical.run_black_scholes
.\.venv\Scripts\python.exe -m experiments.monte_carlo.run_monte_carlo
.\.venv\Scripts\python.exe -m experiments.monte_carlo.run_strong_monte_carlo
.\.venv\Scripts\python.exe -m experiments.monte_carlo.run_mc_convergence
.\.venv\Scripts\python.exe -m experiments.conditional.run_conditional_frozen_study
.\.venv\Scripts\python.exe -m experiments.conditional.check_conditional_study
.\.venv\Scripts\python.exe -m experiments.conditional.run_conditional_high_compute
.\.venv\Scripts\python.exe -m experiments.conditional.check_conditional_high_compute
```

The full experiments can take substantial time because they include repeated runs and large sample budgets. To run the unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Analytical Black--Scholes baseline

`src/black_scholes.py` provides vectorized European call and put prices with continuous dividend yield, plus Delta, Gamma, Vega, Theta, and Rho. Inputs use years and decimal annual rates. Vega and Rho are per unit decimal change; Theta is calendar-time change per year. The implementation covers expiry, zero spot, and zero volatility for prices; Greeks require positive spot, maturity, and volatility.

The analytical experiment uses its own configuration at `experiments/config/analytical_settings.json` and writes the 54-case reference grid to `results/analytical/tables/`.

## Monte Carlo methods

`src/monte_carlo.py` implements plain pseudorandom Monte Carlo and variance-reduced price estimators. `src/sampling.py` provides pseudorandom and scrambled Sobol sampling, including control-variate support. `src/monte_carlo_greeks.py` estimates price and the five standard Greeks using hybrid pathwise/likelihood-ratio, likelihood-ratio, or common-random-number finite-difference estimators.

The experiments compare plain, antithetic, control-variate, Sobol, and Sobol-control approaches. Convergence tests run from $2^{10}$ to $2^{20}$ payoff samples across representative contracts. Recorded pooled empirical convergence exponents are 0.509 for plain Monte Carlo, 0.916 for Sobol, and 1.154 for Sobol-control in that low-dimensional study. These are measured results for the selected cases, not general convergence guarantees.

The tuning and held-out evaluation studies record error, interval coverage, runtime, and memory. Some selected Greek configurations have under-calibrated confidence intervals; their results remain in the tables rather than being replaced after evaluation. See `results/monte_carlo/tables/mc_validation_status.json` and `mc_greek_interval_checks.csv` alongside the error summaries.

## Conditional smoothing

The conditional estimator simulates part of the option life and analytically values the continuation period. This smooths the payoff contribution and can reduce variance in price and Greek estimates. In this one-dimensional study it uses the Black--Scholes formula for the continuation value, so it is a variance-reduction experiment rather than an independent pricing method. The method is relevant to future basket-option work, where integrating out one Gaussian factor can reduce the simulated dimension.

The frozen conditional study reports convergence through $2^{20}$, tests a configuration on held-out cases, and compares an efficient run with a higher-compute run using 64 Sobol scrambles. Results and validation files are under `results/conditional/`.

The project will next add and compare Crank--Nicolson finite differences, PINNs, and DeepONets against these classical baselines.
