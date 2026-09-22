# Option Pricing

A computational study comparing analytical Black--Scholes, Monte Carlo,
randomized quasi-Monte Carlo, conditional smoothing, Crank--Nicolson and neural
methods for European option prices and Greeks.

## Project structure

```text
foundations/                 Learning exercises and introductory simulations
src/                         Reusable pricing and neural implementations
src/neural/                  PINN and DeepONet support code
experiments/                 Runnable studies, grouped by method
experiments/config/classical Classical and shared benchmark configurations
experiments/config/neural/   PINN and DeepONet configurations and registry
experiments/config/legacy/   Retained superseded configuration
results/                     Generated tables and figures, grouped by method
results/pinn/primary/        Frozen reportable PINN result
results/pinn/research_variants/ PINN ablations and optimizer experiments
results/pinn/legacy/         Superseded or incomplete-provenance PINN studies
checkpoints/pinn/primary/    Frozen anchored PINN checkpoint
checkpoints/pinn/research_variants/ Experimental PINN checkpoints
checkpoints/pinn/legacy/     Diagnostic and superseded checkpoints
report/                      LaTeX project report
tests/                       Automated tests
artifacts/logs/              Local ignored run logs and diagnostics
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
.\.venv\Scripts\python.exe -m experiments.finite_difference.run_crank_nicolson
.\.venv\Scripts\python.exe -m experiments.finite_difference.check_crank_nicolson
.\.venv\Scripts\python.exe -m experiments.common_benchmark.prepare_cases
.\.venv\Scripts\python.exe -m experiments.deeponet.prepare_data
```

The full experiments can take substantial time because they include repeated runs and large sample budgets. To run the unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Analytical Black--Scholes baseline

`src/black_scholes.py` provides vectorized European call and put prices with continuous dividend yield, plus Delta, Gamma, Vega, Theta, and Rho. Inputs use years and decimal annual rates. Vega and Rho are per unit decimal change; Theta is calendar-time change per year. The implementation covers expiry, zero spot, and zero volatility for prices; Greeks require positive spot, maturity, and volatility.

The analytical experiment uses its own configuration at `experiments/config/classical/analytical_settings.json` and writes the 54-case reference grid to `results/analytical/tables/`.

## Monte Carlo methods

`src/monte_carlo.py` implements plain pseudorandom Monte Carlo and variance-reduced price estimators. `src/sampling.py` provides pseudorandom and scrambled Sobol sampling, including control-variate support. `src/monte_carlo_greeks.py` estimates price and the five standard Greeks using hybrid pathwise/likelihood-ratio, likelihood-ratio, or common-random-number finite-difference estimators.

The experiments compare plain, antithetic, control-variate, Sobol, and Sobol-control approaches. Convergence tests run from $2^{10}$ to $2^{20}$ payoff samples across representative contracts. Recorded pooled empirical convergence exponents are 0.509 for plain Monte Carlo, 0.916 for Sobol, and 1.154 for Sobol-control in that low-dimensional study. These are measured results for the selected cases, not general convergence guarantees.

The tuning and held-out evaluation studies record error, interval coverage, runtime, and memory. Some selected Greek configurations have under-calibrated confidence intervals; their results remain in the tables rather than being replaced after evaluation. See `results/monte_carlo/tables/mc_validation_status.json` and `mc_greek_interval_checks.csv` alongside the error summaries.

## Conditional smoothing

The conditional estimator simulates part of the option life and analytically values the continuation period. This smooths the payoff contribution and can reduce variance in price and Greek estimates. In this one-dimensional study it uses the Black--Scholes formula for the continuation value, so it is a variance-reduction experiment rather than an independent pricing method. The method is relevant to future basket-option work, where integrating out one Gaussian factor can reduce the simulated dimension.

The frozen conditional study reports convergence through $2^{20}$, tests a configuration on held-out cases, and compares an efficient run with a higher-compute run using 64 Sobol scrambles. Results and validation files are under `results/conditional/`.

## Crank--Nicolson finite differences

`src/crank_nicolson.py` solves the Black--Scholes PDE with Crank--Nicolson time
stepping, Rannacher startup damping, a strike-clustered mesh and Richardson
extrapolation. The validation study compares prices and Greeks with analytical
Black--Scholes values and checks convergence, shape constraints and far-field
boundary sensitivity. Tables and figures are under `results/finite_difference/`.

Crank--Nicolson and the PINN study are implemented. The next active stage is the
DeepONet operator-learning baseline.

## Frozen benchmark and model status

The common benchmark is frozen as `common_option_benchmark_v1`. It contains 68
held-out call and put cases covering standard, difficult, interpolation and
extrapolation settings, with price and five Greeks, repeated timing and memory
measurements. Its settings are in
`experiments/config/classical/common_benchmark_settings.json`.

The anchored PINN is frozen as the primary one-dimensional PINN result. Other
PINN configurations are retained as research variants or legacy diagnostics and
are excluded from main result tables according to
`experiments/config/neural/pinn_model_registry.json`. Because the anchored model accepts
only one fixed call contract, direct comparisons with it use the fixed-surface
profile; the broader variable-parameter call/put benchmark is used for DeepONet
and eligible classical methods.

The DeepONet problem and analytical dataset are frozen before model training in
`experiments/config/neural/deeponet_settings.json`. Contract and market parameters form
the branch input; spot and time to maturity form the trunk input. Black--Scholes
provides price and Greek labels. Frozen benchmark contracts are used only for
testing, while real NSE/BSE observations are reserved for a later external case
study.

## Neural methods

The anchored PINN is the frozen primary one-dimensional neural baseline. Its
checkpoint, validation results and diagnostics are under
`results/pinn/primary/` and `checkpoints/pinn/primary/`.

Other PINN configurations are retained for possible future research, but they
are exploratory and are not part of the main comparison. Their registry and
archived results are listed in
`experiments/config/neural/pinn_model_registry.json`.

The next neural experiment is a supervised DeepONet. Its branch input contains
contract and market parameters (K, T, r, q, sigma and option type), while its
trunk input contains spot and time to maturity (S and tau). The frozen analytical dataset and its
manifest are under `results/deeponet/data/`. The 68 common-benchmark cases are
reserved for final testing and are never used for training or model selection.
