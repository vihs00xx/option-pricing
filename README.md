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

## PINN diagnostics and shape-constraint study

`src/neural/fixed_pinn.py` holds the fixed-parameter Black--Scholes PINN. Its first full-budget training failed the acceptance gates: core price RMSE 0.044 (target 0.02), core Theta RMSE 0.256 (target 0.1), and widespread no-arbitrage violations. `src/neural/pinn_diagnostics.py` extends the violation counts into per-region proportions, maximum sizes, and mean sizes, separating near-expiry, near-strike, low-tail, high-tail, and core regions, and evaluates Greeks only for tau > 0. On the failed checkpoint, 47.8% of grid points priced below the discounted-intrinsic floor (maximum 0.203), 42.3% had negative Gamma, and violations concentrate near expiry and in the tails. Run it with:

```powershell
python -m experiments.pinn.run_pinn_diagnostics --checkpoint checkpoints/pinn/primary/anchored/fixed_black_scholes_pinn.pt --name anchored_diagnostics
```

`experiments/pinn/train_pinn_variant.py` trains three variants under one shared, frozen budget in `experiments/config/neural/pinn_study_settings.json`, all using the same architecture so only the training method differs:

- `baseline_matched`: the original physics-only recipe at the reduced matched budget.
- `constrained`: adds one-sided penalties for the price bounds, Delta bounds, and the log-moneyness convexity condition u_yy - u_y >= 0. Category weights are rebalanced by gradient-norm annealing, points are weighted by their violation size, and some constraint batches are drawn where violations were recently largest (Hoshisashi/Phelan/Barucca-style derivative constraints; Wang et al. gradient annealing).
- `causal`: physics-only training with causal bin weights over time-to-maturity, starting from the terminal payoff (Wang et al. causal training).

`src/neural/nys_newton_cg.py` implements NysNewton-CG (Rathore et al.) with exact damped Hessian-vector products by double backward, because Torch 2.8 cannot run forward-mode jvp through the second-order residual graph. `experiments/pinn/refine_pinn_optimizer.py` compares it with L-BFGS from the same saved Adam checkpoint under the same wall-clock budget. `experiments/pinn/evaluate_pinn_study.py` evaluates every study checkpoint on the shared grid. Constraint penalties at sampled points encourage but do not guarantee the constraints between points.

## Multi-seed study, revised variants, and the anchored comparison

The matched-budget study is run over three seeds (tag `s0`/`s1`/`s2`) with one shared global seed, so the three variants share initial weights and sampled batches and only the loss terms differ. Every variant trains through the same Adam plus L-BFGS schedule: `VariantTrainer.train` runs Adam, `VariantTrainer.refine_lbfgs` then polishes the *variant's own* objective on a fixed full batch (constraint and causal terms are preserved), with a wall-clock cap. The Adam checkpoint is saved as `study_{name}_adam.pt` before L-BFGS, so L-BFGS and NysNewton-CG refine from exactly the same state.

```powershell
# full study (3 seeds x 3 variants, then refinements, then evaluation)
powershell -ExecutionPolicy Bypass -File experiments/pinn/run_study.ps1
# revised constrained + causal variants only, same budget
powershell -ExecutionPolicy Bypass -File experiments/pinn/run_study_revised.ps1
# evaluate every checkpoint (original + revised + anchored) and aggregate
python -m experiments.pinn.evaluate_pinn_study_multi
```

`evaluate_pinn_study_multi.py` discovers all `study_{variant}_{tag}.pt` checkpoints, the refined optimiser checkpoints, and the anchored `fixed_black_scholes_pinn.pt`, then writes `run_summary.csv` (one row per model: core price/Delta/Gamma/Theta RMSE, maximum errors, worst violation proportion and magnitude, wall time, peak RSS) and `seed_aggregate.csv` (mean +/- standard deviation per variant and revision, with the anchored and refined models as single-seed references).

Three corrections were made after inspecting the first study:

- **Causal effective-bin diagnostic.** The reported "effective bins" used `1 / sum(w^2)`, which gives `1/B` for uniform weights over `B` bins. It is now the participation ratio `(sum w)^2 / sum(w^2)`, which gives `B` for uniform weights and `1` for a single dominating bin.
- **Causal concentration.** The causal weights now use `exp(-epsilon * cumulative / total)`, so `epsilon` is scale-free instead of depending on the raw residual magnitude (which drifts over training). With 20 bins this runs from about 19 effective bins at `epsilon = 1` to about 3.5 at `epsilon = 20`.
- **Constraint balancing.** Category weights are balanced against the *mean of the constraint losses* (not the PDE loss) and clipped to `[1, 10]` instead of `[0.01, 100]`. The floor stops Delta and Gamma collapsing toward zero; the tighter ceiling stops the price bounds, whose violation loss is tiny next to the Delta violation loss, from being amplified to the old ceiling of ~98 and dominating the constraint budget.

With the pinned-to-earth Adam plateau removed by the L-BFGS stage, all three physics-only variants still fall well short of the anchored model, which additionally uses analytical Black--Scholes price and Theta labels. See `results/pinn/research_variants/study/tables/seed_aggregate.csv`.

## Anchored PINN v2 ablation

`experiments/pinn/run_anchored_study.ps1` runs a controlled anchored study that asks whether the PDE term improves accuracy or label efficiency beyond ordinary supervised interpolation.

**One fixed dataset** (`src/neural/anchor_data.py`) is generated once and retained at `results/pinn/legacy/anchored_ablation_v2/data/anchored_dataset.npz`, so every model sees identical points. It keeps *separate* splits for training price anchors, training Greek anchors (independently sampled near the strike and near expiry, so they are not a subset of the price anchors), training PDE collocation, training terminal payoff, validation coordinates and labels, validation PDE collocation, validation terminal payoff, and untouched test coordinates. Subsets for the efficiency sweep are **stratified prefixes** of a component-interleaved ordering, so they are nested and approximately proportional across density components as the anchor count grows. Sampling is dtype-explicit (never the process-wide default), densities must sum to one, near-expiry spans cannot exceed maturity, and the two boundary strips are sampled symmetrically. Each dataset carries a schema version, a data-configuration hash and a content hash; reuse is refused when the configuration does not match, and smoke datasets use a separate path.

**Four models** (`src/neural/anchored_pinn.py`) share one architecture, one initialisation seed per run and one budget, and differ only in their loss terms:

| Model | Loss terms |
|---|---|
| `data_only` | price, Theta (no PDE residual) |
| `physics_only` | PDE residual, terminal payoff (no labels) |
| `hybrid` | price, Theta, PDE, terminal |
| `v2` | hybrid plus sparse Delta and Gamma |

Supervised terms are normalised by RMS label scales computed from the *full* training label sets, so the anchor-count sweep does not silently change the objective. Weight handling is explicit: exactly zero **disables** a term, positive weights are clipped into `[weight_min, weight_max]`, and negative or non-finite weights raise a configuration error. A stage's active terms are the intersection of the variant's permitted terms and its positive stage weights; a stage with no active terms is skipped rather than optimising an all-zero objective, and disabled terms incur no derivative work. Staged training runs a price/payoff warm-up, a PDE balance, Greek supervision, then an L-BFGS polish.

Checkpoint selection uses an **immutable per-variant validation objective** declared in the settings (`selection`), evaluated with the same normalisation scales for every candidate, so scores from different stages are comparable. Selection weights are fixed before training; stage losses are logged separately because they are not comparable across stages. The initial model, every stage end and the final model are scored, and both the **selected** and the **last trained** checkpoints are saved. `physics_only` selects on held-out PDE residual and terminal error, so it remains label-free; `data_only` and `hybrid` share the same supervised selection criterion (price + Theta) so that isolating the PDE term is meaningful. Time budgets are enforced inside the L-BFGS loop as well as Adam, checked between completed steps, so an interrupted budget always leaves an accepted parameter state.

**Evaluation** (`experiments/pinn/evaluate_anchored_ablation.py`) reports **two separate metric sets**: the historical 161 x 51 core grid, for comparison with the originally published numbers, and the broad test mixture, which includes harder regions. The reference checkpoint is evaluated with exactly the same evaluator on both. Financial violations are reported **per category** (price lower/upper bound, Delta lower/upper bound, negative Gamma) with tolerance, proportion, maximum and mean magnitude, and the region where the maximum occurs; magnitudes in different units are never combined, and the only unified statistic is the proportion of points violating at least one constraint, named explicitly. Regional errors, maximum absolute errors, the PDE residual (with its normalisation stated), training time, peak memory, and price-only and price+Greeks inference timing (median and IQR after warm-up) are all recorded. Aggregation uses mean and **sample** standard deviation per (model, anchors, checkpoint kind), and the evaluator writes a check confirming that the aggregate table reproduces the per-run rows. Superseded V2 results are retained under `results/pinn/legacy/anchored_ablation_v2/`; any future versioned ablation belongs under `results/pinn/research_variants/`.

Acceptance is decided in **three separate sections**: absolute core-grid thresholds (the published targets, applied *only* to core-grid metrics), improvement over the reference on the broad test mixture, and financial-constraint criteria. A core-grid threshold is never applied to a broad-test error. Per-seed pass/fail is kept alongside the aggregate summary.

The controls `supervised_hybrid_control` (price/Theta + payoff, no PDE) and `supervised_v2_control` (price/Theta/Delta/Gamma + payoff, no PDE) share their labels and selection policy with `hybrid` and `v2`, so a claim about the PDE term can be isolated from the payoff term and from the added Greek labels.

The earlier study under `results/pinn/legacy/anchored_ablation_v1/` is retained but superseded: its training controller and reporting had the defects listed above, so its runs must not be mixed into corrected aggregates. The historical anchored reference was trained with a longer, different recipe and is treated as an accuracy reference, not a compute-matched baseline.

Acceptance for Anchored PINN v2 requires beating the reference: price RMSE < 0.0098 while also targetting Delta RMSE < 0.00247, Gamma RMSE < 0.000872, Theta RMSE < 0.0874, and reducing the worst violation proportion (0.478) and magnitude (0.203).
