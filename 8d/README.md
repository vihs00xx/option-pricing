# Eight-stock Black–Scholes baskets

PINN and DeepONet pricers for European arithmetic basket options on eight correlated stocks under
Black-Scholes, with price and all 82 Greeks (8 Deltas, 36 Gammas, 8 Vegas, theta, rho and 28
correlation sensitivities). It follows the four-stock study in `../4d` step for step: the same
general-d pricing engine and neural code, the same ladder levels and the same scripts.

**Status.** The code and tests are complete. The test set has not been sealed yet, and no network
has been trained.

## What changes from four stocks

- **References are RQMC, not quadrature.** The four-stock reference integrates the 3 residual
  factors of the conditional estimator by tensor Gauss-Hermite. Eight stocks have 7 residual
  factors, and 32^7 points is not feasible. The sealed test set uses the engine's replicated
  RQMC with the geometric control (`rqmc_estimate`), climbing 16,384, 65,536 and 262,144 points
  per replicate until the price standard error is at most 1e-4 (strike 100), and stores every
  quantity's standard error next to its reference. Training labels and the monitor and validation
  references use the torch twin with one fixed scrambled Sobol net and a geometric control
  (`src/neural/reference.py`): 4,096 points for labels and 32,768 for evaluation.
- **Correlations come from a three-factor model.** Uniform entries with rejection, the four-stock
  domain, accept about 1 matrix in 20,000 for eight stocks. The eight-stock domain draws each
  stock's loadings on a market factor (0.2 to 0.9) and two style factors (-0.5 to 0.5), with
  communality at most 0.95, so every matrix is valid with smallest eigenvalue at least 0.05. The
  numpy sampler (`coordinates.factor_loadings`) and the torch collocation sampler
  (`training._correlations`) draw the same distribution.
- **Weights** have a floor of 0.04 (0.1 for four stocks would leave only 0.2 to share).
- The test set has the same 231-case shape (128 interpolation, 32 short maturity, 52
  extrapolation, 19 anchors); its seed is 20261017. Cases above the error target are flagged,
  not replaced.

## Reproduce (from this folder)

```bash
pip install -r requirements.txt
python -m pytest tests -q                                   # about 4 minutes
python -m experiments.reference.rule_check --workers 8      # Phase 0, development contracts only
python -m experiments.reference.make_report                 # results/reference_check_v1/report.md
python -m experiments.test_set.freeze_test_set --workers 8  # seals results/test_set_v1
python -m experiments.verify results/test_set_v1
python -m experiments.neural.run_ladders --threads 8        # every ladder level, one at a time
```

Read the Phase 0 report before training: if the evaluation rule's error is not well below the
networks' validation errors, raise `reference_rule.evaluation.points` in
`experiments/config/neural/study.json` (the data caches are keyed by their settings, so a change
rebuilds them). A single level runs with `python -m experiments.neural.train P2_hard_payoff
--threads 8`; an interrupted level resumes from its last check.

## Computational cost (measured with 4 CPU threads)

| Step | Cost |
|---|---|
| Test-set reference | about 20 minutes per case and process at 65,536 points per replicate with its three cross-check runs; about 9 hours for 231 cases on 8 processes |
| Training labels | 0.05 s per contract at 4,096 points: about 25 minutes for 32,768 contracts |
| Validation reference | about 0.4 s per contract at 32,768 points: about 25 minutes for 4,096 contracts |
| PINN Adam step (4,096 collocation points) | about 2.2 s: about 9 hours per 15,000-step level |
| Differential DeepONet step (1,024 contracts) | about 0.9 s: about 3.6 hours per level |

The full ladders are several days of CPU time, dominated by the PINN levels and their L-BFGS
stages. `run_ladders --only` selects which levels to train, for example the DeepONet ladder alone.

## Layout

- `src/pricing/`: the general-d engine of `../4d` (conditional estimator, RQMC, geometric
  closed form). `src/basket/` is the two-stock code its tests compare against.
- `src/neural/`: coordinates and the factor-model sampler, networks, PDE residuals, the torch
  reference (Gauss-Hermite or Sobol rule), training, Monte Carlo.
- `experiments/config/`: the test-set protocol, the ladders and the study settings.
- `experiments/reference/`: Phase 0 development contracts, rule check and report.
- `experiments/test_set/`: the test-set build. `experiments/neural/`: training, ensembles and
  `run_ladders`.
- `tests/`: the eight-stock reference and models, plus the exact Gauss-Hermite checks with four
  stocks and the two-stock reproduction against `../2d`.
