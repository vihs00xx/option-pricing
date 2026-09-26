# Option Pricing

A reproducible study of European option prices and Greeks using analytical
Black–Scholes, conditional randomized quasi-Monte Carlo, Crank–Nicolson,
an anchored PINN and a parity-preserving DeepONet.

## Main finding

On the fresh shared cases, the classical numerical methods are more accurate.
The trained networks provide fast inference relative to the tested independent
numerical solves, but retain near-expiry Greek errors and financial violations.
An overall computational advantage including training, labels and model
selection has **not** been demonstrated. Analytical Black–Scholes remains
faster and exact under the model assumptions.

## Results and report

- [Shared comparison](results/shared_comparison/v1/report.md): 512 predefined
  queries, regional price/Greek errors, financial checks and repeated timing.
- [Frozen protocol](results/shared_comparison/v1/protocol.md): all five methods
  on fixed-parameter calls; the varying-contract call/put panel excludes the PINN.
- [Project report](report/main.pdf), [LaTeX source](report/main.tex) and
  [Overleaf bundle](report/overleaf_bundle.zip).
- [Milestone record](results/milestone_1d/README.md): integrity checks,
  environment and archive recovery information.

These test cases are now observed. No training or checkpoint selection used
their results. The shared comparison uses the frozen anchored PINN and refined
parity seed 22092027, selected using earlier validation. The older registered
DeepONet primary remains available for its historical evaluation.

## Workspace

| Folder | Purpose |
|---|---|
| `src/` | Pricing methods and reusable neural implementations |
| `experiments/` | Training, evaluation and their configurations |
| `experiments/comparison/` | Frozen shared benchmark and integrity verifier |
| `tests/` | Regression tests |
| `results/` | Saved evidence, datasets, selected models and provenance |
| `checkpoints/pinn/primary/` | Frozen anchored PINN |
| `report/` | Current paper, referenced figures and tables |

Historical evidence cited by the report and dependencies used by the final
training pipeline are retained. Superseded drafts, previews, legacy runs and
unselected last-iteration checkpoints are archived outside the repository;
the milestone record lists their original paths and hashes. Local `docs/`
remains excluded from Git.

## Reproduce

The recorded run used Windows, Python 3.13.3 and CPU PyTorch 2.8.0.
Create an isolated environment; do not change global packages:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-benchmark.txt
.\.venv\Scripts\python.exe -m experiments.comparison.verify_milestone
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m experiments.comparison.summarize_shared
```

The final checkpoints and raw shared predictions are included. Verification
and summary regeneration do not retrain models or reopen a new test set.
See the comparison report for evaluation/timing commands. Completed outputs
are reused; timing workers require the recorded runtime versions. Runtime
measurements are machine-dependent. Vega and Rho use unit decimal changes;
Theta is calendar-time change per year.
