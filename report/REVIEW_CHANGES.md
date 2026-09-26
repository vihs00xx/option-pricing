# Report corrections, 25 September 2026

The revised `main.tex` and `main.pdf` preserve the saved experiment results.
No model was retrained and no new held-out evaluation was performed.

- Corrected the likelihood-ratio derivative identity, hybrid Gamma notation,
  nonuniform-mesh accuracy, clustered-mesh nesting, far-field domain rule,
  logarithmic-coordinate explanation, strike scaling, and put Delta bounds.
- Distinguished validation used for selection from fresh evaluation. Corrected
  optimizer selections to Adam 2 / L-BFGS 4 across six architecture-seed pairs.
- Rebuilt the final-stage PINN loss plot using weighted components and checked
  that they sum to every recorded total. Theta contributes 76.2% of the final
  objective; the price contribution is 0.6%.
- Corrected price and Greek labels, added a Gamma inset, and described negative
  Gamma away from the peak. Changed the classical heatmap to compare RMSEs
  within each quantity rather than colouring unlike units on one scale.
- Distinguished the uniform-grid convergence series from the clustered-mesh
  Richardson series, joint space-time refinement from spatial refinement, and
  fitted median orders from guarantees on individual cases.
- Reconciled the separately recorded PINN training durations and specified the
  different timing workloads. Removed the unsupported break-even calculation.
- Identified the direct-payoff hybrid estimator behind the poor coverage
  figures, keeping that evidence separate from conditional RQMC validation.
- Clarified control-versus-parity shape frequencies, added stress violation
  severity and tolerances, and acknowledged PINN violations inside the core.
- Corrected the spectral-bias citation and narrowed unsupported literature and
  causal claims. Added a reproducibility appendix with recorded settings.

Validation: classical benchmark RMSEs were recomputed from saved errors;
selected validation metrics and optimizer counts were checked against the
selection records; shape severity, loss weights, Gamma locations, and interval
coverage were checked against saved outputs. The LaTeX build and rendered pages
were inspected. Detailed arithmetic is recorded in
`build/report_review_data_audit.json`.

The original PDF, LaTeX source, and figure script are preserved under
`build/report-review-original/`.

To rebuild this revision, run `python report/make_report_figures.py`, then run
`pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice from `report/`.
`main.tex` is the source for this revision. The older
`assemble_current_report.py`, `main_v2` artifacts, and existing Overleaf archive
were not regenerated and should not be used to overwrite this revision.


## Explanatory revision, 26 September 2026

- Rewrote the abstract and introduction without experimental numbers. Added a plain-language introduction to options, hedging, and the five Greeks.
- Expanded the neural explanation: network updates, the difference between price and derivative accuracy, the PINN coordinate transformation, staged anchored losses, the branch/trunk representation, and midpoint price/derivative bounds under parity.
- Distinguished approximation capability from training guarantees and automatic derivatives from accurate Greeks. Clarified that the DeepONet is a finite-parameter solution surrogate, and that strike scaling alone does not enforce homogeneity.
- Added the expert-guide PINN reference and connected the discussion to the option-pricing review and original DeepONet literature. No numerical comparison with unrelated published protocols is claimed.
- Added source-derived price/Greek RMSE tables grouped by evaluation protocol, with an archival appendix for earlier recorded models. Missing values are dashes. Incomplete or provenance-flagged variants are excluded. Neural seed means are means of run-level RMSEs, not ensemble RMSEs.
- Simplified further work and clarified that benchmark results are already observed. Existing results and trained models are unchanged.
- Verified that the current MC convergence budget includes the independent pilot, in both saved metadata and the sampling implementation; its chart was retained.

The current source is `main.tex`, with `tables/rmse_summary.tex`, `tables/rmse_history.tex`, and the referenced figures. `overleaf_bundle.zip` has now been refreshed and supersedes the stale archive noted above. The older assembler and `main_v2` remain historical artifacts and must not overwrite this source.

Main-table sources: `results/common_benchmark/tables/classical_metrics.csv`; `results/pinn/primary/anchored/tables/evaluation_metrics.json`; `results/deeponet/primary/final_test_metrics.json`; and the selected validation/fresh metrics in `results/deeponet/parity_refinement/20260923T155834294516Z/`.

Historical-table sources: `results/pinn/research_variants/study/tables/seed_aggregate.csv`, the DeepONet final-test reference entries, and `results/monte_carlo/tables/monte_carlo_summary.csv` plus `mc_convergence_pooled.csv`. Early MC price RMSEs are pooled using repetition-weighted squared case RMSEs; these should not be compared across different recorded case sets.

The pre-edit LaTeX and PDF are preserved in `archive/before_educational_revision_20260926/`. The local TinyTeX compilation succeeds; the app compiler could not start because its platform directories were unavailable. No new model training or benchmark evaluation was performed.
