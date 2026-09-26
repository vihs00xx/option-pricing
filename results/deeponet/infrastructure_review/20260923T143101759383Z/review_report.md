# Infrastructure review report — DeepONet experimental corrections

Scope: experimental controls, checkpoint evaluation, safe resumption and report
accuracy for the DeepONet studies in this repository. All corrections were
implemented in order (review items 1–7); this report is item 8. No model
selection was changed: the registered primary model, retained references and
refined candidate are preserved byte-identically. Parity remains a promising
architectural candidate only; the registered primary is NOT replaced and parity,
penalties and mixed sampling are NOT combined into an untested configuration.

## 1. Files changed and reasons

| File | Change | Reason (review item) |
|---|---|---|
| `src/neural/run_metadata.py` | new | run metadata schema v2, file-vs-content hash separation, legacy adapter with actionable failures, atomic JSON writes, completion manifests (3, 4) |
| `experiments/deeponet/run_training_ablation.py` | reworked | shared sampling normalization (2); v2 metadata (3); frozen job specs, atomic manifests, strict resume, recovery, attempt preservation (4); gate-driven reports with exceptions and matched-update-count terminology (5) |
| `experiments/deeponet/evaluate.py` | reworked | dataset resolved from each run's own recorded metadata with kind-correct hash verification; explicit legacy adapter; ordinary and parity checkpoints supported (3) |
| `experiments/deeponet/generate_corrected_reports.py` | new | clearly identified corrected reports beside preserved originals (5) |
| `experiments/deeponet/rerun_sampling_correction.py` | new | corrected sampling rerun with shared scales, verified pilot reuse, frozen protocol, update-count accounting (2, 7) |
| `tests/test_deeponet_infra.py` | new | 12 regression tests (6) |
| `README.md`, `experiments/config/neural/deeponet_model_registry.json` | wording | matched-update-count terminology, mandated replication phrasing, sampling normalization caveat, corrected-report pointers (5) |

Generated artifacts (new output directories): `results/deeponet/sampling_correction/20260923T145946556968Z/`
(corrected sampling study) and `report_corrected.md` beside the preserved
originals in the ablation and optimizer-refinement studies. Historical
checkpoints, datasets and reports were not rewritten.

## 2. Tests run and results

`python -m unittest discover -s tests` — **126 tests, all passing** (114
retained, including the parity, finite-derivative, penalty-gradient and
checkpoint round-trip tests; 12 new). The new tests cover: identical
normalization scales and effective Greek-loss coefficients for both sampling
arms; the standard evaluator on tiny ordinary and parity checkpoint fixtures;
rejection of wrong dataset and wrong checkpoint hashes; rejection of metadata
that cannot verify its dataset; resume rejection of changed settings, scales,
source identity and parent checkpoints; recovery of finished training with
missing evaluation outputs; partial metadata unable to mark a run complete;
atomic write behavior; and report generation from metrics containing both
improvements and regressions (exceptions listed, unconditional claims absent).
All test artifacts use temporary directories.

## 3. Preserved-checkpoint hash verification

`pre_edit_hashes.json` and `post_edit_hashes.json` in this folder. All four
registered checkpoints (primary `e95693bea39c…`, Greek-focused reference
`249c0c0a6190…`, price-only reference `833ef6522c4c…`, refined candidate
`79edbd515b52…`), their training metadata, the frozen dataset (`bb892af8c78d…`)
and its manifest are byte-identical before and after the edits. The registry
file hash changed only because its verdict wording was corrected (item 5); its
recorded `checkpoint_sha256` selection fields match the preserved files
exactly, and `final_test_is_for_evaluation_only` remains true.

## 4. Corrected uniform-versus-mixed results (shared normalization)

Study: `results/deeponet/sampling_correction/20260923T145946556968Z/`. Both arms
use one fixed scale set computed from the uniform arm's training labels and
frozen before training (delta 0.5789, gamma 0.0193611, vega 22.6929, theta
12.1187, rho 36.3526). Validation RMSE per seed and mean ± sample SD:

| Quantity | Uniform (26/27/28) | Mixed (26/27/28) | Uniform mean ± SD | Mixed mean ± SD |
|---|---|---|---:|---:|
| price | 0.1427 / 0.1125 / 0.1624 | 0.1702 / 0.1724 / 0.2250 | 0.1392 ± 0.0251 | 0.1892 ± 0.0310 |
| delta | 0.01317 / 0.01088 / 0.01384 | 0.01367 / 0.01552 / 0.01719 | 0.01263 ± 0.00155 | 0.01546 ± 0.00176 |
| gamma | 0.003184 / 0.002716 / 0.003414 | 0.002286 / 0.003081 / 0.003415 | 0.003105 ± 0.000356 | 0.002927 ± 0.000580 |
| vega | 1.046 / 0.974 / 1.203 | 1.131 / 1.216 / 1.294 | 1.074 ± 0.117 | 1.214 ± 0.082 |
| theta | 1.509 / 1.229 / 1.722 | 1.014 / 1.254 / 1.546 | 1.487 ± 0.247 | 1.271 ± 0.267 |
| rho | 1.372 / 1.028 / 1.280 | 1.563 / 1.555 / 1.949 | 1.227 ± 0.178 | 1.689 ± 0.225 |

## 5. Broad and near-expiry/near-strike errors (focus set, observed benchmark)

Mixed vs uniform change (mean over three seeds):

| Region | price | delta | gamma | vega | theta | rho |
|---|---:|---:|---:|---:|---:|---:|
| all (broad + concentrated) | +6.6% | −14.9% | −48.6% | +2.6% | −57.0% | +26.0% |
| near strike | −23.4% | −26.0% | −53.0% | −18.0% | −65.4% | +11.6% |
| near expiry | −34.6% | −42.8% | −57.0% | −26.8% | −59.8% | −15.4% |
| near strike and expiry | −55.6% | −43.4% | −58.0% | −52.3% | −66.7% | −18.4% |

Validation (broad) cost of mixed: price +35.9%, delta +22.4%, vega +13.0%,
rho +37.7%; gamma −5.7% and theta −14.5% improve.

## 6. Constraint-violation frequency and maximum severity

| Dataset | Constraint | Uniform freq | Mixed freq | Uniform max sev | Mixed max sev |
|---|---|---:|---:|---:|---:|
| validation | price bounds | 0.0931 | 0.0952 | 0.4168 | 0.6207 |
| validation | delta bounds | 0.0775 | 0.0721 | 0.0831 | 0.0460 |
| validation | negative gamma | 0.0506 | 0.0617 | 0.0137 | 0.0140 |
| stress | price bounds | 0.1592 | 0.1587 | 1.639 | 2.772 |
| stress | delta bounds | 0.0864 | 0.0662 | 0.0744 | 0.0740 |
| stress | negative gamma | 0.1478 | 0.1815 | 0.0133 | 0.0220 |

Frequency and maximum severity move differently and are reported separately
everywhere in the corrected outputs.

## 7. Actual training steps and runtime (budgets are matched update counts)

| Run | Updates | Runtime (s) | Stop reason | Premature |
|---|---:|---:|---|---|
| uniform_22092026 | 12,800 | 845.0 | epochs_completed | no |
| uniform_22092027 | 12,800 | 846.9 | epochs_completed | no |
| uniform_22092028 | 12,800 | 848.2 | epochs_completed | no |
| mixed_22092026 | 12,800 | 1284.3 | epochs_completed | no |
| mixed_22092027 | 12,800 | 1287.7 | epochs_completed | no |
| mixed_22092028 | 12,800 | 1289.4 | epochs_completed | no |

All runs completed their full update counts (no premature termination). Equal
update counts did not produce equal wall time: the mixed runs needed ~52% more
wall time per update batch set, which is exactly why budgets are labeled
matched update counts rather than matched compute.

## 8. Which previous conclusions remain supported and which changed

Still supported:
- **Parity by construction**: call-put inconsistency is eliminated exactly and
  most RMSEs improve at matched update counts. Measured exceptions (slightly
  worse stress Rho, small negative-Gamma frequency increase, and the penalty
  and sampling caveats) are listed in the corrected reports. Retained as a
  promising architectural candidate; not promoted to primary.
- **Optimizer replication**: refinement helped validation price accuracy on all
  three seeds; Adam won two seeds and L-BFGS one under the recorded selection
  rule.
- **Target-region gains of concentrated sampling** persist under isolated
  sampling and are LARGER than previously reported: near-strike-and-expiry
  Theta −66.7% (was −56.1%), Gamma −58.0% (was −43.2%); near-expiry Theta
  −59.8%, Gamma −57.0%; near-strike Theta −65.4%, Gamma −53.0%.
- **Penalty conclusion** (from the preserved ablation study, whose penalty and
  parity comparisons used one shared frozen dataset and are unaffected by the
  normalization defect): penalties reduce targeted violations only at an
  accuracy cost; the penalty-weight accuracy gate failed in the rerun and two
  penalty arms fail the acceptability gate.

Changed by the correction (previous results changed sampling AND
normalization jointly):
- The broad-accuracy cost of mixed sampling was mis-stated: under shared scales
  the validation price cost grows from +9.7% to +35.9% and rho from +11.8% to
  +37.7%, while gamma and theta flip from small regressions to improvements.
  Any decision about mixed sampling must use the corrected numbers.
- The focus-set rho regression is much larger than previously measured (+26.0%
  overall vs +1.5%), and mixed increases negative-Gamma violation frequency
  (+22–23%) and price-bound maximum severity (+49–69%) where the joint-change
  result suggested reductions. Trade-offs involving Rho accuracy or shape
  violations are worse than the previous sampling tables showed.

Remaining limitations: the corrected comparison covers three seeds and one
training budget; the evaluation package is an observed benchmark (reused for
direct comparison), so a later independent assessment needs new fresh cases;
the final benchmark was evaluated once in an earlier one-time study
(`results/deeponet/primary/final_test_metrics.json`) and is already observed —
the latest work did not access it again; wall time per update varies with
host state (reported separately from update counts); penalties and parity were
not combined with mixed sampling, and no such combination should be claimed
without a dedicated experiment.
