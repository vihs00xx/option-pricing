Corrected report (generated 2026-09-23 15:58 UTC). The original report.md in this folder is preserved unchanged; corrections cover terminology (matched update counts), gate-driven conclusions, and explicit exceptions.

# Optimizer refinement replication (corrected description)

Refinement helped validation price accuracy on all three seeds; Adam won two seeds and L-BFGS one under the recorded selection rule. Both refined arms (small-step Adam, full-batch L-BFGS) used 180-second wall budgets per arm; actual optimizer steps and runtime are listed below because equal wall time did not produce equal update counts.

| Seed | Parent price RMSE | Adam price RMSE | L-BFGS price RMSE | Arm selected by the rule |
|---|---:|---:|---:|---|
| 22092026 | 0.182668 | 0.117586 | 0.128508 | adam_small_steps |
| 22092028 | 0.214117 | 0.147621 | 0.15289 | adam_small_steps |

Measured validation price RMSE changes of the selected arm: -35.6%, -31.1%.

Budget accounting (recorded values):

| Seed | Arm | Optimizer updates / closure calls | Runtime (s) | Stop reason |
|---|---|---:|---:|---|
| 22092026 | adam_small_steps | 8597 | 180.0 | wall_budget |
| 22092026 | lbfgs | 150 | 182.3 | wall_budget_checked_between_lbfgs_steps |
| 22092028 | adam_small_steps | 5004 | 180.1 | wall_budget |
| 22092028 | lbfgs | 116 | 183.0 | wall_budget_checked_between_lbfgs_steps |

The verdict wording "L-BFGS is a dependable improvement: False" in the original report stands as measured; the corrected description above states the replication outcome without implying either optimizer is uniformly superior.
