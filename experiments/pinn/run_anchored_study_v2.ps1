$ErrorActionPreference = "Continue"
$log = Join-Path $PSScriptRoot "..\..\anchored_study_v2.log"
function Log($m) { "{0} {1}" -f (Get-Date -Format o), $m | Out-File $log -Append }

Log "START corrected anchored study (v2)"

# Stage A: dataset build + verification only (training is run with --smoke
# separately so its n256 artifacts cannot contaminate production tables).
Log "STAGE_A dataset build and verification"
python -m experiments.pinn.train_anchored_ablation --dataset-only *>> $log
Log "STAGE_A_DONE exit=$LASTEXITCODE"

# Stage B: one-seed pilot at 8192 price anchors, including the two
# PDE-isolating controls.
Log "STAGE_B pilot: 6 models x 1 seed at 8192 price anchors"
python -m experiments.pinn.train_anchored_ablation --models data_only physics_only hybrid v2 supervised_hybrid_control supervised_v2_control --anchor-counts 8192 --seeds 19102026 *>> $log
Log "STAGE_B_TRAIN_DONE exit=$LASTEXITCODE"

Log "STAGE_B evaluate"
python -m experiments.pinn.evaluate_anchored_ablation *>> $log
Log "STAGE_B_EVAL_DONE exit=$LASTEXITCODE"

# Stage C: three-seed comparison at 8192 price anchors.
Log "STAGE_C three-seed comparison"
python -m experiments.pinn.train_anchored_ablation --anchor-counts 8192 --seeds 19102026 19112026 19122026 *>> $log
python -m experiments.pinn.train_anchored_ablation --models supervised_hybrid_control supervised_v2_control --anchor-counts 8192 --seeds 19112026 19122026 *>> $log
Log "STAGE_C_TRAIN_DONE exit=$LASTEXITCODE"

Log "STAGE_C evaluate"
python -m experiments.pinn.evaluate_anchored_ablation *>> $log
Log "STAGE_C_EVAL_DONE exit=$LASTEXITCODE"

# Stage D: label-efficiency sweep. Greek-label counts scale with the price
# anchors (N // 4, capped) and are reported separately.
Log "STAGE_D label efficiency"
python -m experiments.pinn.train_anchored_ablation --models data_only hybrid v2 supervised_hybrid_control supervised_v2_control --anchor-counts 512 1024 2048 4096 --seeds 19102026 19112026 19122026 *>> $log
Log "STAGE_D_TRAIN_DONE exit=$LASTEXITCODE"

Log "STAGE_D evaluate"
python -m experiments.pinn.evaluate_anchored_ablation *>> $log
Log "STAGE_D_EVAL_DONE exit=$LASTEXITCODE"

Log "DONE corrected anchored study"