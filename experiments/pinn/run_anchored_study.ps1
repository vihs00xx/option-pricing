$ErrorActionPreference = "Continue"
$log = Join-Path $PSScriptRoot "..\..\anchored_study.log"
function Log($m) { "{0} {1}" -f (Get-Date -Format o), $m | Out-File $log -Append }

Log "START anchored study"

# The fixed dataset is built once and reused by every model and seed.
Log "STEP 1 dataset"
python -m experiments.pinn.train_anchored_ablation --dataset-only *>> $log
Log "DATASET_DONE exit=$LASTEXITCODE"

# Primary question: does the PDE help beyond supervised interpolation?
Log "STEP 2 core comparison: 4 models x 3 seeds at 8192 anchors"
python -m experiments.pinn.train_anchored_ablation --anchor-counts 8192 *>> $log
Log "CORE_TRAIN_DONE exit=$LASTEXITCODE"

Log "STEP 3 evaluate core"
python -m experiments.pinn.evaluate_anchored_ablation *>> $log
Log "EVAL_DONE exit=$LASTEXITCODE"

# Data-efficiency question: how few labels are needed with the PDE?
Log "STEP 4 label efficiency: data_only, hybrid, v2 at 512..4096 anchors"
python -m experiments.pinn.train_anchored_ablation --models data_only hybrid v2 `
    --anchor-counts 512 1024 2048 4096 *>> $log
Log "LABEL_TRAIN_DONE exit=$LASTEXITCODE"

Log "STEP 5 evaluate all"
python -m experiments.pinn.evaluate_anchored_ablation *>> $log
Log "EVAL_DONE exit=$LASTEXITCODE"

Log "DONE anchored study"