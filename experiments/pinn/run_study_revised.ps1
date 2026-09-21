$ErrorActionPreference = "Continue"
$log = Join-Path $PSScriptRoot "..\..\study_run_revised.log"
function Log($m) { "{0} {1}" -f (Get-Date -Format o), $m | Out-File $log -Append }

Log "START revised study"

# Revised: corrected causal scaling/tuned concentration, rebalanced
# constraint categories. Same Adam + L-BFGS budget as the original study.
$seeds = @(19102026, 19112026, 19122026)
$tags = @("v0", "v1", "v2")
$variants = @("constrained", "causal")

for ($i = 0; $i -lt $seeds.Count; $i++) {
    foreach ($v in $variants) {
        Log "TRAIN $v seed=$($seeds[$i]) tag=$($tags[$i])"
        python -m experiments.pinn.train_pinn_variant --variant $v --seed $seeds[$i] --tag $tags[$i] *>> $log
        Log "TRAIN_DONE $v tag=$($tags[$i]) exit=$LASTEXITCODE"
    }
}

Log "EVALUATE"
python -m experiments.pinn.evaluate_pinn_study_multi *>> $log
Log "EVALUATE_DONE exit=$LASTEXITCODE"

Log "DONE revised study"